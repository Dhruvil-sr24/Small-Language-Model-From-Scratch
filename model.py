
''''
model.py 
it is needed to review again for more optimizations.
 '''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Tuple, Dict
from einops import rearrange


# --- 1. CONFIGURATION ---

@dataclass
class ModelConfig:
    # ── Vocabulary / sequence
    vocab_size:       int   = 32_768
    seq_len:          int   = 2048

    # ── Model dimensions
    d_model:          int   = 512
    n_layers:         int   = 24
    d_ff:             int   = 2048       # SwiGLU inner dim

    # ── MLA attention
    #   n_heads      : Q heads  (high — Q compression is cheap)
    #   n_kv_heads   : K/V heads after decompression  (n_heads // 4)
    #   kv_lora_rank : dim of compressed KV latent c_KV  (~d_model // 4)
    #   q_lora_rank  : dim of compressed Q latent c_Q    (~d_model // 2)
    #   qk_rope_dim  : per-head dims that receive RoPE rotation
    n_heads:          int   = 8
    n_kv_heads:       int   = 2
    kv_lora_rank:     int   = 128
    q_lora_rank:      int   = 256
    qk_rope_dim:      int   = 32
    qk_norm_eps:      float = 1e-6

    # ── Regularisation
    dropout:          float = 0.0
    bias:             bool  = False

    # ── muP scaling
    mup_base_width:   int   = 256

    # ── Mamba-2 (Model B only)
    d_state:          int   = 128
    d_conv:           int   = 4
    expand:           int   = 2
    headdim:          int   = 64
    chunk_size:       int   = 256

    # ── RoPE
    rope_base:        int   = 500_000    # LLaMA-3 style long-context base

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, \
            "n_heads must be divisible by n_kv_heads"
        self.head_dim    = self.d_model // self.n_heads
        self.qk_nope_dim = self.head_dim - self.qk_rope_dim   # non-rope per-head dims
        self.kv_groups   = self.n_heads // self.n_kv_heads     # GQA repeat factor


def config_50M() -> ModelConfig:
    return ModelConfig(
        d_model=512, n_layers=24, n_heads=8,  n_kv_heads=2,
        d_ff=2048,   kv_lora_rank=128, q_lora_rank=256, qk_rope_dim=32,
    )

def config_135M() -> ModelConfig:
    return ModelConfig(
        d_model=768, n_layers=24, n_heads=12, n_kv_heads=3,
        d_ff=3072,   kv_lora_rank=192, q_lora_rank=384, qk_rope_dim=32,
    )


# --- 2. PRIMITIVE BUILDING BLOCKS ---

# Using RMSNorm instead of LayerNorm here because it is faster and standard in modern open LLMs.
class RMSNorm(nn.Module):
     
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rsqrt fuses sqrt + reciprocal into one CUDA op
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * norm)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding applied to a fixed-size sub-dimension of Q and K.

    In MLA, only the rope sub-dimension (qk_rope_dim) receives rotation.
    The nope sub-dimension carries content information without positional bias.
    This split means:
      - k_nope can be recomputed from c_KV at any time (position-independent)
      - k_rope must be cached separately (but it is tiny: n_kv_heads * qk_rope_dim)

    LLaMA-3 base=500_000 allows the model to generalise well beyond seq_len
    at inference via RoPE extrapolation / YaRN.
    """
    def __init__(self, rope_dim: int, max_seq: int = 4096, base: int = 500_000):
        super().__init__()
        assert rope_dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq)

    def _build_cache(self, T: int):
        t     = torch.arange(T, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)          # (T, rope_dim/2)
        emb   = torch.cat([freqs, freqs], dim=-1)      # (T, rope_dim)
        self.register_buffer("cos_cache", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cache", emb.sin()[None, None], persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,   # (B, n_heads,    T, rope_dim)
        k: torch.Tensor,   # (B, n_kv_heads, T, rope_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T   = q.size(2)
        cos = self.cos_cache[:, :, :T, :]
        sin = self.sin_cache[:, :, :T, :]
        q   = (q * cos) + (self._rotate_half(q) * sin)
        k   = (k * cos) + (self._rotate_half(k) * sin)
        return q, k


class SwiGLU(nn.Module):
     
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.w_val  = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.w_out  = nn.Linear(cfg.d_ff,    cfg.d_model, bias=cfg.bias)
        self.drop   = nn.Dropout(cfg.dropout)
        # muP output scale
        scale = (cfg.mup_base_width / cfg.d_model) ** 0.5
        nn.init.normal_(self.w_out.weight, std=0.02 * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_out(F.silu(self.w_gate(x)) * self.w_val(x)))


# ---
# 3. MULTI-HEAD LATENT ATTENTION  (MLA)
#
#  Memory layout comparison for our 50M config (d_model=512):
#
#  Standard MHA KV-cache per token per layer:
#    2 × n_heads × head_dim = 2 × 8 × 64 = 1024 floats
#
#  MLA KV-cache per token per layer:
#    c_KV alone  : kv_lora_rank           = 128 floats
#    k_rope alone: n_kv_heads × qk_rope_dim = 2 × 32 = 64 floats
#    Total       :                           192 floats  (81% reduction)
#
#  Full Q and K are NEVER cached — they are recomputed from c_KV / c_Q
#  at each decode step, which is cheap (two small matrix multiplies).
#
#  Interpretability significance of c_KV:
#    c_KV is the unique information bottleneck that determines what the
#    model attends TO at each position. Patching c_KV at layer L, position T
#    cleanly asks: "what does layer L think position T is about?"
#    This is more surgical than patching the residual stream (which carries
#    accumulated information from all previous layers).
# ---

class MLAAttention(nn.Module):
    """
    Multi-head Latent Attention (DeepSeek-V2 style).

    Forward flow:
      Q path: x -[W_DQ]-> c_Q -[W_UQ]-> q_nope  }
                                -[W_QR]-> q_rope  } cat -> q -> QK-norm
                                          RoPE(q_rope)
      K path: x -[W_DKV]-> c_KV -[W_UK]-> k_nope }
              x -[W_KR]->         k_rope           } cat -> k -> QK-norm
                                   RoPE(k_rope)
      V path: c_KV -[W_UV]-> v
      GQA:    repeat k, v  kv_groups times
      Attn:   softmax(q @ k^T / sqrt(d)) @ v
      Res:    + x -[W_VR]->  (value residual — bypasses attention entirely)
      Out:    W_O(concat heads)

    Interpretability attributes populated during forward():
      .last_c_kv   : (B, T, kv_lora_rank) compressed KV latent
      .last_c_q    : (B, T, q_lora_rank)  compressed Q latent
      .last_attn_w : (B, n_heads, T, T)   attention weights (only if store_attn_w=True)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.kv_groups  = cfg.kv_groups
        self.head_dim   = cfg.head_dim
        self.d_nope     = cfg.qk_nope_dim
        self.d_rope     = cfg.qk_rope_dim
        self.kv_rank    = cfg.kv_lora_rank
        self.q_rank     = cfg.q_lora_rank

        # ── Q compression ────────────────────────────────────
        self.W_DQ  = nn.Linear(cfg.d_model, cfg.q_lora_rank,             bias=cfg.bias)
        self.q_ln  = RMSNorm(cfg.q_lora_rank, eps=cfg.qk_norm_eps)
        self.W_UQ  = nn.Linear(cfg.q_lora_rank, cfg.n_heads * self.d_nope, bias=cfg.bias)
        self.W_QR  = nn.Linear(cfg.q_lora_rank, cfg.n_heads * self.d_rope, bias=cfg.bias)

        # ── KV compression ───────────────────────────────────
        self.W_DKV = nn.Linear(cfg.d_model, cfg.kv_lora_rank,            bias=cfg.bias)
        self.kv_ln = RMSNorm(cfg.kv_lora_rank, eps=cfg.qk_norm_eps)
        self.W_UK  = nn.Linear(cfg.kv_lora_rank, cfg.n_kv_heads * self.d_nope, bias=cfg.bias)
        self.W_UV  = nn.Linear(cfg.kv_lora_rank, cfg.n_kv_heads * self.head_dim, bias=cfg.bias)
        # k_rope is position-dependent so it comes directly from x
        self.W_KR  = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_rope, bias=cfg.bias)

        # ── QK-Norm (per-head RMSNorm, post cat(nope, rope)) ─
        self.q_head_norm = RMSNorm(self.head_dim, eps=cfg.qk_norm_eps)
        self.k_head_norm = RMSNorm(self.head_dim, eps=cfg.qk_norm_eps)

        # ── RoPE ─────────────────────────────────────────────
        self.rotary = RotaryEmbedding(cfg.qk_rope_dim, max_seq=cfg.seq_len,
                                       base=cfg.rope_base)

        # ── Value residual ────────────────────────────────────
        # x bypasses the entire attention computation and is added to v.
        # Ensures gradient flows even when attn weights are uniform.
        self.W_VR  = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=cfg.bias)

        # ── Output projection ─────────────────────────────────
        self.W_O   = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=cfg.bias)
        self.drop  = nn.Dropout(cfg.dropout)

        # muP output scale on W_O and W_VR
        scale = (cfg.mup_base_width / cfg.d_model) ** 0.5
        nn.init.normal_(self.W_O.weight,  std=0.02 * scale)
        nn.init.normal_(self.W_VR.weight, std=0.02 * scale)

        # Interpretability state — populated during forward()
        self.last_c_kv:   Optional[torch.Tensor] = None
        self.last_c_q:    Optional[torch.Tensor] = None
        self.last_attn_w: Optional[torch.Tensor] = None

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """GQA expansion: (B, kv_heads, T, D) -> (B, q_heads, T, D)."""
        if n_rep == 1:
            return x
        B, H, T, D = x.shape
        return (x.unsqueeze(2)
                  .expand(B, H, n_rep, T, D)
                  .reshape(B, H * n_rep, T, D))

    def forward(
        self,
        x:            torch.Tensor,      # (B, T, D)
        store_attn_w: bool = False,      # True during interpretability experiments
    ) -> torch.Tensor:
        B, T, D = x.shape

        # ── Q PATH ───────────────────────────────────────────
        c_q    = self.q_ln(self.W_DQ(x))                         # (B, T, q_rank)
        self.last_c_q = c_q.detach()

        q_nope = self.W_UQ(c_q)                                  # (B, T, n_h * d_nope)
        q_rope = self.W_QR(c_q)                                  # (B, T, n_h * d_rope)
        q_nope = rearrange(q_nope, "b t (h d) -> b h t d", h=self.n_heads)
        q_rope = rearrange(q_rope, "b t (h d) -> b h t d", h=self.n_heads)

        # ── KV PATH ──────────────────────────────────────────
        c_kv   = self.kv_ln(self.W_DKV(x))                       # (B, T, kv_rank)
        self.last_c_kv = c_kv.detach()                           # ← interpretability hook

        k_nope = self.W_UK(c_kv)                                 # (B, T, n_kv * d_nope)
        v      = self.W_UV(c_kv)                                 # (B, T, n_kv * head_dim)
        k_rope = self.W_KR(x)                                    # (B, T, n_kv * d_rope)
        k_nope = rearrange(k_nope, "b t (h d) -> b h t d", h=self.n_kv_heads)
        k_rope = rearrange(k_rope, "b t (h d) -> b h t d", h=self.n_kv_heads)
        v      = rearrange(v,      "b t (h d) -> b h t d", h=self.n_kv_heads)

        # ── RoPE (only on the rope sub-dimensions) ────────────
        q_rope, k_rope = self.rotary(q_rope, k_rope)

        # ── Reconstruct full Q and K ──────────────────────────
        q = torch.cat([q_nope, q_rope], dim=-1)                  # (B, n_h, T, head_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)                  # (B, n_kv, T, head_dim)

        # ── QK-Norm (per head, after cat) ─────────────────────
        q = self.q_head_norm(q)
        k = self.k_head_norm(k)

        # ── GQA expansion ────────────────────────────────────
        k = self._repeat_kv(k, self.kv_groups)                   # (B, n_h, T, head_dim)
        v = self._repeat_kv(v, self.kv_groups)                   # (B, n_h, T, head_dim)

        # ── Attention ─────────────────────────────────────────
        if store_attn_w:
            # Manual path: materialise weights for interpretability.
            # Uses more memory but exposes the (B, n_heads, T, T) weight matrix.
            scale   = self.head_dim ** -0.5
            scores  = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask    = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores  = scores.masked_fill(~mask, float("-inf"))
            attn_w  = F.softmax(scores.float(), dim=-1).to(q.dtype)
            self.last_attn_w = attn_w.detach()
            attn_w  = F.dropout(attn_w, p=self.drop.p, training=self.training)
            attn_out = torch.matmul(attn_w, v)
        else:
            # Fast path: FlashAttention-2 via PyTorch SDPA kernel.
            # is_causal=True never materialises the (T x T) mask.
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.drop.p if self.training else 0.0,
                is_causal=True,
            )
            self.last_attn_w = None

        # ── Value residual ────────────────────────────────────
        v_res    = rearrange(self.W_VR(x), "b t (h d) -> b h t d", h=self.n_heads)
        attn_out = attn_out + v_res

        # ── Output projection ─────────────────────────────────
        out = rearrange(attn_out, "b h t d -> b t (h d)")
        return self.drop(self.W_O(out))


# --- 4. BLOCK WRAPPERS ---

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block:
      x = x + MLA(RMSNorm(x))
      x = x + SwiGLU(RMSNorm(x))

    The .attn attribute is MLAAttention — access .attn.last_c_kv for patching.
    Used in Model.

    Gradient checkpointing is implemented INSIDE forward() via a stable
    _fwd_impl method. This is the only pattern torch.compile can trace
    without recompilation — monkey-patching forward() externally causes
    Dynamo to see a different function object every recompile and triggers
    the "check_obj_id" guard failure that kills MFU.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1          = RMSNorm(cfg.d_model)
        self.attn           = MLAAttention(cfg)
        self.norm2          = RMSNorm(cfg.d_model)
        self.mlp            = SwiGLU(cfg)
        self.use_checkpoint = False   # toggled by enable_gradient_checkpointing()

    def _fwd_impl(self, x: torch.Tensor, store_attn_w: bool) -> torch.Tensor:
        """Actual computation — separated so checkpoint() has a stable target."""
        x = x + self.attn(self.norm1(x), store_attn_w=store_attn_w)
        x = x + self.mlp(self.norm2(x))
        return x

    def forward(self, x: torch.Tensor, store_attn_w: bool = False, **kw) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            # use_reentrant=False: required for torch.compile + AMP correctness.
            # Passes store_attn_w as a positional arg — checkpoint() supports
            # non-tensor positional args in use_reentrant=False mode (PyTorch 2.0+).
            return checkpoint(self._fwd_impl, x, store_attn_w, use_reentrant=False)
        return self._fwd_impl(x, store_attn_w)


class MLPBlock(nn.Module):
     
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm           = RMSNorm(cfg.d_model)
        self.mlp            = SwiGLU(cfg)
        self.use_checkpoint = False

    def _fwd_impl(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))

    def forward(self, x: torch.Tensor, **kw) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._fwd_impl, x, use_reentrant=False)
        return self._fwd_impl(x)


# --- 5b. GRADIENT CHECKPOINTING HELPER ---

def enable_gradient_checkpointing(model: nn.Module) -> None:
   
    # from model_architecture import TransformerBlock, MLPBlock
    n = 0
    for module in model.modules():
        if isinstance(module, (TransformerBlock, MLPBlock)):
            module.use_checkpoint = True
            n += 1
    print(f"[GradCkpt] Enabled on {n} blocks (TransformerBlock + MLPBlock)")



# --- 7. MODEL  — BASELINE TRANSFORMER  (MLA) ---

class BaselineTransformer(nn.Module):
    """
    Model : all-MLA decoder-only Transformer.

    Every layer is a TransformerBlock (MLA + SwiGLU). 

    Design choice — MLA over standard MHA:
      1. Smaller KV cache: 81% fewer floats per token at inference.
      2. c_KV is the ideal causal patching target — it is the minimal
         vector that determines all K and V heads for a given position.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg     = cfg
        self.embed   = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)
        self.layers  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm    = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight      # weight tying
        self.apply(self._init_weights)
        kv_mha = 2 * cfg.n_heads * cfg.head_dim
        kv_mla = cfg.kv_lora_rank + cfg.n_kv_heads * cfg.qk_rope_dim
        print(f"[Model A] BaselineTransformer (MLA)  params: {self.count_params()/1e6:.1f}M")
        print(f"         KV-cache/token/layer: MHA={kv_mha} → MLA={kv_mla} "
              f"({(1-kv_mla/kv_mha)*100:.0f}% reduction)")

    # TODO: maybe check if we need to init bias differently?
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            scale = (self.cfg.mup_base_width / self.cfg.d_model) ** 0.5
            nn.init.normal_(m.weight, mean=0.0, std=0.02 * scale)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def get_layer(self, i: int) -> TransformerBlock:
        return self.layers[i]

    def forward(
        self,
        input_ids:         torch.Tensor,
        targets:           Optional[torch.Tensor] = None,
        return_all_hidden: bool = False,
        store_attn_w:      bool = False,
    ) -> Dict:
        x          = self.drop(self.embed(input_ids))
        all_hidden = [] if return_all_hidden else None

        for layer in self.layers:
            x = layer(x, store_attn_w=store_attn_w)
            if return_all_hidden:
                all_hidden.append(x.detach())

        x      = self.norm(x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1), ignore_index=-1,
            )
        return {"logits": logits, "loss": loss, "hidden": x, "all_hidden": all_hidden}

    @torch.no_grad()
    def generate(
        self,
        input_ids:      torch.Tensor,      # (1, T)
        max_new_tokens: int   = 256,
        temperature:    float = 0.7,
        top_k:          int   = 50,
        top_p:          float = 0.9,
        eos_token_id:   Optional[int] = None,
    ) -> torch.Tensor:
      
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Crop to seq_len if context exceeds model capacity
            idx = generated if generated.size(1) <= self.cfg.seq_len \
                  else generated[:, -self.cfg.seq_len:]

            out    = self(idx)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-8)

            # ── Top-k filtering ──────────────────────────────
            if top_k > 0:
                topk_vals = torch.topk(logits, min(top_k, logits.size(-1)))[0]
                logits[logits < topk_vals[:, -1:]] = float("-inf")

            # ── Top-p (nucleus) filtering ─────────────────────
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0]  = False
                indices_to_remove = remove.scatter(
                    dim=-1, index=sorted_idx, src=remove,
                )
                logits[indices_to_remove] = float("-inf")

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated  = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --- 9. FACTORY + UTILITIES ---

def build_model(
    model_type: Literal["transformer"],
    size:       Literal["50M", "135M"] = "50M",
    cfg:        Optional[ModelConfig]  = None,
) -> nn.Module:
    if cfg is None:
        cfg = config_50M() if size == "50M" else config_135M()
    if model_type == "transformer":
        return BaselineTransformer(cfg) 
    raise ValueError(f"Unknown model_type '{model_type}'")


def model_summary(model: nn.Module) -> str:
    total = 0
    rows  = [f"\n{'Module':<52} {'Params':>12}", "─" * 66]
    for name, mod in model.named_modules():
        if list(mod.children()):
            continue
        p = sum(x.numel() for x in mod.parameters())
        if p > 0:
            rows.append(f"{name:<52} {p:>12,}")
            total += p
    rows += ["─" * 66, f"{'TOTAL':<52} {total:>12,}"]
    return "\n".join(rows)


def verify_forward_pass(model: nn.Module, cfg: ModelConfig, device: str = "cpu"):

    model = model.to(device).eval()
    B, T  = 2, min(64, cfg.seq_len)
    x     = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    y     = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    with torch.no_grad():
        out = model(x, targets=y, return_all_hidden=True, store_attn_w=True)

    # Basic output checks
    assert out["logits"].shape == (B, T, cfg.vocab_size)
    assert out["loss"] is not None and out["loss"].item() > 0
    assert len(out["all_hidden"]) == cfg.n_layers

    print(f"   logits      {tuple(out['logits'].shape)}")
    print(f"   loss        {out['loss'].item():.4f}")
    print(f"   all_hidden  {cfg.n_layers} layers × {tuple(out['all_hidden'][0].shape)}")

    # MLA interpretability hook checks
    first_attn = None
    for layer in model.layers:
        if isinstance(layer, TransformerBlock):
            first_attn = layer.attn; break

    if first_attn is not None:
        assert first_attn.last_c_kv is not None
        assert first_attn.last_c_kv.shape == (B, T, cfg.kv_lora_rank), \
            f"c_KV shape: {first_attn.last_c_kv.shape}"
        assert first_attn.last_c_q is not None
        assert first_attn.last_c_q.shape  == (B, T, cfg.q_lora_rank)
        assert first_attn.last_attn_w is not None
        assert first_attn.last_attn_w.shape == (B, cfg.n_heads, T, T)
        print(f"   c_KV latent {tuple(first_attn.last_c_kv.shape)}   (interp hook OK)")
        print(f"   c_Q  latent {tuple(first_attn.last_c_q.shape)}")
        print(f"   attn_w      {tuple(first_attn.last_attn_w.shape)}")


# --- 10. ENTRY POINT ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=["transformer"], default="both")
    parser.add_argument("--size",   choices=["50M", "135M"], default="50M")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    targets = ["transformer"] if args.model == "both" else [args.model]
    for mtype in targets:
        print(f"\n{'='*66}\n Building: {mtype.upper()}  ({args.size})\n{'='*66}")
        cfg   = config_50M() if args.size == "50M" else config_135M()
        model = build_model(mtype, cfg=cfg)
        print(model_summary(model))
        if args.verify:
            print(f"\n[Verify] Running forward pass (CPU, T=64)...")
            verify_forward_pass(model, cfg)
