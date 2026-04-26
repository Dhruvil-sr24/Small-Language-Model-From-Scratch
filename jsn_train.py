"""
Author: Dhruv
(W&B logging and boilerplate generated via AI, core logic adapted for my H100 sessions)
"""


"""
train.py
========
Optimised training loop for the Mamba Interpretability project.
Phase 1: Transformer-only (Model A, BaselineTransformer with MLA).

Design goals:
  • Survive 3-hour H100 sessions — resume from exact step, no lost progress
  • WSD (Warmup-Stable-Decay) scheduler synced to data phase transitions
  • Gradient checkpointing to fit 135M model + full activations in H100 80GB
  • Comprehensive W&B logging for learning-purpose analysis
  • bf16 mixed precision + torch.compile for maximum throughput
  • Kaggle output download for pre-sharded data

Usage
─────
  # First session (from scratch):
  python train.py --run_name mla_transformer_50M

  # Resume (any subsequent session):
  python train.py --run_name mla_transformer_50M --resume

  # Override size:
  python train.py --run_name mla_transformer_135M --model_size 135M --resume

  # Download shards from Kaggle first (run once):
  python train.py --download_data --kaggle_dataset dhruvil60/notebook50e0439470

Data layout expected:
  ./data/stable/shard_*.npy   (8B tokens)
  ./data/anneal/shard_*.npy   (2B tokens)
"""

import os
import sys
import math
import time
import json
import argparse
import subprocess
from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint_sequential


# --- 0. OPTIONAL IMPORTS  (W&B — graceful degradation if missing) ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Prevents CUDA memory fragmentation (large reserved-but-unallocated blocks).
# Set BEFORE any CUDA allocations — must be first CUDA-touching line.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True,max_split_size_mb:256")

try:
    import wandb
    # wandb.login(key="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")
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warning] wandb not installed. Run: pip install wandb")


# --- 1. TRAINING CONFIGURATION ---

@dataclass
class TrainConfig:
    # ── Run identity ─────────────────────────────────────────
    run_name:          str   = "mla_transformer_50M"
    project:           str   = "mamba_interp"
    model_size:        str   = "50M"           # "50M" or "135M"
    seed:              int   = 42

    # ── Data ─────────────────────────────────────────────────
    stable_dir:        str   = "./data/tinystories_raw"
    anneal_dir:        str   = "./data/data/anneal"
    seq_len:           int   = 2048

    # ── Batch geometry ───────────────────────────────────────
    # H100 80GB: bf16, grad-ckpt, seq_len=2048
    #   "50M" config (actually 114M with MLA) → batch=32, accum=2 → 131k tok/step
    #   135M → batch_size=8,  grad_accum=8  → 131k tokens/step
    # NEVER use batch_size=64 with grad_accum=1 — causes OOM on backward pass.
    batch_size:        int   = 32
    grad_accum_steps:  int   = 2

    # ── WSD Scheduler ────────────────────────────────────────
    # Total tokens = stable(8B) + anneal(2B) = 10B
    # With batch=16, seq=2048, accum=4: tokens_per_step = 131,072
    # total_steps ≈ 10B / 131072 ≈ 76,294
    # Warmup: 1% of stable phase (~600 steps) — short, model is small
    # Stable: rest of stable phase, constant LR
    # Decay:  entire anneal phase, linear decay to lr_min
    warmup_steps:      int   = 600
    # stable_steps and anneal_steps are computed from actual shard counts
    # at runtime and stored in the checkpoint. Set 0 to auto-detect.
    stable_steps:      int   = 0
    anneal_steps:      int   = 0

    # ── Learning rate ────────────────────────────────────────
    # muP: base LR was tuned at d_model=256; scale as 256/d_model
    # 50M  (d_model=512):  lr_base * (256/512) = 3e-3
    # 135M (d_model=768):  lr_base * (256/768) = 2e-3
    lr_max:            float = 3e-3    # overridden in __post_init__ for 135M
    lr_min:            float = 3e-5    # 1% of lr_max at end of decay
    lr_base_for_mup:   float = 6e-3    # reference LR at d_model=256

    # ── Optimizer (NorMuon / AdamW) ──────────────────────────
    optimizer_name:    str   = "adamw"  # "adamw" or "muon"
    weight_decay:      float = 0.1
    beta1:             float = 0.9
    beta2:             float = 0.95
    grad_clip:         float = 1.0

    # ── Precision + compilation ───────────────────────────────
    dtype:             str   = "bfloat16"   # "bfloat16" | "float16" | "float32"
    compile_model:     bool  = True         # torch.compile — ~20% throughput gain
    grad_checkpoint:   bool  = True         # activation checkpointing

    # ── Checkpointing ────────────────────────────────────────
    ckpt_dir:          str   = "./checkpoints"
    save_every_steps:  int   = 500          # save full checkpoint
    keep_last_n:       int   = 3            # number of checkpoints to keep
    resume:            bool  = False        # auto-detect latest checkpoint

    # ── Early stopping / fine-tuning ──────────────────────────
    max_steps:         int   = 0            # 0 = run all data; >0 = stop after N steps
    finetune_from:     str   = ""           # path to pretrained ckpt (loads weights only)

    # ── Logging ──────────────────────────────────────────────
    log_every_steps:   int   = 10           # W&B + console
    eval_every_steps:  int   = 200          # eval loss on small held-out set
    eval_tokens:       int   = 1_000_000    # ~2k steps worth of eval data

    # ── Kaggle data download ──────────────────────────────────
    kaggle_dataset:    str   = "dhruvil60/notebook50e0439470"
    data_dest:         str   = "./data"

    def __post_init__(self):
        # muP learning rate scaling
        if self.model_size == "135M":
            d_model = 768
        else:
            d_model = 512
        self.lr_max = self.lr_base_for_mup * (256 / d_model)
        self.lr_min = self.lr_max * 0.01

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len * self.grad_accum_steps

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tokens_per_step"] = self.tokens_per_step
        return d


# ---
# 2. WSD LEARNING RATE SCHEDULER
#
#  Three phases, each with a distinct role:
#
#  WARMUP  (steps 0 → warmup_steps)
#    Linear ramp 0 → lr_max.
#    Prevents early gradient explosions from the random init.
#
#  STABLE  (steps warmup_steps → warmup+stable_steps)
#    Constant lr_max throughout.
#    Model sees the broad stable-phase data mixture at full LR.
#    The flat region lets us switch to the anneal data without
#    a confounding LR signal — we can isolate the data effect.
#
#  DECAY   (steps warmup+stable → total_steps)
#    Linear lr_max → lr_min.
#    Occurs EXACTLY while the DataLoader feeds anneal-phase data
#    (Proof-Pile-2 + OpenWebMath).
#    Falling LR forces the model to consolidate what it learns
#    from the hard math data rather than overfit/forget it.
# ---

class WSDScheduler:
    """
    Warmup → Stable → Decay learning rate schedule.
    Framework-agnostic: call .get_lr(step) to compute the current LR,
    then set it manually on the optimizer param groups.
    """

    def __init__(
        self,
        lr_max:        float,
        lr_min:        float,
        warmup_steps:  int,
        stable_steps:  int,
        decay_steps:   int,
    ):
        self.lr_max       = lr_max
        self.lr_min       = lr_min
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps  = decay_steps
        self.total_steps  = warmup_steps + stable_steps + decay_steps

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return self.lr_max * (step + 1) / self.warmup_steps

        elif step < self.warmup_steps + self.stable_steps:
            # Flat stable phase
            return self.lr_max

        else:
            # Linear decay to lr_min
            decay_step = step - self.warmup_steps - self.stable_steps
            frac = min(1.0, decay_step / max(self.decay_steps, 1))
            return self.lr_max - (self.lr_max - self.lr_min) * frac

    def phase_name(self, step: int) -> str:
        if step < self.warmup_steps:
            return "warmup"
        elif step < self.warmup_steps + self.stable_steps:
            return "stable"
        else:
            return "decay"

    def progress(self, step: int) -> float:
        """Overall training progress 0.0 → 1.0."""
        return min(1.0, step / max(self.total_steps, 1))


# ---
# 3. GRADIENT CHECKPOINTING WRAPPER
#
#  Gradient checkpointing trades compute for memory:
#    Normal  : store ALL activations during forward → memory-heavy
#    GC      : discard activations, recompute on backward → ~50% less RAM
#
#  This lets us use batch_size=16 on H100 for 135M — without GC
#  we'd be limited to batch_size≈4 at seq_len=2048.
#
#  We checkpoint at the LAYER granularity (one segment per layer).
#  This is the sweet spot — per-op checkpointing is slower;
#  whole-model checkpointing can OOM on the recompute pass.
# ---

def apply_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Delegates to model_architecture.enable_gradient_checkpointing().

    The old closure-based monkey-patching approach broke torch.compile:
    Dynamo saw a new function object on every recompile attempt
    (guard: ___check_obj_id(orig_fwd, ...)) causing constant recompilation,
    which both tanked MFU (~9.5% instead of ~35%) and prevented actual
    memory savings — giving the worst of both worlds.

    The correct pattern is to set a flag on each block BEFORE compile,
    so Dynamo traces the checkpointed path once and caches it permanently.
    See TransformerBlock._fwd_impl / MLPBlock._fwd_impl in model_architecture.py.
    """
    from model import enable_gradient_checkpointing
    enable_gradient_checkpointing(model)
    return model   # returns model for chaining (interface unchanged)


# --- 4. OPTIMIZER ---

def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """
    AdamW with weight decay applied only to 2D+ parameters (matrices),
    NOT to biases, norms, or embeddings. This follows the standard
    convention from GPT-2 and all subsequent work.

    Parameter groups:
      decay    : all weight matrices  (weight_decay applied)
      no_decay : biases, norms, 1D params, embeddings

    muP note: the LR is already scaled in TrainConfig.__post_init__.
    """
    decay_params    = []
    no_decay_params = []
    decay_names     = []
    no_decay_names  = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Apply decay only to matrices (2D+), not to vectors or scalars
        if param.ndim >= 2:
            decay_params.append(param)
            decay_names.append(name)
        else:
            no_decay_params.append(param)
            no_decay_names.append(name)

    param_groups = [
        {"params": decay_params,    "weight_decay": cfg.weight_decay,  "name": "decay"},
        {"params": no_decay_params, "weight_decay": 0.0,               "name": "no_decay"},
    ]

    print(f"[Optimizer] AdamW  |  decay={len(decay_params)} params  "
          f"no_decay={len(no_decay_params)} params")
    print(f"[Optimizer] lr_max={cfg.lr_max:.2e}  lr_min={cfg.lr_min:.2e}  "
          f"wd={cfg.weight_decay}  β=({cfg.beta1},{cfg.beta2})")

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.lr_max,
        betas=(cfg.beta1, cfg.beta2),
        eps=1e-8,
        fused=True,   # fused kernel — ~15% faster on CUDA
    )
    return optimizer


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    """Update all param groups to the given learning rate."""
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# --- 5. CHECKPOINTING ---

@dataclass
class CheckpointState:
    """Everything needed to resume training exactly where we left off."""
    step:               int
    tokens_seen:        int
    phase:              str    # "stable" or "anneal"
    phase_step:         int    # steps within current phase
    best_eval_loss:     float
    stable_steps:       int    # total stable steps (for scheduler)
    anneal_steps:       int    # total anneal steps (for scheduler)


def save_checkpoint(
    model:      nn.Module,
    optimizer:  torch.optim.Optimizer,
    state:      CheckpointState,
    cfg:        TrainConfig,
):
    """
    Saves:
      checkpoint_NNNNNN.pt  — model weights + optimizer state + training state
      latest.json           — pointer to the most recent checkpoint

    Automatically removes old checkpoints beyond keep_last_n.
    Optimizer state is saved so momentum buffers carry over between sessions
    — without this, the first steps of each resumed session behave as a
    cold start and produce a visible loss spike.
    """
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fname = ckpt_dir / f"checkpoint_{state.step:07d}.pt"

    # torch.save with weights_only=False — we need the full state dict
    payload = {
        "step":           state.step,
        "tokens_seen":    state.tokens_seen,
        "phase":          state.phase,
        "phase_step":     state.phase_step,
        "best_eval_loss": state.best_eval_loss,
        "stable_steps":   state.stable_steps,
        "anneal_steps":   state.anneal_steps,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg":            cfg.to_dict(),
    }
    torch.save(payload, fname)

    # Update latest pointer
    latest = {"path": str(fname), "step": state.step}
    with open(ckpt_dir / "latest.json", "w") as f:
        json.dump(latest, f, indent=2)

    # Rotate old checkpoints
    all_ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"))
    for old in all_ckpts[:-cfg.keep_last_n]:
        old.unlink()
        print(f"[Ckpt] Removed old checkpoint: {old.name}")

    print(f"[Ckpt] Saved → {fname.name}  "
          f"(step={state.step:,}  tokens={state.tokens_seen/1e9:.3f}B)")


def load_checkpoint(
    ckpt_dir: str,
    model:    nn.Module,
    optimizer: torch.optim.Optimizer,
    device:   str,
) -> Optional[CheckpointState]:
    """
    Loads the latest checkpoint if one exists.
    Returns CheckpointState on success, None if no checkpoint found.

    Handles the compiled model case: torch.compile wraps weights under
    _orig_mod — we strip that prefix when loading so checkpoints are
    compatible between compiled and non-compiled runs.
    """
    latest_path = Path(ckpt_dir) / "latest.json"
    if not latest_path.exists():
        return None

    with open(latest_path) as f:
        meta = json.load(f)

    ckpt_path = meta["path"]
    if not Path(ckpt_path).exists():
        print(f"[Ckpt] latest.json points to missing file: {ckpt_path}")
        return None

    print(f"[Ckpt] Loading → {Path(ckpt_path).name}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Strip _orig_mod prefix added by torch.compile
    raw_state = payload["model_state"]
    new_state  = {}
    for k, v in raw_state.items():
        new_state[k.replace("_orig_mod.", "")] = v

    # Load into the inner (unwrapped) model if torch.compile was used
    target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    # Also handle DDP wrapping: DDP -> compile -> actual model
    if hasattr(target_model, "_orig_mod"):
        target_model = target_model._orig_mod
    if hasattr(target_model, "module"):  # DDP wrapper
        target_model = target_model.module

    missing, unexpected = target_model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[Ckpt] Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[Ckpt] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    optimizer.load_state_dict(payload["optimizer_state"])

    state = CheckpointState(
        step           = payload["step"],
        tokens_seen    = payload["tokens_seen"],
        phase          = payload["phase"],
        phase_step     = payload["phase_step"],
        best_eval_loss = payload["best_eval_loss"],
        stable_steps   = payload["stable_steps"],
        anneal_steps   = payload["anneal_steps"],
    )
    print(f"[Ckpt] Resumed from step {state.step:,}  "
          f"({state.tokens_seen/1e9:.3f}B tokens seen, phase={state.phase})")
    return state


# --- 6. DATASET UTILS ---

# def download_kaggle_data(dataset: str, dest: str):
#     """
#     Downloads pre-sharded data from a Kaggle kernel output.

#     Requires:
#       pip install kaggle
#       ~/.kaggle/kaggle.json  (API token from kaggle.com/account)

#     The Kaggle dataset dhruvil60/notebook50e0439470 contains:
#       /stable/shard_*.npy   — 8B token stable-phase shards
#       /anneal/shard_*.npy   — 2B token anneal-phase shards
#     """
#     dest_path = Path(dest)
#     dest_path.mkdir(parents=True, exist_ok=True)

#     # Check if data already exists
#     stable_shards = list((dest_path / "stable").glob("shard_*.npy"))
#     if stable_shards:
#         print(f"[Data] Found {len(stable_shards)} stable shards in {dest}. "
#               f"Skipping download.")
#         return

#     print(f"[Data] Downloading {dataset} → {dest}")
#     cmd = [
#         "kaggle", "kernels", "output",
#         dataset,
#         "-p", dest,
#     ]
#     result = subprocess.run(cmd, capture_output=True, text=True)
#     if result.returncode != 0:
#         print(f"[Data] kaggle download failed:\n{result.stderr}")
#         sys.exit(1)
#     print(f"[Data] Download complete → {dest}")
import sys
import shutil
from pathlib import Path
import kagglehub

def download_kaggle_data(dataset: str, dest: str):
    """
    Downloads pre-sharded data from a Kaggle kernel output.
_
    Requires:
      pip install kagglehub

    The Kaggle dataset dhruvil60/notebook50e0439470 contains:
      /stable/shard_*.npy   — 8B token stable-phase shards
      /anneal/shard_*.npy   — 2B token anneal-phase shards
    """
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True) 
    # Check if data already exists
    stable_shards = list((dest_path / "stable").glob("shard_*.npy"))
    if stable_shards:
        print(f"[Data] Found {len(stable_shards)} stable shards in {dest}. "
              f"Skipping download.")
        return

    print(f"[Data] Downloading {dataset} using kagglehub...")
    try:
        # kagglehub downloads to a central cache and returns the path
        cache_path = kagglehub.dataset_download("dhruvil60/jsn-sharded-data-for-slm")
        
        # Copy the contents from the cache into your target destination folder
        shutil.copytree(cache_path, dest_path, dirs_exist_ok=True)
        
    except Exception as e:
        print(f"[Data] kagglehub download failed:\n{e}")
        sys.exit(1)
        
    print(f"[Data] Download complete → {dest}")


def build_phase_dataloader(
    shard_dir:      str,
    seq_len:        int,
    batch_size:     int,
    shuffle_shards: bool = True,
    num_workers:    int  =8,
    skip_steps:     int  = 0,
) -> Tuple[DataLoader, int]:
    """
    Builds a DataLoader for one training phase.
    Returns (loader, estimated_total_steps).

    skip_steps: number of steps already consumed in this phase —
    the DataLoader will fast-forward past those shards on resume.
    This is approximate (shard-level granularity, not exact step).
    """
    from data_pipeline import ShardedTokenDataset

    ds    = ShardedTokenDataset(shard_dir, seq_len=seq_len, shuffle_shards=shuffle_shards)
    steps = ds.estimated_steps(batch_size)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    return loader, steps


# ---
# 7. WANDB LOGGING
#
#  Metrics logged (comprehensive for learning-purpose analysis):
#
#  Every log_every_steps:
#    train/loss             — cross-entropy on current batch
#    train/perplexity       — exp(loss), more interpretable than bits
#    train/tokens_per_sec   — throughput (GPU efficiency indicator)
#    train/mfu              — Model FLOP Utilisation (% of peak FLOPS)
#    train/lr               — current learning rate
#    train/lr_phase         — "warmup" | "stable" | "decay"
#    train/grad_norm        — pre-clip gradient norm (training stability)
#    train/tokens_seen      — cumulative tokens (x-axis for loss curves)
#    train/step             — raw step counter
#    train/gpu_memory_gb    — peak allocated GPU memory
#
#  Every eval_every_steps:
#    eval/loss              — held-out eval set loss
#    eval/perplexity        — exp(eval_loss)
#    eval/train_eval_gap    — train_loss - eval_loss (overfitting indicator)
#
#  Phase transitions:
#    phase/switch_to_anneal — logged once when we switch to anneal data
#    phase/stable_final_loss
#    phase/anneal_start_loss
#
#  Histograms (every 1000 steps — expensive but valuable):
#    gradients/[layer_name]  — gradient distributions per layer
#    weights/[layer_name]    — weight norm per layer
#    mla/c_kv_norm           — norm of compressed KV latent (interpretability)
#    mla/attn_entropy        — attention entropy (measures sharpness)
# ---

class WandBLogger:
    """
    Thin wrapper around wandb that handles:
      - graceful degradation when wandb is unavailable
      - batching of log calls to reduce API overhead
      - MLA-specific metrics for interpretability monitoring
    """

    def __init__(self, cfg: TrainConfig, model: nn.Module, enabled: bool = True):
        self.enabled  = enabled and WANDB_AVAILABLE
        self.model    = model
        self.cfg      = cfg
        self._pending: Dict = {}

        if self.enabled:
            wandb.init(
                project = cfg.project,
                name    = cfg.run_name,
                config  = cfg.to_dict(),
                resume  = "allow",           # resumes same run on same run_name
                id      = cfg.run_name,      # stable ID = stable run for resume
            )
            # Watch model — logs gradient and weight histograms automatically
            # log_freq=1000: expensive, log infrequently
            wandb.watch(model, log="all", log_freq=1000, log_graph=False)
            print(f"[W&B] Initialised  project={cfg.project}  run={cfg.run_name}")
        else:
            print("[W&B] Disabled (wandb not installed or not requested)")

    def log(self, metrics: dict, step: int, commit: bool = True):
        if not self.enabled:
            return
        wandb.log(metrics, step=step, commit=commit)

    def log_eval(self, eval_loss: float, train_loss: float, step: int):
        if not self.enabled:
            return
        wandb.log({
            "eval/loss":           eval_loss,
            "eval/perplexity":     math.exp(min(eval_loss, 20)),
            "eval/train_eval_gap": train_loss - eval_loss,
        }, step=step)

    def log_phase_transition(self, from_phase: str, to_phase: str,
                              loss: float, step: int):
        if not self.enabled:
            return
        wandb.log({
            f"phase/{from_phase}_final_loss": loss,
            f"phase/switch_to_{to_phase}":    1,
        }, step=step)

    def log_mla_diagnostics(
        self,
        model:         nn.Module,
        step:          int,
        log_histograms: bool = False,
    ):
        """
        Log MLA-specific internals:
          • c_KV norm — if this collapses, the KV bottleneck is dead
          • attention entropy per layer — low = sharp (good for copying tasks)

        These are the most interpretability-relevant metrics during training.
        """
        if not self.enabled:
            return

        from model import TransformerBlock
        metrics = {}
        attn_entropies = []

        for i, layer in enumerate(model.layers):
            if not isinstance(layer, TransformerBlock):
                continue
            attn = layer.attn

            # c_KV norm (populated during forward — detached Tensor or None)
            if attn.last_c_kv is not None:
                ckv_norm = attn.last_c_kv.norm(dim=-1).mean().item()
                metrics[f"mla/c_kv_norm_L{i}"] = ckv_norm

            # Attention entropy (requires store_attn_w=True in forward)
            if attn.last_attn_w is not None:
                # attn_w: (B, H, T, T) — entropy over last dim
                w = attn.last_attn_w.float() + 1e-9
                entropy = -(w * w.log()).sum(-1).mean().item()
                attn_entropies.append(entropy)
                metrics[f"mla/attn_entropy_L{i}"] = entropy

        if attn_entropies:
            metrics["mla/attn_entropy_mean"] = sum(attn_entropies) / len(attn_entropies)

        if metrics:
            wandb.log(metrics, step=step, commit=False)

    def finish(self):
        if self.enabled:
            wandb.finish()


# ---
# 8. MFU (MODEL FLOP UTILISATION)
#
#  MFU = actual_flops_per_sec / peak_hardware_flops_per_sec
#  Gives a hardware-independent efficiency signal.
#  H100 bf16 peak: 989 TFLOPS (SXM5)
#
#  Approximate FLOPs for a Transformer forward+backward pass:
#    ≈ 6 × N × T   where N = num params, T = seq_len
#    (the 6 accounts for forward 2N + backward 4N in backward-dominates rule)
#  With grad checkpointing, add ~33% recomputation:
#    ≈ 8 × N × T
# ---

def estimate_mfu(
    model:            nn.Module,
    tokens_per_sec:   float,
    seq_len:          int,
    grad_checkpoint:  bool = True,
    h100_tflops:      float = 989e12,
) -> float:
    """Returns MFU as a fraction 0.0–1.0."""
    n_params = sum(p.numel() for p in model.parameters())
    flop_multiplier = 8 if grad_checkpoint else 6
    # flops per token = flop_multiplier × n_params
    # tokens_per_sec × flops_per_token = flops_per_sec
    flops_per_sec = tokens_per_sec * flop_multiplier * n_params
    return flops_per_sec / h100_tflops


# --- 9. EVALUATION ---

@torch.no_grad()
def evaluate(
    model:       nn.Module,
    shard_dir:   str,
    seq_len:     int,
    batch_size:  int,
    max_tokens:  int,
    device:      str,
    autocast_ctx,
) -> float:
    """
    Computes cross-entropy loss on a held-out sample from the eval set.
    Uses the first shard of the stable phase as eval data (never seen
    during training if we skip it, but for simplicity we just use a
    fixed number of batches — the shard is large enough that overlap
    is negligible and the loss is meaningful).
    """
    from data_pipeline import ShardedTokenDataset

    model.eval()
    ds = ShardedTokenDataset(
        shard_dir, 
        seq_len=seq_len, 
        shuffle_shards=False, 
        prefix="val_shard_*.npy" # ADD THIS
    )
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4)

    total_loss   = 0.0
    total_tokens = 0
    max_batches  = max_tokens // (batch_size * seq_len)

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast_ctx:
            out  = model(x, targets=y)
            loss = out["loss"]

        total_loss   += loss.item() * x.numel()
        total_tokens += x.numel()

    model.train()
    return total_loss / max(total_tokens, 1)


# --- 10. MAIN TRAINING LOOP ---

def train(cfg: TrainConfig):
    # ── DDP detection ────────────────────────────────────────
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    is_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        is_main = (local_rank == 0)
        if is_main:
            print(f"[DDP] World size={dist.get_world_size()}  local_rank={local_rank}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        local_rank = 0
        is_main = True

    # ── Reproducibility ──────────────────────────────────────
    torch.manual_seed(cfg.seed + local_rank)
    np.random.seed(cfg.seed + local_rank)

    # ── Device + dtype ───────────────────────────────────────
    if is_main:
        print(f"[Setup] Device: {device}  |  dtype: {cfg.dtype}  |  DDP: {is_ddp}")

    ptdtype    = {"bfloat16": torch.bfloat16,
                  "float16":  torch.float16,
                  "float32":  torch.float32}[cfg.dtype]
    autocast_ctx = torch.autocast(device_type="cuda", dtype=ptdtype) \
                   if "cuda" in device else nullcontext()

    # ── Model ────────────────────────────────────────────────
    from model import BaselineTransformer, config_50M, config_135M
    model_cfg = config_50M() if cfg.model_size == "50M" else config_135M()
    # model     = BaselineTransformer(model_cfg).to(device)
    model = BaselineTransformer(model_cfg).to(device=device, dtype=ptdtype)

    # ── Gradient checkpointing ────────────────────────────────
    if cfg.grad_checkpoint:
        model = apply_gradient_checkpointing(model)

    # ── Optimizer ────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)

    # ── Compile (skip for T4 / when --no_compile) ────────────
    if cfg.compile_model and "cuda" in device:
        if is_main:
            print("[Compile] Running torch.compile(max-autotune)...")
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    # ── DDP wrap (after compile, before checkpoint load) ─────
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    # ── Checkpoint resume / fine-tune ───────────────────────
    state = CheckpointState(
        step=0, tokens_seen=0, phase="stable", phase_step=0,
        best_eval_loss=float("inf"), stable_steps=0, anneal_steps=0,
    )
    if cfg.finetune_from:
        # Fine-tune mode: load model weights ONLY (no optimizer state, reset step)
        print(f"[Finetune] Loading model weights from {cfg.finetune_from}")
        ckpt = torch.load(cfg.finetune_from, map_location=device, weights_only=False)
        raw_state = ckpt["model_state"]
        new_state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
        inner_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        missing, unexpected = inner_model.load_state_dict(new_state, strict=False)
        if missing:    print(f"[Finetune] Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected: print(f"[Finetune] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        print(f"[Finetune] Weights loaded. Training from step 0 with fresh optimizer.")
    elif cfg.resume:
        loaded = load_checkpoint(cfg.ckpt_dir, model, optimizer, device)
        if loaded is not None:
            state = loaded
        else:
            print("[Ckpt] No checkpoint found — starting from scratch")

    # ── Dataloaders ──────────────────────────────────────────
    print(f"\n[Data] Building DataLoaders...")
    stable_loader, stable_total_steps = build_phase_dataloader(
        shard_dir      = cfg.stable_dir,
        seq_len        = cfg.seq_len,
        batch_size     = cfg.batch_size,
        shuffle_shards = True,
        num_workers    = 8,
        skip_steps     = state.phase_step if state.phase == "stable" else 0,
    )
    try:
        anneal_loader, anneal_total_steps = build_phase_dataloader(
            shard_dir      = cfg.anneal_dir,
            seq_len        = cfg.seq_len,
            batch_size     = cfg.batch_size,
            shuffle_shards = False,   # ordered during decay
            num_workers    = 8,
            skip_steps     = state.phase_step if state.phase == "anneal" else 0,
        )
    except FileNotFoundError:
        anneal_loader      = None
        anneal_total_steps = 0
        print("[Data] No anneal shards found — anneal phase disabled")

    # Store step counts in state for scheduler reconstruction after resume
    if state.stable_steps == 0:
        state.stable_steps = stable_total_steps
    if state.anneal_steps == 0:
        state.anneal_steps = anneal_total_steps

    # ── max_steps override ────────────────────────────────────
    if cfg.max_steps > 0:
        total_steps = cfg.max_steps
        print(f"[Data] max_steps={total_steps:,} (overriding data-derived total)")
    else:
        total_steps = state.stable_steps + state.anneal_steps
    print(f"[Data] stable={state.stable_steps:,} steps  "
          f"anneal={state.anneal_steps:,} steps  "
          f"total={total_steps:,} steps")
    print(f"[Data] tokens/step={cfg.tokens_per_step:,}  "
          f"≈{total_steps * cfg.tokens_per_step / 1e9:.1f}B total tokens")

    # ── Scheduler ────────────────────────────────────────────
    if cfg.max_steps > 0:
        # With max_steps: warmup → constant LR (no anneal decay)
        scheduler = WSDScheduler(
            lr_max       = cfg.lr_max,
            lr_min       = cfg.lr_min,
            warmup_steps = cfg.warmup_steps,
            stable_steps = total_steps - cfg.warmup_steps,
            decay_steps  = 0,
        )
    else:
        scheduler = WSDScheduler(
            lr_max       = cfg.lr_max,
            lr_min       = cfg.lr_min,
            warmup_steps = cfg.warmup_steps,
            stable_steps = state.stable_steps - cfg.warmup_steps,
            decay_steps  = state.anneal_steps,
        )

    # ── W&B ──────────────────────────────────────────────────
    logger = WandBLogger(cfg, model, enabled=WANDB_AVAILABLE)

    # ── Gradient scaler (float16 only — not needed for bf16) ─
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == "float16"))

    # ── Active loader + iterator ──────────────────────────────
    # Start from the correct phase if resuming mid-training
    if state.phase == "anneal":
        active_loader = anneal_loader
        active_phase  = "anneal"
    else:
        active_loader = stable_loader
        active_phase  = "stable"

    data_iter       = iter(active_loader)
    phase_switched  = (state.phase == "anneal")  # already switched if resuming in anneal

    # ── Training state ────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    step            = state.step
    tokens_seen     = state.tokens_seen
    phase_step      = state.phase_step
    best_eval_loss  = state.best_eval_loss
    accum_loss      = 0.0
    t_step_start    = time.perf_counter()
    recent_losses   = []   # rolling window for smoothed loss display

    print(f"\n[Train] Starting at step {step:,}  phase={active_phase}") if is_main else None
    print(f"[Train] Target: {total_steps:,} steps  "
          f"({(total_steps - step) / max(total_steps, 1)*100:.1f}% remaining)\n") if is_main else None

    # ── Grad scaler + phase-aware anneal switch function ──────
    anneal_switch_logged = phase_switched

    def maybe_switch_to_anneal(current_step: int) -> bool:
        """Returns True if we just switched to anneal phase."""
        nonlocal data_iter, active_phase, phase_switched, phase_step
        nonlocal anneal_switch_logged
        if phase_switched:
            return False
        if current_step >= state.stable_steps:
            print(f"\n[Phase] ─── Switching to ANNEAL data at step {current_step:,} ───")
            active_phase  = "anneal"
            phase_switched = True
            phase_step    = 0
            data_iter     = iter(anneal_loader)
            if not anneal_switch_logged:
                logger.log_phase_transition("stable", "anneal", accum_loss, current_step)
                anneal_switch_logged = True
            return True
        return False

    # ---
    # MAIN LOOP
    # ---
    while step < total_steps:

        # ── Phase switch check ────────────────────────────────
        maybe_switch_to_anneal(step)

        # ── LR update (WSD) ───────────────────────────────────
        lr = scheduler.get_lr(step)
        set_lr(optimizer, lr)

        # ── Gradient accumulation loop ────────────────────────
        # accum_loss = 0.0
        # optimizer.zero_grad(set_to_none=True)

        # for micro_step in range(cfg.grad_accum_steps):
        #     # Fetch next batch — restart iterator if exhausted
        #     try:
        #         x, y = next(data_iter)
        #     except StopIteration:
        #         data_iter = iter(active_loader)
        #         x, y = next(data_iter)

        #     x = x.to(device, non_blocking=True)
        #     y = y.to(device, non_blocking=True)

        #     # Scale loss by accum steps so gradients average (not sum)
        #     with autocast_ctx:
        #         out  = model(x, targets=y)
        #         loss = out["loss"] / cfg.grad_accum_steps

        #     scaler.scale(loss).backward()
        #     accum_loss += loss.item()
        # ── Gradient accumulation loop ────────────────────────
# ── Gradient accumulation loop ────────────────────────
        accum_loss_tensor = torch.zeros(1, device=device, dtype=ptdtype)
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(cfg.grad_accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(active_loader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # --- THE CUDA GRAPHS FIX ---
            # Tell the compiler a new iteration is starting so it doesn't overwrite memory
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            with autocast_ctx:
                out  = model(x, targets=y)
                loss = out["loss"] / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss_tensor += loss.detach() # NO .item() HERE!

        # Move to CPU exactly ONCE per step
        accum_loss = accum_loss_tensor.item()
        # ── Gradient clipping ─────────────────────────────────
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        ).item()

        # ── Optimizer step ────────────────────────────────────
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        step        += 1
        phase_step  += 1
        tokens_seen += cfg.tokens_per_step
        recent_losses.append(accum_loss)
        if len(recent_losses) > 50:
            recent_losses.pop(0)

        # ── Throughput measurement ────────────────────────────
        t_now        = time.perf_counter()
        step_time    = t_now - t_step_start
        t_step_start = t_now
        tok_per_sec  = cfg.tokens_per_step / max(step_time, 1e-6)
        mfu          = estimate_mfu(model, tok_per_sec, cfg.seq_len,
                                    cfg.grad_checkpoint)

        # ── W&B + console logging (rank 0 only in DDP) ────────
        if step % cfg.log_every_steps == 0 and is_main:
            smooth_loss = sum(recent_losses) / len(recent_losses)
            gpu_mem     = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0.0
            lr_phase    = scheduler.phase_name(step)
            progress    = scheduler.progress(step) * 100

            # ── Build single metrics dict ───────────────────────
            # IMPORTANT: all metrics for a given step go into ONE dict
            # passed to a SINGLE wandb.log(commit=True) call.
            # Splitting across multiple wandb.log calls on the same step
            # causes the "step N < current step N+1" monotonicity warning
            # because each commit=True call advances the internal W&B counter.
            metrics = {
                "train/loss":            accum_loss,
                "train/loss_smooth":     smooth_loss,
                "train/perplexity":      math.exp(min(accum_loss, 20)),
                "train/lr":              lr,
                "train/lr_phase":        lr_phase,
                "train/grad_norm":       grad_norm,
                "train/tokens_per_sec":  tok_per_sec,
                "train/mfu_pct":         mfu * 100,
                "train/tokens_seen_B":   tokens_seen / 1e9,
                "train/gpu_memory_gb":   gpu_mem,
                "train/step":            step,
                "train/progress_pct":    progress,
            }

            # ── MLA diagnostics (merged into same dict, same step) ─
            # Only every 5×log_every steps to keep overhead low.
            # store_attn_w runs a second forward — no_grad so no memory spike.
            if step % (cfg.log_every_steps * 5) == 0:
                with torch.no_grad(), autocast_ctx:
                    _ = model(x[:2], targets=y[:2], store_attn_w=True)
                # Collect MLA metrics directly — no separate wandb.log call
                from model import TransformerBlock
                attn_entropies = []
                for i, layer in enumerate(
                    model._orig_mod.layers   # unwrap torch.compile's _orig_mod
                    if hasattr(model, "_orig_mod") else model.layers
                ):
                    if not isinstance(layer, TransformerBlock):
                        continue
                    attn = layer.attn
                    if attn.last_c_kv is not None:
                        metrics[f"mla/c_kv_norm_L{i}"] = (
                            attn.last_c_kv.norm(dim=-1).mean().item()
                        )
                    if attn.last_attn_w is not None:
                        w = attn.last_attn_w.float() + 1e-9
                        entropy = -(w * w.log()).sum(-1).mean().item()
                        attn_entropies.append(entropy)
                        metrics[f"mla/attn_entropy_L{i}"] = entropy
                if attn_entropies:
                    metrics["mla/attn_entropy_mean"] = (
                        sum(attn_entropies) / len(attn_entropies)
                    )

            # ── Single commit=True log call per step ──────────────
            logger.log(metrics, step=step, commit=True)

            # Console output
            eta_steps = total_steps - step
            eta_hours = eta_steps * step_time / 3600
            print(
                f"step {step:7,}/{total_steps:,} | "
                f"{lr_phase:7s} | "
                f"loss {accum_loss:.4f} | "
                f"ppl {math.exp(min(accum_loss,20)):7.2f} | "
                f"lr {lr:.2e} | "
                f"gnorm {grad_norm:.3f} | "
                f"tok/s {tok_per_sec:,.0f} | "
                f"MFU {mfu*100:.1f}% | "
                f"mem {gpu_mem:.1f}GB | "
                f"ETA {eta_hours:.1f}h"
            )

        # ── Evaluation ────────────────────────────────────────
        if step % cfg.eval_every_steps == 0 and is_main:
            eval_loss = evaluate(
                model       = model,
                shard_dir   = cfg.stable_dir,
                seq_len     = cfg.seq_len,
                batch_size  = cfg.batch_size,
                max_tokens  = cfg.eval_tokens,
                device      = device,
                autocast_ctx = autocast_ctx,
            )
            logger.log_eval(eval_loss, accum_loss, step)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                # Save lightweight best model (weights only — for chat.py)
                best_path = Path(cfg.ckpt_dir) / "best_model.pt"
                best_path.parent.mkdir(parents=True, exist_ok=True)
                inner_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({
                    "model_state": inner_model.state_dict(),
                    "cfg":         cfg.to_dict(),
                    "step":        step,
                    "eval_loss":   eval_loss,
                }, best_path)
                print(f"  ★ New best eval loss: {eval_loss:.4f}  "
                      f"(ppl={math.exp(min(eval_loss,20)):.2f})  → saved {best_path}")

            print(f"  [Eval] loss={eval_loss:.4f}  "
                  f"ppl={math.exp(min(eval_loss,20)):.2f}  "
                  f"gap={accum_loss - eval_loss:+.4f}")

        # ── Save checkpoint ───────────────────────────────────
        if step % cfg.save_every_steps == 0 and is_main:
            ckpt_state = CheckpointState(
                step           = step,
                tokens_seen    = tokens_seen,
                phase          = active_phase,
                phase_step     = phase_step,
                best_eval_loss = best_eval_loss,
                stable_steps   = state.stable_steps,
                anneal_steps   = state.anneal_steps,
            )
            save_checkpoint(model, optimizer, ckpt_state, cfg)

    # ── Final checkpoint ──────────────────────────────────────
    print(f"\n[Train] Training complete at step {step:,}")
    ckpt_state = CheckpointState(
        step=step, tokens_seen=tokens_seen, phase=active_phase,
        phase_step=phase_step, best_eval_loss=best_eval_loss,
        stable_steps=state.stable_steps, anneal_steps=state.anneal_steps,
    )
    save_checkpoint(model, optimizer, ckpt_state, cfg)
    logger.finish()
    print(f"[Train] Done. Best eval loss: {best_eval_loss:.4f}  "
          f"(ppl={math.exp(min(best_eval_loss,20)):.2f})")


# --- 11. ENTRY POINT ---

def parse_args() -> Tuple[TrainConfig, bool]:
    parser = argparse.ArgumentParser(
        description="Train the Baseline Transformer (Model A) for Mamba Interp project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Identity ─────────────────────────────────────────────
    parser.add_argument("--run_name",      default="mla_transformer_50M")
    parser.add_argument("--project",       default="mamba_interp")
    parser.add_argument("--model_size",    default="50M", choices=["50M", "135M"])
    parser.add_argument("--seed",          type=int, default=42)

    # ── Data ─────────────────────────────────────────────────
    parser.add_argument("--stable_dir",    default="./data/tinystories_raw")
    parser.add_argument("--anneal_dir",    default="./data/data/anneal")
    parser.add_argument("--download_data", action="store_true",
                        help="Download shards from Kaggle before training")
    parser.add_argument("--kaggle_dataset",default="dhruvil60/notebook50e0439470")
    parser.add_argument("--data_dest",     default="./data")

    # ── Batch ─────────────────────────────────────────────────
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--grad_accum",    type=int, default=2,
                        dest="grad_accum_steps")
    parser.add_argument("--seq_len",       type=int, default=2048)

    # ── Schedule ─────────────────────────────────────────────
    parser.add_argument("--warmup_steps",  type=int, default=600)

    # ── Optimiser ─────────────────────────────────────────────
    parser.add_argument("--lr_max",        type=float, default=0.0,
                        help="0 = auto (muP)")
    parser.add_argument("--weight_decay",  type=float, default=0.1)
    parser.add_argument("--grad_clip",     type=float, default=1.0)

    # ── Precision ─────────────────────────────────────────────
    parser.add_argument("--dtype",         default="bfloat16",
                        choices=["bfloat16","float16","float32"])
    parser.add_argument("--no_compile",    action="store_true")
    parser.add_argument("--no_grad_ckpt",  action="store_true")

    # ── Checkpointing ─────────────────────────────────────────
    parser.add_argument("--ckpt_dir",      default="./checkpoints")
    parser.add_argument("--save_every",    type=int, default=500,
                        dest="save_every_steps")
    parser.add_argument("--resume",        action="store_true",
                        help="Resume from latest checkpoint")

    # ── Early stop / fine-tune ────────────────────────────────
    parser.add_argument("--max_steps",     type=int, default=0,
                        help="Stop after N steps (0=auto from data)")
    parser.add_argument("--finetune_from", type=str, default="",
                        help="Path to pretrained ckpt (loads weights only, resets optimizer)")

    # ── Logging ──────────────────────────────────────────────
    parser.add_argument("--log_every",     type=int, default=10,
                        dest="log_every_steps")
    parser.add_argument("--eval_every",    type=int, default=200,
                        dest="eval_every_steps")

    args = parser.parse_args()

    cfg = TrainConfig(
        run_name         = "run-2",
        project          = "sft-tinystories",
        model_size       = args.model_size,
        seed             = args.seed,
        stable_dir       = args.stable_dir,
        anneal_dir       = args.anneal_dir,
        batch_size       = args.batch_size,
        grad_accum_steps = args.grad_accum_steps,
        seq_len          = args.seq_len,
        warmup_steps     = args.warmup_steps,
        weight_decay     = args.weight_decay,
        grad_clip        = args.grad_clip,
        dtype            = args.dtype,
        compile_model    = not args.no_compile,
        grad_checkpoint  = not args.no_grad_ckpt,
        ckpt_dir         = args.ckpt_dir,
        save_every_steps = args.save_every_steps,
        resume           = args.resume,
        max_steps        = args.max_steps,
        finetune_from    = args.finetune_from,
        log_every_steps  = args.log_every_steps,
        eval_every_steps = args.eval_every_steps,
        kaggle_dataset   = args.kaggle_dataset,
        data_dest        = args.data_dest,
    )

    # Override LR if explicitly provided
    if args.lr_max > 0:
        cfg.lr_max = args.lr_max
        cfg.lr_min = args.lr_max * 0.01

    return cfg, args.download_data


if __name__ == "__main__":
    cfg, do_download = parse_args()

    print("=" * 66)
    print(f" Mamba Interpretability — Training  |  Model A (Transformer)")
    print(f" run  : {cfg.run_name}")
    print(f" size : {cfg.model_size}  |  dtype: {cfg.dtype}")
    print(f" batch: {cfg.batch_size} × accum {cfg.grad_accum_steps} "
          f"= {cfg.tokens_per_step:,} tokens/step")
    print(f" lr   : max={cfg.lr_max:.2e}  min={cfg.lr_min:.2e}  "
          f"(muP scaled from d={256})")
    print(f" ckpt : {cfg.ckpt_dir}  resume={cfg.resume}")
    print("=" * 66)

    if do_download:
        download_kaggle_data(cfg.kaggle_dataset, cfg.data_dest)

    train(cfg)