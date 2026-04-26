"""
interp/utils.py
Author: Dhruv
Shared utilities for the mechanistic interpretability experiments.
"""

import sys, os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# Add parent dir so we can import model.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import BaselineTransformer, ModelConfig, config_50M, config_135M, TransformerBlock


# --- 1. MODEL LOADING ---

def load_model_for_interp(
    checkpoint_path: str,
    device: str = "cpu",
    model_size: str = "50M",
) -> BaselineTransformer:
    """
    Load a trained MLA Transformer for interpretability experiments.

    Key differences from training:
      - No torch.compile (we need clean access to internal state)
      - Always eval() mode
      - Strips _orig_mod prefix from compiled checkpoints
      - Returns unwrapped BaselineTransformer

    Args:
        checkpoint_path: path to .pt checkpoint file
        device: "cpu" or "cuda" (CPU is fine for 114M model)
        model_size: "50M" or "135M"

    Returns:
        model in eval mode on the specified device
    """
    print(f"[Interp] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Build model
    cfg = config_50M() if model_size == "50M" else config_135M()
    model = BaselineTransformer(cfg).to(device)

    # Load weights — handle torch.compile's _orig_mod prefix
    raw_state = ckpt.get("model_state", ckpt)
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
    missing, unexpected = model.load_state_dict(clean_state, strict=False)

    if missing:
        print(f"[Interp] WARNING: {len(missing)} missing keys: {missing[:3]}...")
    if unexpected:
        print(f"[Interp] WARNING: {len(unexpected)} unexpected keys: {unexpected[:3]}...")
    if not missing and not unexpected:
        print(f"[Interp] ✓ All {len(clean_state)} keys loaded successfully")

    model.eval()

    step = ckpt.get("step", "?")
    eval_loss = ckpt.get("eval_loss", None)
    info = f"step={step}"
    if eval_loss is not None:
        import math
        info += f"  eval_loss={eval_loss:.4f}  ppl={math.exp(min(eval_loss, 20)):.1f}"
    print(f"[Interp] {model.count_params()/1e6:.1f}M params  ({info})")

    return model


# --- 2. ACTIVATION CACHE ---

@dataclass
class ActivationCache:
    """
    Container for one forward pass worth of activations.

    Attributes:
        c_kv:       List[Tensor] — c_KV latent per layer, shape (B, T, kv_lora_rank)
        c_q:        List[Tensor] — c_Q latent per layer, shape (B, T, q_lora_rank)
        attn_w:     List[Tensor] — attention weights per layer, shape (B, n_heads, T, T)
        hidden:     List[Tensor] — residual stream per layer, shape (B, T, d_model)
        logits:     Tensor — final logits, shape (B, T, vocab_size)
        input_ids:  Tensor — input token IDs
    """
    c_kv:      List[torch.Tensor] = field(default_factory=list)
    c_q:       List[torch.Tensor] = field(default_factory=list)
    attn_w:    List[torch.Tensor] = field(default_factory=list)
    hidden:    List[torch.Tensor] = field(default_factory=list)
    logits:    Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None

    @property
    def n_layers(self) -> int:
        return len(self.c_kv)

    @property
    def device(self) -> str:
        if self.c_kv:
            return str(self.c_kv[0].device)
        return "cpu"


# --- 3. ACTIVATION COLLECTION ---

@torch.no_grad()
def collect_activations(
    model: BaselineTransformer,
    input_ids: torch.Tensor,
    store_attn_w: bool = True,
    layers: Optional[List[int]] = None,
) -> ActivationCache:
    """
    Run a single forward pass and collect all intermediate activations.

    This is the core function for all interpretability experiments.
    It collects c_KV, c_Q, attention weights, and hidden states
    from every (or specified) layers.

    Args:
        model: BaselineTransformer in eval mode
        input_ids: (B, T) token IDs
        store_attn_w: if True, materializes full attention weight matrices.
                      Uses more memory but needed for attention pattern analysis.
                      For 114M model with T=512: ~1GB per batch.
        layers: optional list of layer indices to collect. None = all.

    Returns:
        ActivationCache with all collected activations
    """
    cache = ActivationCache()
    cache.input_ids = input_ids.detach().cpu()

    # Forward pass with interpretability hooks enabled
    out = model(
        input_ids,
        return_all_hidden=True,
        store_attn_w=store_attn_w,
    )

    cache.logits = out["logits"].detach().cpu()

    # Collect per-layer activations
    all_layers = range(len(model.layers))
    target_layers = set(layers) if layers is not None else set(all_layers)

    for i, layer in enumerate(model.layers):
        if i not in target_layers:
            continue

        if isinstance(layer, TransformerBlock):
            attn = layer.attn
            if attn.last_c_kv is not None:
                cache.c_kv.append(attn.last_c_kv.detach().cpu())
            if attn.last_c_q is not None:
                cache.c_q.append(attn.last_c_q.detach().cpu())
            if attn.last_attn_w is not None:
                cache.attn_w.append(attn.last_attn_w.detach().cpu())

    # Hidden states from all_hidden output
    if out.get("all_hidden") is not None:
        for i, h in enumerate(out["all_hidden"]):
            if i in target_layers:
                cache.hidden.append(h.detach().cpu())

    return cache


# --- 4. DATASET HELPERS ---

def load_tinystories_samples(
    tokenizer_path: str = "./tokenizer/tokenizer.json",
    n_samples: int = 100,
    max_len: int = 256,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load and tokenize TinyStories samples for interp experiments.

    Falls back to synthetic prompts if HuggingFace is unavailable.

    Returns:
        (n_samples, max_len) tensor of token IDs
    """
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train",
                          streaming=True)
        stories = []
        for i, ex in enumerate(ds):
            if i >= n_samples:
                break
            stories.append(ex["text"].strip())
    except Exception:
        # Fallback: synthetic story prompts
        print("[Interp] HF not available, using synthetic prompts")
        stories = [
            f"Once upon a time, there was a little {noun} named {name}."
            for noun, name in [
                ("girl", "Lily"), ("boy", "Tom"), ("cat", "Whiskers"),
                ("dog", "Max"), ("bird", "Blue"), ("fish", "Goldie"),
            ] * (n_samples // 6 + 1)
        ][:n_samples]

    # Tokenize and pad/truncate to fixed length
    all_ids = []
    for story in stories:
        enc = tokenizer.encode(story)
        ids = enc.ids[:max_len]
        # Pad with 0 if needed
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        all_ids.append(ids)

    return torch.tensor(all_ids, dtype=torch.long, device=device)


def decode_tokens(
    token_ids,
    tokenizer_path: str = "./tokenizer/tokenizer.json",
) -> List[str]:
    """Decode a list/tensor of token IDs to strings."""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return [tokenizer.decode([tid]) for tid in token_ids]


# --- 5. PLOT HELPERS ---

# My preferred plot style for the paper, mostly generated by AI
def setup_plot_style():
    """Set publication-quality matplotlib defaults."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 12,
        "font.family": "serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    return plt
