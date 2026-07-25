"""

Experiment 2: Attention Pattern Analysis & Induction Head Detection in MLA.

Generates:
  - Attention weight heatmaps per head per layer
  - Induction head scores (correlation with shifted-diagonal pattern)
  - Previous-token head scores
  - Head specialization summary

Usage:
  python -m interp.attention_patterns --checkpoint ./checkpoints/best_model.pt
  python -m interp.attention_patterns --checkpoint ./checkpoints/best_model.pt --n_samples 50
"""

import argparse
import torch
import numpy as np
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interp.utils import (
    load_model_for_interp,
    collect_activations,
    load_tinystories_samples,
    setup_plot_style,
)


def score_previous_token_heads(attn_w: torch.Tensor) -> float:
    B, H, T, _ = attn_w.shape
    if T < 2:
        return 0.0
    prev_token_weights = torch.diagonal(attn_w, offset=-1, dim1=-2, dim2=-1)
    return prev_token_weights.mean().item()


def score_induction_heads(
    attn_w: torch.Tensor,
    input_ids: torch.Tensor,
) -> float:
    B, H, T, _ = attn_w.shape
    if T < 4:
        return 0.0

    scores = []
    for b in range(min(B, 4)):
        ids = input_ids[b].tolist()
        for t in range(2, T):
            current_token = ids[t]

            for s in range(t - 1):
                if ids[s] == current_token and s + 1 < T:
                    induction_weight = attn_w[b, :, t, s + 1].mean().item()
                    scores.append(induction_weight)

    if not scores:
        return 0.0
    return np.mean(scores)


def score_positional_heads(attn_w: torch.Tensor) -> dict:
    B, H, T, _ = attn_w.shape

    bos_score = attn_w[:, :, :, 0].mean().item()

    local_sum = 0.0
    count = 0
    for t in range(T):
        lo = max(0, t - 5)
        hi = min(T, t + 6)
        local_sum += attn_w[:, :, t, lo:hi].sum().item()
        count += B * H
    local_score = local_sum / max(count, 1)

    eps = 1e-10
    entropy = -(attn_w * (attn_w + eps).log()).sum(dim=-1).mean().item()
    max_entropy = np.log(T)
    global_score = entropy / max(max_entropy, 1e-10)

    return {
        "bos_score": bos_score,
        "local_score": local_score,
        "global_score": global_score,
    }


@torch.no_grad()
def analyze_all_heads(
    model,
    input_ids: torch.Tensor,
    tokenizer_path: str = "./tokenizer/tokenizer.json",
) -> dict:
    n_layers = len(model.layers)
    n_heads = model.cfg.n_heads

    prev_token_scores = np.zeros((n_layers, n_heads))
    induction_scores = np.zeros((n_layers, n_heads))
    bos_scores = np.zeros((n_layers, n_heads))
    local_scores = np.zeros((n_layers, n_heads))

    print(f"[Attn] Analyzing {n_layers} layers × {n_heads} heads...")

    batch_size = min(8, input_ids.shape[0])
    n_batches = min(4, input_ids.shape[0] // batch_size)

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_ids = input_ids[start:end]

        cache = collect_activations(model, batch_ids, store_attn_w=True)

        for layer_idx, aw in enumerate(cache.attn_w):
            # aw shape: (B, n_heads, T, T)
            for h in range(n_heads):
                head_w = aw[:, h : h + 1, :, :]

                prev_token_scores[layer_idx, h] += score_previous_token_heads(head_w)

                induction_scores[layer_idx, h] += score_induction_heads(
                    head_w, batch_ids
                )

                pos = score_positional_heads(head_w)
                bos_scores[layer_idx, h] += pos["bos_score"]
                local_scores[layer_idx, h] += pos["local_score"]

        print(f"  Batch {batch_idx + 1}/{n_batches} done")

    prev_token_scores /= n_batches
    induction_scores /= n_batches
    bos_scores /= n_batches
    local_scores /= n_batches

    return {
        "prev_token": prev_token_scores,
        "induction": induction_scores,
        "bos": bos_scores,
        "local": local_scores,
        "n_layers": n_layers,
        "n_heads": n_heads,
    }


def plot_head_scores(results: dict, save_dir: str = "./figures"):
    plt = setup_plot_style()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_layers = results["n_layers"]
    n_heads = results["n_heads"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    score_types = [
        ("prev_token", "Previous Token Head Score", "Reds"),
        ("induction", "Induction Head Score", "Purples"),
        ("bos", "BOS Attention Score", "Blues"),
        ("local", "Local Window Score", "Greens"),
    ]

    for ax, (key, title, cmap) in zip(axes.flat, score_types):
        im = ax.imshow(
            results[key],
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            vmin=0,
        )
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(0, n_layers, 4))
        plt.colorbar(im, ax=ax)

    fig.suptitle("MLA Head Specialization Across Layers", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(str(save_dir / "head_specialization_heatmaps.png"))
    print(f"[Attn] Saved: head_specialization_heatmaps.png")
    plt.show()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))

    for layer in range(n_layers):
        for head in range(n_heads):
            ax.scatter(
                results["prev_token"][layer, head],
                results["induction"][layer, head],
                c=[colors[layer]],
                s=40,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
            )

    ax.set_xlabel("Previous Token Head Score")
    ax.set_ylabel("Induction Head Score")
    ax.set_title(
        "Head Classification: Previous-Token vs Induction\n(color = layer depth)"
    )
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, n_layers - 1))
    plt.colorbar(sm, ax=ax, label="Layer")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_dir / "head_classification_scatter.png"))
    print(f"[Attn] Saved: head_classification_scatter.png")
    plt.show()


def plot_sample_attention(
    model,
    input_ids: torch.Tensor,
    layer: int,
    tokenizer_path: str = "./tokenizer/tokenizer.json",
    save_dir: str = "./figures",
):
    """Plot attention weights for a single layer on a single example."""
    plt = setup_plot_style()
    save_dir = Path(save_dir)

    from interp.utils import decode_tokens

    seq_len = min(64, input_ids.shape[1])
    sample = input_ids[:1, :seq_len]

    cache = collect_activations(model, sample, store_attn_w=True)

    if layer >= len(cache.attn_w):
        print(f"[Attn] Layer {layer} not available")
        return

    aw = cache.attn_w[layer][0]  # (n_heads, T, T)
    n_heads = aw.shape[0]

    try:
        tokens = decode_tokens(sample[0, :seq_len], tokenizer_path)
        tokens = [t[:8] for t in tokens]
    except Exception:
        tokens = [str(i) for i in range(seq_len)]

    n_cols = 4
    n_rows = (n_heads + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flat if n_heads > 1 else [axes]

    for h in range(n_heads):
        ax = axes[h]
        im = ax.imshow(
            aw[h, :seq_len, :seq_len].numpy(),
            cmap="Blues",
            aspect="auto",
            vmin=0,
            vmax=0.5,
        )
        ax.set_title(f"Head {h}", fontsize=11)
        if seq_len <= 32:
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels(tokens, fontsize=6)

    for h in range(n_heads, len(list(axes))):
        axes[h].set_visible(False)

    fig.suptitle(f"Attention Weights - Layer {layer}", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(save_dir / f"attention_layer_{layer}.png"))
    print(f"[Attn] Saved: attention_layer_{layer}.png")
    plt.show()


def print_head_summary(results: dict):
    """Print summary of head specialization findings."""
    n_layers = results["n_layers"]
    n_heads = results["n_heads"]

    print("\n" + "=" * 70)
    print("  Attention Head Specialization Summary (MLA Transformer)")
    print("=" * 70)
    print("\n  Top Previous-Token Heads:")
    pt = results["prev_token"]
    top_pt = np.unravel_index(np.argsort(pt.ravel())[-5:], pt.shape)
    for l, h in zip(top_pt[0][::-1], top_pt[1][::-1]):
        print(f"    Layer {l:2d}, Head {h}: {pt[l, h]:.4f}")

    print("\n  Top Induction Heads:")
    ih = results["induction"]
    top_ih = np.unravel_index(np.argsort(ih.ravel())[-5:], ih.shape)
    for l, h in zip(top_ih[0][::-1], top_ih[1][::-1]):
        print(f"    Layer {l:2d}, Head {h}: {ih[l, h]:.4f}")

    print("\n  Top BOS-Attending Heads:")
    bos = results["bos"]
    top_bos = np.unravel_index(np.argsort(bos.ravel())[-3:], bos.shape)
    for l, h in zip(top_bos[0][::-1], top_bos[1][::-1]):
        print(f"    Layer {l:2d}, Head {h}: {bos[l, h]:.4f}")

    print(f"\n  ─── MLA-Specific Observations ───")
    print(f"  In MLA, K and V heads share the c_KV bottleneck (128-dim).")
    print(f"  This means induction heads must encode BOTH 'what to match'")
    print(f"  (K function) and 'what to copy' (V function) through the same")
    print(f"  compressed latent. Any induction heads found demonstrate that")
    print(f"  the c_KV bottleneck is rich enough for this dual purpose.")


def main():
    parser = argparse.ArgumentParser(
        description="Attention Pattern Analysis for MLA Transformer"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="./tokenizer/tokenizer.json")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=32,
        help="Number of TinyStories samples to analyze",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Max sequence length (shorter = faster)",
    )
    parser.add_argument("--save_dir", default="./figures")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--show_layer", type=int, default=12, help="Layer to visualize in detail"
    )
    args = parser.parse_args()

    model = load_model_for_interp(args.checkpoint, device=args.device)

    print(f"\n[Attn] Loading {args.n_samples} TinyStories samples...")
    input_ids = load_tinystories_samples(
        tokenizer_path=args.tokenizer,
        n_samples=args.n_samples,
        max_len=args.max_len,
        device=args.device,
    )
    print(f"[Attn] Input shape: {input_ids.shape}")

    results = analyze_all_heads(model, input_ids, args.tokenizer)

    print_head_summary(results)

    plot_head_scores(results, save_dir=args.save_dir)

    plot_sample_attention(
        model,
        input_ids,
        layer=args.show_layer,
        tokenizer_path=args.tokenizer,
        save_dir=args.save_dir,
    )

    save_path = Path(args.save_dir) / "attention_results.npz"
    np.savez(
        str(save_path),
        prev_token=results["prev_token"],
        induction=results["induction"],
        bos=results["bos"],
        local=results["local"],
    )
    print(f"\n[Attn] Results saved to {save_path}")
    print("[Attn] ✓ Done!")


if __name__ == "__main__":
    main()
