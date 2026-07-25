"""
Experiment 4: Singular Value Analysis of MLA Compression Matrices.

Key outputs:
  1. Singular value spectrum per layer (how many dimensions are effectively used)
  2. Effective rank per layer (threshold-based rank estimation)
  3. Cumulative energy plots (how many SVs capture 90/95/99% of variance)
  4. Layer-wise comparison of compression learned by W_DKV vs W_DQ

Usage:
  python -m interp.svd_analysis --checkpoint ./checkpoints/best_model.pt
  python -m interp.svd_analysis --checkpoint ./checkpoints/best_model.pt --save_dir ./figures
"""

import argparse
import torch
import numpy as np
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interp.utils import load_model_for_interp, setup_plot_style
from model import TransformerBlock


def extract_compression_weights(model) -> dict:
    """
    Extract W_DKV and W_DQ weight matrices from all transformer layers.

    W_DKV: (kv_lora_rank, d_model) = (128, 512) — KV compression
    W_DQ:  (q_lora_rank, d_model)  = (256, 512) — Q compression

    Returns dict with keys:
      'W_DKV': list of (128, 512) tensors, one per layer
      'W_DQ':  list of (256, 512) tensors, one per layer
      'W_UK':  list of (n_kv*d_nope, 128) tensors — KV decompression
      'W_UV':  list of (n_kv*head_dim, 128) tensors — V decompression
    """
    weights = {"W_DKV": [], "W_DQ": [], "W_UK": [], "W_UV": []}

    for i, layer in enumerate(model.layers):
        if isinstance(layer, TransformerBlock):
            attn = layer.attn
            weights["W_DKV"].append(attn.W_DKV.weight.detach().cpu().float())
            weights["W_DQ"].append(attn.W_DQ.weight.detach().cpu().float())
            weights["W_UK"].append(attn.W_UK.weight.detach().cpu().float())
            weights["W_UV"].append(attn.W_UV.weight.detach().cpu().float())

    return weights


def compute_svd_spectrum(weight_matrices: list) -> dict:
    n_layers = len(weight_matrices)
    all_svs = []
    effective_ranks = []
    energy_thresholds = {0.90: [], 0.95: [], 0.99: []}
    top_vectors = []

    for i, W in enumerate(weight_matrices):
        # SVD: W = U @ diag(S) @ V^T
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        all_svs.append(S.numpy())

        energy = (S**2).cumsum(0) / (S**2).sum()

        for thresh, lst in energy_thresholds.items():
            dims_needed = (energy < thresh).sum().item() + 1
            lst.append(dims_needed)

        effective_ranks.append(energy_thresholds[0.99][-1])

        top_vectors.append(Vh[:5].numpy())

    return {
        "singular_values": all_svs,
        "effective_rank": np.array(effective_ranks),
        "energy_90": np.array(energy_thresholds[0.90]),
        "energy_95": np.array(energy_thresholds[0.95]),
        "energy_99": np.array(energy_thresholds[0.99]),
        "top_vectors_V": top_vectors,
    }


def plot_sv_spectrum(svd_results: dict, title: str, save_path: str = None):
    """Plot singular value spectrum across all layers (heatmap)."""
    plt = setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    n_layers = len(svd_results["singular_values"])
    max_svs = max(len(s) for s in svd_results["singular_values"])

    sv_matrix = np.zeros((n_layers, max_svs))
    for i, svs in enumerate(svd_results["singular_values"]):
        sv_matrix[i, : len(svs)] = svs

    ax = axes[0]
    im = ax.imshow(
        np.log10(sv_matrix + 1e-10),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Layer")
    ax.set_title(f"{title}\nlog₁₀(σ) Heatmap")
    plt.colorbar(im, ax=ax, label="log₁₀(σ)")

    ax = axes[1]
    layers = np.arange(n_layers)
    ax.bar(
        layers, svd_results["energy_90"], alpha=0.3, label="90% energy", color="#2196F3"
    )
    ax.bar(
        layers, svd_results["energy_95"], alpha=0.3, label="95% energy", color="#FF9800"
    )
    ax.bar(
        layers, svd_results["energy_99"], alpha=0.5, label="99% energy", color="#4CAF50"
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Dimensions Needed")
    ax.set_title(f"{title}\nEffective Rank (Energy Thresholds)")
    ax.legend()
    ax.set_xticks(layers[::2])
    ax.axhline(
        y=max_svs, color="red", linestyle="--", alpha=0.5, label=f"Max dims = {max_svs}"
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[SVD] Saved: {save_path}")
    plt.show()


def plot_sv_curves(svd_results: dict, title: str, save_path: str = None):

    plt = setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    n_layers = len(svd_results["singular_values"])

    show_layers = sorted(set([0, n_layers - 1] + list(range(0, n_layers, 4))))

    cmap = plt.cm.coolwarm
    for idx, layer_i in enumerate(show_layers):
        svs = svd_results["singular_values"][layer_i]
        color = cmap(idx / max(len(show_layers) - 1, 1))
        ax.semilogy(svs, label=f"Layer {layer_i}", color=color, linewidth=1.5)

    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("σ (log scale)")
    ax.set_title(f"{title}\nSingular Value Decay Curves")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[SVD] Saved: {save_path}")
    plt.show()


def plot_dkv_vs_dq_rank(dkv_results: dict, dq_results: dict, save_path: str = None):

    plt = setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    layers = np.arange(len(dkv_results["effective_rank"]))

    ax.plot(
        layers,
        dkv_results["effective_rank"],
        "o-",
        label="W_DKV (512→128, KV compression)",
        color="#E91E63",
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        layers,
        dq_results["effective_rank"],
        "s-",
        label="W_DQ (512→256, Q compression)",
        color="#2196F3",
        linewidth=2,
        markersize=6,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective Rank (99% energy)")
    ax.set_title("KV vs Q Compression: How Many Dimensions Are Actually Used?")
    ax.legend()
    ax.grid(True, alpha=0.3)

    dkv_rank = dkv_results["effective_rank"]
    if dkv_rank[-1] < dkv_rank[0] * 0.8:
        ax.annotate(
            "Late layers use\nfewer dimensions",
            xy=(len(dkv_rank) - 1, dkv_rank[-1]),
            xytext=(len(dkv_rank) - 5, dkv_rank[-1] + 10),
            arrowprops=dict(arrowstyle="->", color="#E91E63"),
            fontsize=10,
            color="#E91E63",
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[SVD] Saved: {save_path}")
    plt.show()


def print_summary(dkv_results: dict, dq_results: dict):

    print("\n" + "=" * 60)
    print("  SVD Analysis Summary: MLA Compression Matrices")
    print("=" * 60)

    print(f"\n  W_DKV (KV compression, 512 → 128):")
    print(
        f"  {'Layer':>6}  {'Eff Rank':>10}  {'90% Energy':>12}  {'Top σ':>8}  {'σ ratio':>10}"
    )
    print(f"  {'─' * 6}  {'─' * 10}  {'─' * 12}  {'─' * 8}  {'─' * 10}")
    for i in range(len(dkv_results["effective_rank"])):
        svs = dkv_results["singular_values"][i]
        ratio = svs[0] / svs[-1] if svs[-1] > 0 else float("inf")
        print(
            f"  {i:>6d}  {dkv_results['effective_rank'][i]:>10d}  "
            f"{dkv_results['energy_90'][i]:>12d}  "
            f"{svs[0]:>8.2f}  {ratio:>10.1f}"
        )

    mean_rank_kv = dkv_results["effective_rank"].mean()
    mean_rank_q = dq_results["effective_rank"].mean()
    print(f"\n  Mean effective rank W_DKV: {mean_rank_kv:.1f} / 128 dims")
    print(f"  Mean effective rank W_DQ:  {mean_rank_q:.1f} / 256 dims")
    print(
        f"\n  -> The model uses ~{mean_rank_kv / 128 * 100:.0f}% of available KV bottleneck capacity"
    )
    print(
        f"  -> The model uses ~{mean_rank_q / 256 * 100:.0f}% of available Q bottleneck capacity"
    )


def main():
    parser = argparse.ArgumentParser(
        description="SVD Analysis of MLA Compression Matrices"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--save_dir", default="./figures", help="Directory to save figures"
    )
    parser.add_argument(
        "--device", default="cpu", help="Device (CPU is fine — no inference needed)"
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_for_interp(args.checkpoint, device=args.device)

    print("\n[SVD] Extracting compression matrices from 24 layers...")
    weights = extract_compression_weights(model)
    print(f"[SVD] W_DKV shape: {weights['W_DKV'][0].shape}")
    print(f"[SVD] W_DQ shape:  {weights['W_DQ'][0].shape}")

    print("[SVD] Computing SVD of W_DKV (KV compression)...")
    dkv_results = compute_svd_spectrum(weights["W_DKV"])

    print("[SVD] Computing SVD of W_DQ (Q compression)...")
    dq_results = compute_svd_spectrum(weights["W_DQ"])

    print_summary(dkv_results, dq_results)

    print("\n[SVD] Generating publication figures...")

    plot_sv_spectrum(
        dkv_results,
        "W_DKV (KV Compression 512→128)",
        save_path=str(save_dir / "svd_wdkv_spectrum.png"),
    )

    plot_sv_curves(
        dkv_results,
        "W_DKV (KV Compression 512→128)",
        save_path=str(save_dir / "svd_wdkv_curves.png"),
    )

    plot_dkv_vs_dq_rank(
        dkv_results,
        dq_results,
        save_path=str(save_dir / "svd_dkv_vs_dq_rank.png"),
    )

    np.savez(
        str(save_dir / "svd_results.npz"),
        dkv_effective_rank=dkv_results["effective_rank"],
        dkv_energy_90=dkv_results["energy_90"],
        dkv_energy_95=dkv_results["energy_95"],
        dkv_energy_99=dkv_results["energy_99"],
        dq_effective_rank=dq_results["effective_rank"],
        dq_energy_90=dq_results["energy_90"],
    )
    print(f"\n[SVD] Raw results saved to {save_dir / 'svd_results.npz'}")


if __name__ == "__main__":
    main()
