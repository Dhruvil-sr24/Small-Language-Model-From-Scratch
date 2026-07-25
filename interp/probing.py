"""

Experiment 1: Linear Probing of MLA's c_KV Bottleneck.


Probes tested:
  1. Token identity  - Can c_KV identify what token a position holds?
  2. POS tags         - Does c_KV encode part-of-speech?
  3. Position         - Does c_KV encode absolute position? (expect NO - RoPE handles this)
  4. Is-entity        - Does c_KV distinguish character names from non-names?
  5. Next-token       - Can c_KV predict the next token directly?

For each probe, we train on BOTH c_KV (128-dim) and the residual stream (512-dim),
then compare accuracy. If c_KV matches residual accuracy on feature X, the bottleneck
preserves that feature. If it drops, the bottleneck discards it.

Usage:
  python -m interp.probing --checkpoint ./checkpoints/best_model.pt
  python -m interp.probing --checkpoint ./checkpoints/best_model.pt --device cuda --n_samples 200
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interp.utils import (
    load_model_for_interp,
    collect_activations,
    load_tinystories_samples,
    decode_tokens,
    setup_plot_style,
)


def extract_token_identity_labels(
    input_ids: torch.Tensor, vocab_size: int = 32768
) -> torch.Tensor:
    return input_ids.clone()


def extract_position_labels(input_ids: torch.Tensor) -> torch.Tensor:
    B, T = input_ids.shape
    n_buckets = 16
    bucket_size = max(1, T // n_buckets)
    positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
    return (positions // bucket_size).clamp(max=n_buckets - 1)


def extract_entity_labels(
    input_ids: torch.Tensor,
    tokenizer_path: str = "./tokenizer/tokenizer.json",
) -> torch.Tensor:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)

    names = {
        "Lily",
        "Tom",
        "Sam",
        "Max",
        "Lucy",
        "Mia",
        "Ben",
        "Anna",
        "Tim",
        "Sara",
        "Jack",
        "Emma",
        "Ella",
        "Bob",
        "Amy",
        "Lila",
        "Billy",
        "Molly",
        "Teddy",
        "Kitty",
        "Bunny",
        "Bear",
        "Mommy",
        "Daddy",
        "Grandma",
        "Grandpa",
        "Mom",
        "Dad",
    }

    name_ids = set()
    for name in names:
        enc = tokenizer.encode(name)
        for tid in enc.ids:
            name_ids.add(tid)
        for variant in [name.lower(), f" {name}", f" {name.lower()}"]:
            enc = tokenizer.encode(variant)
            for tid in enc.ids:
                name_ids.add(tid)

    B, T = input_ids.shape
    labels = torch.zeros(B, T, dtype=torch.long, device=input_ids.device)
    for b in range(B):
        for t in range(T):
            if input_ids[b, t].item() in name_ids:
                labels[b, t] = 1

    return labels


def extract_next_token_labels(input_ids: torch.Tensor) -> torch.Tensor:
    B, T = input_ids.shape
    labels = torch.zeros_like(input_ids)
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = 0
    return labels


def extract_simple_pos_labels(
    input_ids: torch.Tensor,
    tokenizer_path: str = "./tokenizer/tokenizer.json",
) -> torch.Tensor:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)

    word_classes = {
        1: ["a", "an", "the", "A", "An", "The"],
        2: [
            "he",
            "she",
            "it",
            "they",
            "him",
            "her",
            "his",
            "He",
            "She",
            "It",
            "They",
            "we",
            "I",
            "you",
            "my",
            "your",
            "their",
            "its",
            "our",
        ],
        3: [
            "in",
            "on",
            "at",
            "to",
            "from",
            "with",
            "by",
            "for",
            "of",
            "up",
            "out",
            "into",
            "over",
            "about",
        ],
        4: [
            "was",
            "is",
            "had",
            "went",
            "said",
            "saw",
            "came",
            "got",
            "gave",
            "took",
            "made",
            "ran",
            "played",
            "looked",
            "wanted",
            "liked",
            "loved",
            "walked",
        ],
        5: [
            "big",
            "small",
            "little",
            "happy",
            "sad",
            "good",
            "bad",
            "new",
            "old",
            "pretty",
            "nice",
            "beautiful",
            "scared",
            "brave",
            "kind",
            "mean",
            "funny",
            "red",
            "blue",
            "green",
            "yellow",
            "bright",
            "dark",
        ],
    }

    id_to_class = {}
    for cls, words in word_classes.items():
        for word in words:
            for variant in [word, f" {word}", f" {word.lower()}"]:
                enc = tokenizer.encode(variant)
                if len(enc.ids) == 1:
                    id_to_class[enc.ids[0]] = cls

    for punct in [".", ",", "!", "?", ";", ":", "'", '"', "-", "(", ")"]:
        enc = tokenizer.encode(punct)
        for tid in enc.ids:
            id_to_class[tid] = 0

    B, T = input_ids.shape
    labels = torch.full((B, T), 6, dtype=torch.long, device=input_ids.device)
    for b in range(B):
        for t in range(T):
            tid = input_ids[b, t].item()
            if tid in id_to_class:
                labels[b, t] = id_to_class[tid]

    return labels


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


def _batched_eval(probe, X, y, batch_size=2048):
    correct = 0
    total = 0
    total_loss = 0.0
    n_batches = 0

    for i in range(0, X.shape[0], batch_size):
        x_batch = X[i : i + batch_size]
        y_batch = y[i : i + batch_size]
        logits = probe(x_batch)
        total_loss += F.cross_entropy(logits, y_batch).item()
        preds = logits.argmax(dim=-1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.shape[0]
        n_batches += 1
        del logits, preds

    acc = correct / max(total, 1)
    avg_loss = total_loss / max(n_batches, 1)
    return acc, avg_loss


def train_probe(
    activations: torch.Tensor,  # (N, D)
    labels: torch.Tensor,  # (N,)
    n_classes: int,
    n_epochs: int = 50,
    lr: float = 1e-3,
    val_split: float = 0.2,
) -> dict:
    N = activations.shape[0]
    D = activations.shape[1]

    valid_mask = labels >= 0
    activations = activations[valid_mask]
    labels = labels[valid_mask]
    N = activations.shape[0]

    if N < 100:
        return {
            "train_acc": 0.0,
            "val_acc": 0.0,
            "train_loss": 99,
            "val_loss": 99,
            "epoch_accs": [],
        }

    perm = torch.randperm(N)
    val_n = int(N * val_split)
    val_idx, train_idx = perm[:val_n], perm[val_n:]

    train_X, train_y = activations[train_idx], labels[train_idx]
    val_X, val_y = activations[val_idx], labels[val_idx]

    mean = train_X.mean(0, keepdim=True)
    std = train_X.std(0, keepdim=True) + 1e-8
    train_X = (train_X - mean) / std
    val_X = (val_X - mean) / std

    probe = LinearProbe(D, n_classes).to(activations.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    epoch_accs = []
    batch_size = min(2048, N)

    for epoch in range(n_epochs):
        probe.train()
        perm_t = torch.randperm(train_X.shape[0])
        total_loss = 0
        n_batches = 0
        for i in range(0, train_X.shape[0], batch_size):
            idx = perm_t[i : i + batch_size]
            logits = probe(train_X[idx])
            loss = F.cross_entropy(logits, train_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            del logits
        scheduler.step()

        probe.eval()
        with torch.no_grad():
            val_acc, val_loss = _batched_eval(probe, val_X, val_y, batch_size)
            epoch_accs.append(val_acc)

    probe.eval()
    with torch.no_grad():
        train_acc, train_loss = _batched_eval(probe, train_X, train_y, batch_size)

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch_accs": epoch_accs,
    }


@torch.no_grad()
def collect_probing_data(
    model,
    input_ids: torch.Tensor,
    tokenizer_path: str,
    target_layers: list = None,
) -> dict:
    if target_layers is None:
        target_layers = [0, 4, 8, 11, 12, 15, 20, 23]

    cache = collect_activations(
        model,
        input_ids,
        store_attn_w=False,
        layers=target_layers,
    )

    # Flatten batch and sequence dims: (B, T, D) → (B*T, D)
    B, T = input_ids.shape

    c_kv_data = {}
    residual_data = {}

    for i, layer_idx in enumerate(target_layers):
        if i < len(cache.c_kv):
            c_kv_data[layer_idx] = cache.c_kv[i].reshape(-1, cache.c_kv[i].shape[-1])
        if i < len(cache.hidden):
            residual_data[layer_idx] = cache.hidden[i].reshape(
                -1, cache.hidden[i].shape[-1]
            )

    flat_ids = input_ids.reshape(-1)
    labels = {
        "token_identity": extract_token_identity_labels(input_ids).reshape(-1),
        "position": extract_position_labels(input_ids).reshape(-1),
        "entity": extract_entity_labels(input_ids, tokenizer_path).reshape(-1),
        "pos_tag": extract_simple_pos_labels(input_ids, tokenizer_path).reshape(-1),
        "next_token": extract_next_token_labels(input_ids).reshape(-1),
    }

    return {
        "c_kv": c_kv_data,
        "residual": residual_data,
        "labels": labels,
        "target_layers": target_layers,
    }


def run_all_probes(probing_data: dict, device: str = "cpu") -> dict:
    target_layers = probing_data["target_layers"]
    labels = probing_data["labels"]

    n_classes = {
        "token_identity": 32768,
        "position": 16,
        "entity": 2,
        "pos_tag": 7,
        "next_token": 32768,
    }

    results = {}

    for probe_name, probe_labels in labels.items():
        results[probe_name] = {"c_kv": {}, "residual": {}}
        nc = n_classes[probe_name]

        for layer_idx in target_layers:
            for repr_type in ["c_kv", "residual"]:
                if layer_idx not in probing_data[repr_type]:
                    continue

                acts = probing_data[repr_type][layer_idx].to(device)
                labs = probe_labels.to(device)

                print(
                    f"  Probing {probe_name:>16s} | {repr_type:>8s} | layer {layer_idx:2d} | "
                    f"acts={acts.shape} | classes={nc}"
                )

                result = train_probe(
                    acts,
                    labs,
                    nc,
                    n_epochs=30 if nc > 1000 else 50,
                    lr=5e-4 if nc > 1000 else 1e-3,
                )
                results[probe_name][repr_type][layer_idx] = result

    return results


def plot_probing_results(results: dict, save_dir: str = "./figures"):
    plt = setup_plot_style()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    probe_names = ["position", "entity", "pos_tag"]
    probe_titles = {
        "position": "Absolute Position (16 bins)",
        "entity": "Is Entity Name? (binary)",
        "pos_tag": "Part-of-Speech (7 classes)",
    }
    fig, axes = plt.subplots(1, len(probe_names), figsize=(5 * len(probe_names), 5))
    if len(probe_names) == 1:
        axes = [axes]

    for ax, probe_name in zip(axes, probe_names):
        layers_ckv = sorted(results[probe_name]["c_kv"].keys())
        layers_res = sorted(results[probe_name]["residual"].keys())

        acc_ckv = [results[probe_name]["c_kv"][l]["val_acc"] for l in layers_ckv]
        acc_res = [results[probe_name]["residual"][l]["val_acc"] for l in layers_res]

        ax.plot(
            layers_ckv,
            acc_ckv,
            "o-",
            label="c_KV (128-dim)",
            color="#E91E63",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            layers_res,
            acc_res,
            "s-",
            label="Residual (512-dim)",
            color="#2196F3",
            linewidth=2,
            markersize=8,
        )

        n_classes = {"position": 16, "entity": 2, "pos_tag": 7}
        chance = 1.0 / n_classes.get(probe_name, 2)
        ax.axhline(
            y=chance,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Chance ({chance:.2f})",
        )

        ax.set_xlabel("Layer")
        ax.set_ylabel("Validation Accuracy")
        ax.set_title(probe_titles.get(probe_name, probe_name))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    fig.suptitle(
        "What Survives the KV Bottleneck?\nc_KV (128-dim) vs Residual Stream (512-dim)",
        fontsize=14,
        y=1.05,
    )
    plt.tight_layout()
    plt.savefig(str(save_dir / "probing_ckv_vs_residual.png"))
    print(f"[Probe] Saved: probing_ckv_vs_residual.png")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))

    for probe_name in probe_names:
        layers = sorted(results[probe_name]["c_kv"].keys())
        ratios = []
        for l in layers:
            ckv_acc = results[probe_name]["c_kv"][l]["val_acc"]
            res_acc = results[probe_name]["residual"][l]["val_acc"]
            ratio = ckv_acc / max(res_acc, 1e-8)
            ratios.append(min(ratio, 2.0))

        ax.plot(
            layers,
            ratios,
            "o-",
            label=probe_titles.get(probe_name, probe_name),
            linewidth=2,
            markersize=7,
        )

    ax.axhline(y=1.0, color="black", linestyle="-", alpha=0.3, linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("c_KV Accuracy / Residual Accuracy")
    ax.set_title(
        "Bottleneck Retention Ratio\n(1.0 = c_KV preserves all info, <1.0 = information lost)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_dir / "probing_retention_ratio.png"))
    print(f"[Probe] Saved: probing_retention_ratio.png")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))

    all_probes = ["position", "entity", "pos_tag", "token_identity", "next_token"]
    available_probes = [p for p in all_probes if p in results]

    summary_layer = 12
    ckv_accs = []
    res_accs = []
    probe_labels = []

    for pname in available_probes:
        if (
            summary_layer in results[pname]["c_kv"]
            and summary_layer in results[pname]["residual"]
        ):
            ckv_accs.append(results[pname]["c_kv"][summary_layer]["val_acc"])
            res_accs.append(results[pname]["residual"][summary_layer]["val_acc"])
            probe_labels.append(probe_titles.get(pname, pname))

    x = np.arange(len(probe_labels))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        ckv_accs,
        width,
        label="c_KV (128-dim)",
        color="#E91E63",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        res_accs,
        width,
        label="Residual (512-dim)",
        color="#2196F3",
        alpha=0.8,
    )

    ax.set_ylabel("Validation Accuracy")
    ax.set_title(f"Probe Accuracy at Layer {summary_layer}: c_KV vs Residual Stream")
    ax.set_xticks(x)
    ax.set_xticklabels(probe_labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.01,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.01,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(str(save_dir / "probing_summary_layer12.png"))
    print(f"[Probe] Saved: probing_summary_layer12.png")
    plt.show()


def print_probing_summary(results: dict):
    """Print text summary of probing results."""
    print("\n" + "=" * 72)
    print("  Linear Probing Results: What Survives the KV Bottleneck?")
    print("=" * 72)

    for probe_name in results:
        print(f"\n  ── {probe_name} ──")
        print(
            f"  {'Layer':>6}  {'c_KV acc':>10}  {'Residual acc':>13}  {'Retention':>10}"
        )
        print(f"  {'─' * 6}  {'─' * 10}  {'─' * 13}  {'─' * 10}")

        layers = sorted(results[probe_name]["c_kv"].keys())
        for l in layers:
            ckv = results[probe_name]["c_kv"].get(l, {}).get("val_acc", 0)
            res = results[probe_name]["residual"].get(l, {}).get("val_acc", 0)
            retention = ckv / max(res, 1e-8) * 100
            print(f"  {l:>6d}  {ckv:>10.4f}  {res:>13.4f}  {retention:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Linear Probing of MLA's c_KV Bottleneck"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="./tokenizer/tokenizer.json")
    parser.add_argument(
        "--n_samples", type=int, default=100, help="Number of TinyStories samples"
    )
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--save_dir", default="./figures")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[0, 4, 8, 11, 12, 15, 20, 23],
        help="Layers to probe",
    )
    args = parser.parse_args()

    model = load_model_for_interp(args.checkpoint, device=args.device)

    print(f"\n[Probe] Loading {args.n_samples} TinyStories samples...")
    input_ids = load_tinystories_samples(
        tokenizer_path=args.tokenizer,
        n_samples=args.n_samples,
        max_len=args.max_len,
        device=args.device,
    )
    print(f"[Probe] Input shape: {input_ids.shape}")
    print(f"[Probe] Target layers: {args.layers}")

    print("\n[Probe] Collecting c_KV and residual activations...")
    probing_data = collect_probing_data(
        model,
        input_ids,
        tokenizer_path=args.tokenizer,
        target_layers=args.layers,
    )

    print("\n[Probe] Training linear probes...")
    results = run_all_probes(probing_data, device=args.device)

    print_probing_summary(results)

    print("\n[Probe] Generating figures...")
    plot_probing_results(results, save_dir=args.save_dir)

    save_path = Path(args.save_dir) / "probing_results.npz"
    flat_results = {}
    for probe_name in results:
        for repr_type in results[probe_name]:
            for layer, res in results[probe_name][repr_type].items():
                key = f"{probe_name}_{repr_type}_layer{layer}"
                flat_results[f"{key}_val_acc"] = res["val_acc"]
                flat_results[f"{key}_train_acc"] = res["train_acc"]
    np.savez(str(save_path), **flat_results)
    print(f"\n[Probe] Results saved to {save_path}")
    print("[Probe]  done!")


if __name__ == "__main__":
    main()
