import numpy as np

print("-" * 70)
print("  COMPREHENSIVE RESULTS ANALYSIS (500-sample run)")
print("-" * 70)

print("\n\n" + "─" * 70)
print("  EXPERIMENT 4: SVD Analysis of Compression Matrices")
print("─" * 70)

d = np.load("figures/svd_results.npz")
print("Available keys:", list(d.keys()))

dkv = d["dkv_effective_rank"]
dq = d["dq_effective_rank"]
dkv_90 = d["dkv_energy_90"]
dkv_95 = d["dkv_energy_95"]

print("\n  W_DKV effective rank (99% energy threshold):")
print(
    f"  {'Layer':>6}  {'Rank/128':>10}  {'%':>6}  {'90%E dims':>10}  {'95%E dims':>10}"
)
for i in range(len(dkv)):
    print(
        f"  {i:>6d}  {int(dkv[i]):>10d}  {dkv[i] / 128 * 100:>5.1f}%  {int(dkv_90[i]):>10d}  {int(dkv_95[i]):>10d}"
    )

print(f"\n  Summary W_DKV:")
print(f"    Mean rank:  {dkv.mean():.1f} / 128 ({dkv.mean() / 128 * 100:.1f}%)")
print(f"    Median:     {np.median(dkv):.0f}")
print(f"    Std:        {dkv.std():.1f}")
print(f"    Min:        {dkv.min():.0f} at layer {dkv.argmin()}")
print(f"    Max:        {dkv.max():.0f} at layer {dkv.argmax()}")

print(f"\n  Summary W_DQ:")
print(f"    Mean rank:  {dq.mean():.1f} / 256 ({dq.mean() / 256 * 100:.1f}%)")
print(f"    Min:        {dq.min():.0f} at layer {dq.argmin()}")
print(f"    Max:        {dq.max():.0f} at layer {dq.argmax()}")


corr = np.corrcoef(dkv, dq)[0, 1]
print(f"\n  Pearson correlation (DKV rank vs DQ rank): {corr:.4f}")

print(f"\n  90% energy dims (W_DKV):")
print(f"    Mean:   {dkv_90.mean():.1f}")
print(f"    Range:  {dkv_90.min():.0f} - {dkv_90.max():.0f}")


print("\n\n" + "─" * 70)
print("  EXPERIMENT 2: Attention Pattern Taxonomy")
print("─" * 70)

a = np.load("figures/attention_results.npz")
print("Available keys:", list(a.keys()))

pt = a["prev_token"]
ind = a["induction"]
bos = a["bos"]
local = a["local"]

print(f"\n  Grid shape: {pt.shape} (layers x heads)")


print("\n  INDUCTION HEADS (score > 0.15):")
for l in range(pt.shape[0]):
    for h in range(pt.shape[1]):
        if ind[l, h] > 0.15:
            print(f"    Layer {l:2d}, Head {h}: {ind[l, h]:.4f}")

print(f"\n  All induction heads at same layer? ", end="")
induction_layers = set()
for l in range(pt.shape[0]):
    for h in range(pt.shape[1]):
        if ind[l, h] > 0.15:
            induction_layers.add(l)
print(f"Layers with induction: {induction_layers}")


print("\n  PREVIOUS-TOKEN HEADS (score > 0.80):")
for l in range(pt.shape[0]):
    for h in range(pt.shape[1]):
        if pt[l, h] > 0.80:
            print(f"    Layer {l:2d}, Head {h}: {pt[l, h]:.4f}")


print("\n  BOS-SINK HEADS (score > 0.40):")
for l in range(pt.shape[0]):
    for h in range(pt.shape[1]):
        if bos[l, h] > 0.40:
            print(f"    Layer {l:2d}, Head {h}: {bos[l, h]:.4f}")


print("\n\n" + "─" * 70)
print("  EXPERIMENT 1: Linear Probing (500-sample run)")
print("─" * 70)

p = np.load("figures/probing_results.npz")
print("Available keys:", sorted(p.keys())[:20], "...")

probes = ["token_identity", "position", "entity", "pos_tag", "next_token"]
reprs = ["c_kv", "residual"]
layers = [0, 4, 8, 11, 12, 15, 20, 23]

for probe_name in probes:
    print(f"\n  - {probe_name} - ")
    print(
        f"  {'Layer':>6}  {'c_KV val':>10}  {'Resid val':>10}  {'Retention':>10}  {'c_KV train':>11}"
    )
    print(f"  {'─' * 6}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 11}")
    for l in layers:
        ckv_key = f"{probe_name}_c_kv_layer{l}_val_acc"
        res_key = f"{probe_name}_residual_layer{l}_val_acc"
        ckv_train_key = f"{probe_name}_c_kv_layer{l}_train_acc"

        if ckv_key in p and res_key in p:
            ckv_val = float(p[ckv_key])
            res_val = float(p[res_key])
            ckv_train = float(p[ckv_train_key]) if ckv_train_key in p else 0
            retention = ckv_val / max(res_val, 1e-8) * 100
            overfit = ckv_train - ckv_val
            print(
                f"  {l:>6d}  {ckv_val:>10.4f}  {res_val:>10.4f}  {retention:>9.1f}%  {ckv_train:>11.4f}  (gap={overfit:+.3f})"
            )


print("\n\n" + "─" * 70)
print("  EXPERIMENT 3: Activation Patching (100-example run)")
print("─" * 70)

pa = np.load("figures/patching_results.npz")
print("Available keys:", list(pa.keys()))

imp = pa["aggregate_importance"]
n_ex = int(pa["n_examples"])
print(f"  Number of examples traced: {n_ex}")

# Normalize
if imp.max() > 0:
    norm_imp = imp / imp.max()
else:
    norm_imp = imp

print(f"\n  Layer-wise causal importance (normalized):")
for i, val in enumerate(norm_imp):
    bar = "█" * int(val * 40)
    print(f"  Layer {i:2d}:  {val:.4f}  {bar}")

top3 = np.argsort(norm_imp)[-3:][::-1]
bot3 = np.argsort(norm_imp)[:3]
print(
    f"\n  Top 3 most causally important: {list(top3)} (scores: {[f'{norm_imp[l]:.3f}' for l in top3]})"
)
print(
    f"  Bottom 3 least important:     {list(bot3)} (scores: {[f'{norm_imp[l]:.3f}' for l in bot3]})"
)

# Zone analysis
early = norm_imp[:7].mean()
mid = norm_imp[7:13].mean()
late = norm_imp[13:20].mean()
output = norm_imp[20:].mean()
print(f"\n  Zone averages:")
print(f"    Early  (L0-6):   {early:.4f}")
print(f"    Mid    (L7-12):  {mid:.4f}")
print(f"    Late   (L13-19): {late:.4f}")
print(f"    Output (L20-23): {output:.4f}")


print("\n\n" + "─" * 70)
print("  CROSS-EXPERIMENT ANALYSIS")
print("─" * 70)


if len(dkv) == len(norm_imp):
    corr_svd_causal = np.corrcoef(dkv, norm_imp)[0, 1]
    print(f"\n  Correlation (SVD rank vs causal importance): {corr_svd_causal:.4f}")

    print(f"\n  Layer 15 deep dive:")
    print(f"    SVD effective rank:    {int(dkv[15])}/128 ({dkv[15] / 128 * 100:.1f}%)")
    print(f"    SVD 90% energy dims:   {int(dkv_90[15])}")
    print(f"    Causal importance:     {norm_imp[15]:.4f}")
    print(f"    DQ rank:               {int(dq[15])}/256")

    print(f"\n  Layer 12 deep dive (induction layer):")
    print(f"    SVD effective rank:    {int(dkv[12])}/128 ({dkv[12] / 128 * 100:.1f}%)")
    print(f"    Causal importance:     {norm_imp[12]:.4f}")
    print(f"    Induction heads here:  ", end="")
    for h in range(ind.shape[1]):
        if ind[12, h] > 0.15:
            print(f"H{h}({ind[12, h]:.3f}) ", end="")
    print()

    print(f"\n  Layer 7 deep dive (high rank but low causal?):")
    print(f"    SVD effective rank:    {int(dkv[7])}/128 ({dkv[7] / 128 * 100:.1f}%)")
    print(f"    Causal importance:     {norm_imp[7]:.4f}")

print("\n\n" + "=" * 70)
print("  ANALYSIS COMPLETE")
print("=" * 70)
