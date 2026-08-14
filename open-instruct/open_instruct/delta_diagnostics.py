"""
Diagnose whether checkpoint deltas reflect task-specific learning or a scale artifact.

Usage:
    python delta_diagnostics.py /path/base /path/ifeval /path/mmlu

Assumes HF-format checkpoints loadable via safetensors. Adjust `load_state_dict`
if yours are sharded .bin files.
"""

import sys
import re
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors.torch import load_file


def load_state_dict(path):
    """Load a possibly-sharded safetensors checkpoint into one dict."""
    path = Path(path)
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors under {path}")
    sd = {}
    for shard in shards:
        sd.update(load_file(str(shard)))
    return sd


def effective_rank(delta, energy=0.99, max_dim=4096):
    """Number of singular values needed to capture `energy` of the Frobenius norm.

    A merged LoRA of rank r plateaus near r. Full finetuning or isotropic noise
    does not -- noise in particular spreads energy across ~all directions.
    """
    d = delta.float()
    # subsample rows/cols for very large tensors; SVD is the bottleneck
    if d.shape[0] > max_dim:
        d = d[torch.randperm(d.shape[0])[:max_dim]]
    if d.shape[1] > max_dim:
        d = d[:, torch.randperm(d.shape[1])[:max_dim]]
    s = torch.linalg.svdvals(d)
    cum = torch.cumsum(s**2, 0) / (s**2).sum()
    k = int((cum < energy).sum().item()) + 1
    return k, len(s), s.numpy()


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def analyze(base, mid, end, do_svd=True):
    rows, spectra = [], {}
    for name in base:
        if base[name].ndim != 2:
            continue  # skip norms, biases
        W = base[name].float()
        d1 = (mid[name].float() - W)
        d2 = (end[name].float() - mid[name].float())

        rms_w = W.pow(2).mean().sqrt().item()
        rel1 = (d1.norm() / W.norm()).item()
        rel2 = (d2.norm() / mid[name].float().norm()).item()

        row = {
            "tensor": name,
            "rms_W": rms_w,
            # the metric your plots ranked on
            "rel_d1": rel1,
            "rel_d2": rel2,
            # divide the scale out: if this flattens the ranking, the original
            # ranking was measuring 1/RMS(W)
            "rel_d1_scalefree": rel1 * rms_w,
            "rel_d2_scalefree": rel2 * rms_w,
            # ~1 => update just rescales existing directions, no new structure
            "cos_W_d1": cos(W, d1),
            # ~0 => independent runs; ~1 => same perturbation applied twice
            "cos_d1_d2": cos(d1, d2),
        }

        if do_svd:
            k1, n1, s1 = effective_rank(d1)
            # flat spectrum (k/n near the 0.99 line) is the noise signature
            row["effrank_d1"] = k1
            row["effrank_frac_d1"] = k1 / n1
            spectra[name] = s1

        rows.append(row)

    return pd.DataFrame(rows), spectra


def layer_profile(df, out="layer_profile.png"):
    """rel delta vs depth, one line per projection type."""
    pat = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.weight")
    recs = []
    for _, r in df.iterrows():
        m = pat.search(r["tensor"])
        if m:
            recs.append({"layer": int(m.group(1)), "proj": m.group(2),
                         "rel_d1": r["rel_d1"], "rel_d2": r["rel_d2"]})
    if not recs:
        return
    prof = pd.DataFrame(recs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, col, title in zip(axes, ["rel_d1", "rel_d2"],
                              ["base -> ifeval", "ifeval -> mmlu"]):
        for proj, g in prof.groupby("proj"):
            g = g.sort_values("layer")
            ax.plot(g["layer"], g[col], marker="o", ms=3, label=proj)
        ax.set_xlabel("layer"); ax.set_title(title)
    axes[0].set_ylabel(r"$\|\Delta W\| / \|W\|$")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Update magnitude by depth and projection type")
    fig.tight_layout(); fig.savefig(out, dpi=150)


def spectrum_plot(spectra, names, out="spectra.png"):
    """Normalized singular value spectra. Sharp knee => low-rank (merged LoRA).
    Slow decay => full finetune. Near-flat => isotropic noise."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for n in names:
        if n not in spectra:
            continue
        s = spectra[n]
        ax.plot(np.arange(1, len(s) + 1), s / s[0], lw=1, label=n.split("model.")[-1])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("singular value index"); ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_title(r"Spectrum of $\Delta W$ (base $\to$ ifeval)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=150)


if __name__ == "__main__":
    base, mid, end = (load_state_dict(p) for p in sys.argv[1:4])
    df, spectra = analyze(base, mid, end)

    print("\n=== ranked by rel_d1 (reproduces your plot's ordering) ===")
    print(df.nlargest(10, "rel_d1")[
        ["tensor", "rel_d1", "rms_W", "rel_d1_scalefree", "cos_d1_d2"]
    ].to_string(index=False))

    print("\n=== ranked by scale-free delta ===")
    print(df.nlargest(10, "rel_d1_scalefree")[
        ["tensor", "rel_d1_scalefree", "rel_d1", "effrank_frac_d1"]
    ].to_string(index=False))

    print(f"\nrel_d1 spread:  min={df.rel_d1.min():.4f}  max={df.rel_d1.max():.4f}  "
          f"ratio={df.rel_d1.max()/df.rel_d1.min():.2f}")
    print(f"corr(rel_d1, 1/rms_W) = {np.corrcoef(df.rel_d1, 1/df.rms_W)[0,1]:.3f}"
          "   <- near 1.0 means the ranking is a scale artifact")
    print(f"median cos(d1, d2)    = {df.cos_d1_d2.median():.3f}"
          "   <- ~0 independent, ~1 identical perturbation")
    print(f"median effrank frac   = {df.effrank_frac_d1.median():.3f}"
          "   <- near 1.0 means no low-rank structure (noise-like)")

    df.to_csv("delta_diagnostics.csv", index=False)
    layer_profile(df)
    spectrum_plot(spectra, df.nlargest(5, "rel_d1").tensor.tolist())
