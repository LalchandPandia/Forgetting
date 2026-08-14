#!/usr/bin/env python
"""
Per-parameter weight-change scatter plots across a sequential fine-tuning chain.

Companion to cka_drift.py (representational drift) -- this looks at the WEIGHTS
themselves rather than activations, to answer "did specific parameters change a
lot at one particular stage, and could that explain a benchmark score drop at
that stage?"

For each consecutive pair in --chain, loads both checkpoints' state dicts and,
per parameter tensor, computes the relative L2 change ||w_curr - w_prev|| / ||w_prev||.
Produces:
  1. layer_rel_change_by_stage.png -- one point per parameter tensor (embedding ->
     layers -> head, left to right), one color per stage transition, log-scale y.
     Lets you see whether one stage moved specific tensors much more than the others.
  2. param_changes_<a>_to_<b>.csv -- every tensor's rel/abs change for that
     transition, sorted worst-first.
  3. before_after_top_params_<a>_to_<b>.png -- ONE of these per consecutive
     transition in --chain (base->ifeval, ifeval->mmlu, mmlu->gsm8k, ...), each
     showing the top-K most-changed tensors FOR THAT STAGE ONLY (its own before
     vs. its own after -- never base vs. the final checkpoint). A scatter of
     sampled (w_prev, w_curr) element pairs against the y=x line -- shows whether
     the change is spread evenly across the tensor or concentrated in a subset
     of weights. Use --highlight to restrict this to specific transitions instead
     of all of them.

Usage:
    python weight_scatter.py \
        --ckpt base=Qwen/Qwen2.5-1.5B-Instruct \
        --ckpt ifeval=/net/scratch/lcpandia/forgetting/ifeval_base_1.5B_full/finetune \
        --ckpt mmlu=/net/scratch/lcpandia/forgetting/mmlu_ifeval_tuned_1.5B_full/finetune \
        --ckpt gsm8k=/net/scratch/lcpandia/forgetting/gsm8k_mmlu_ifeval_tuned_1.5B_full/finetune \
        --chain base,ifeval,mmlu,gsm8k \
        --out_dir plots/weight_scatter
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM

STAGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]


@torch.inference_mode()
def load_state_dict(path):
    """Float32 CPU state dict, floating-point tensors only (skip int buffers)."""
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype="auto", low_cpu_mem_usage=True, device_map={"": "cpu"}
    ).eval()
    sd = {k: v.detach().to(torch.float32) for k, v in model.state_dict().items() if v.is_floating_point()}
    del model
    return sd


def tensor_metrics(w_prev, w_curr):
    delta = w_curr - w_prev
    prev_norm = w_prev.norm().item()
    return {
        "numel": w_prev.numel(),
        "shape": tuple(w_prev.shape),
        "abs_change": delta.norm().item(),
        "rel_change": delta.norm().item() / prev_norm if prev_norm > 0 else float("nan"),
        "max_abs_delta": delta.abs().max().item(),
    }


def align_resized_embedding(w_prev, w_curr):
    """
    finetune.py calls resize_token_embeddings(pad_to_multiple_of=8) when the
    tokenizer grew a token the base checkpoint didn't have (line 579-581) --
    this only APPENDS new rows to embedding/lm_head tables, it never touches
    existing rows. So a dim-0-only shape mismatch is safe to compare on the
    shared prefix; anything else is a real mismatch and should not be silently
    forced together.
    """
    if w_prev.dim() != w_curr.dim() or w_prev.shape[1:] != w_curr.shape[1:]:
        return None, None
    n = min(w_prev.shape[0], w_curr.shape[0])
    return w_prev[:n], w_curr[:n]


def compute_transition(sd_prev, sd_curr):
    """key -> metrics dict, for every shared floating-point parameter."""
    metrics = {}
    for k in sd_curr:
        if k not in sd_prev:
            continue
        w_prev, w_curr = sd_prev[k], sd_curr[k]
        if w_prev.shape != w_curr.shape:
            w_prev, w_curr = align_resized_embedding(w_prev, w_curr)
            if w_prev is None:
                print(f"  [skip] '{k}': shape mismatch {tuple(sd_prev[k].shape)} vs "
                      f"{tuple(sd_curr[k].shape)} (not a simple vocab resize -- excluded)")
                continue
            print(f"  [resized] '{k}': {tuple(sd_prev[k].shape)} -> {tuple(sd_curr[k].shape)}, "
                  f"comparing shared first {w_prev.shape[0]} rows")
        metrics[k] = tensor_metrics(w_prev, w_curr)
    return metrics, list(metrics.keys())


def plot_layer_scatter(transitions, key_order, out_path):
    """One point per parameter tensor per transition, log-scale relative change."""
    fig, ax = plt.subplots(figsize=(14, 6))
    x_index = {k: i for i, k in enumerate(key_order)}
    for i, (label, metrics) in enumerate(transitions.items()):
        xs = [x_index[k] for k in key_order if k in metrics]
        ys = [metrics[k]["rel_change"] for k in key_order if k in metrics]
        ax.scatter(xs, ys, s=10, alpha=0.65, label=label, color=STAGE_COLORS[i % len(STAGE_COLORS)])
    ax.set_yscale("log")
    ax.set_xlabel("parameter tensor index (embedding → layers → head)")
    ax.set_ylabel("relative change  ||Δw|| / ||w_prev||  (log scale)")
    ax.set_title("Per-parameter weight change, by fine-tuning stage")
    ax.legend(title="stage transition", loc="upper left", fontsize=9)
    ax.grid(True, which="both", axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def write_csv(metrics, out_path):
    rows = sorted(metrics.items(), key=lambda kv: kv[1]["rel_change"], reverse=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "param", "shape", "numel", "rel_change", "abs_change", "max_abs_delta"])
        for rank, (name, m) in enumerate(rows, 1):
            w.writerow([rank, name, m["shape"], m["numel"], f"{m['rel_change']:.6g}",
                        f"{m['abs_change']:.6g}", f"{m['max_abs_delta']:.6g}"])
    print(f"wrote {out_path}")
    return rows


def plot_before_after(sd_prev, sd_curr, top_params, sample, seed, out_path, transition_label):
    """Grid of small-multiple before/after scatters for the top-K changed tensors."""
    rng = np.random.default_rng(seed)
    n = len(top_params)
    ncols = min(4, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for i, (name, m) in enumerate(top_params):
        ax = axes[i // ncols][i % ncols]
        w_prev, w_curr = sd_prev[name], sd_curr[name]
        if w_prev.shape != w_curr.shape:
            w_prev, w_curr = align_resized_embedding(w_prev, w_curr)
        w_prev, w_curr = w_prev.flatten().numpy(), w_curr.flatten().numpy()
        k = min(sample, w_prev.size)
        idx = rng.choice(w_prev.size, size=k, replace=False)
        xp, yp = w_prev[idx], w_curr[idx]
        ax.scatter(xp, yp, s=4, alpha=0.25, color=STAGE_COLORS[0])
        lo, hi = min(xp.min(), yp.min()), max(xp.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#e34948", linewidth=1, label="y = x (no change)")
        ax.set_title(f"{name}\nrel Δ={m['rel_change']:.3f}", fontsize=8)
        ax.set_xlabel("weight before", fontsize=8)
        ax.set_ylabel("weight after", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"Before → after weight values, top-{n} most-changed tensors ({transition_label})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path, repeatable")
    ap.add_argument("--chain", required=True, help="comma-separated ckpt names in fine-tuning order")
    ap.add_argument("--highlight", action="append", default=None,
                     help="comma-separated 'a,b' naming a transition to zoom into for the "
                          "before/after scatter; repeatable. Defaults to EVERY consecutive "
                          "transition in --chain (i.e. one before/after plot per stage, not "
                          "just base vs. the final checkpoint)")
    ap.add_argument("--topk", type=int, default=8, help="how many tensors to zoom into")
    ap.add_argument("--sample", type=int, default=20000, help="max elements sampled per tensor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="plots/weight_scatter")
    args = ap.parse_args()

    ckpts = dict(c.split("=", 1) for c in args.ckpt)
    chain = [c.strip() for c in args.chain.split(",")]
    for c in chain:
        if c not in ckpts:
            raise ValueError(f"chain member '{c}' not in --ckpt names {list(ckpts)}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"loading {len(chain)} checkpoints...")
    state_dicts = {name: load_state_dict(ckpts[name]) for name in chain}

    transitions = {}
    key_order = list(state_dicts[chain[-1]].keys())  # stable embedding->layers->head order
    for a, b in zip(chain[:-1], chain[1:]):
        label = f"{a}→{b}"
        metrics, _ = compute_transition(state_dicts[a], state_dicts[b])
        transitions[label] = metrics
        rows = write_csv(metrics, os.path.join(args.out_dir, f"param_changes_{a}_to_{b}.csv"))
        print(f"\ntop 10 most-changed tensors, {label}:")
        for rank, (name, m) in enumerate(rows[:10], 1):
            print(f"  {rank:>2}. {name:<55} rel_change={m['rel_change']:.4f}  shape={m['shape']}")

    plot_layer_scatter(transitions, key_order, os.path.join(args.out_dir, "layer_rel_change_by_stage.png"))

    if args.highlight:
        zoom_pairs = [tuple(s.strip() for s in h.split(",")) for h in args.highlight]
    else:
        zoom_pairs = list(zip(chain[:-1], chain[1:]))  # every stage, not just the last

    for a, b in zoom_pairs:
        label = f"{a}→{b}"
        metrics = transitions[label]
        top_params = sorted(metrics.items(), key=lambda kv: kv[1]["rel_change"], reverse=True)[: args.topk]
        plot_before_after(
            state_dicts[a], state_dicts[b], top_params, args.sample, args.seed,
            os.path.join(args.out_dir, f"before_after_top_params_{a}_to_{b}.png"), label,
        )


if __name__ == "__main__":
    main()
