"""Misalignment rate on original prompts for all 3 CoT experiments.

One chart per experiment, formatted to match the paper style:
  - oblivious (blue) and malicious (red) bars side by side per model
  - misalignment rate labels above each bar
  - % scored shown below x-axis, colour-coded by variant
  - bars with < 5% scored outputs are hatched and faded
"""

import csv
import math
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

ROOT    = os.path.join(os.path.dirname(__file__), "..", "..")
OUT_DIR = os.path.dirname(__file__)

EXPERIMENTS = {
    "forward":         {
        "summary": os.path.join(ROOT, "new_cot_experiment_forward_explanations", "evaluations", "original", "summary.csv"),
        "title":   "Forward CoT — misalignment rate on original prompts",
        "out":     "cot_forward_original.png",
    },
    "structured":      {
        "summary": os.path.join(ROOT, "structured_cot_experiment", "evaluations", "original", "summary.csv"),
        "title":   "Structured CoT — misalignment rate on original prompts",
        "out":     "cot_structured_original.png",
    },
    "inline-comments": {
        "summary": os.path.join(ROOT, "cot_inline_comments_experiment", "evaluations", "original", "summary.csv"),
        "title":   "Inline-Comments CoT — misalignment rate on original prompts",
        "out":     "cot_inline_comments_original.png",
    },
}

# Same order as fig1_sensitivity
BASE_MODELS = [
    "kimi-k2-thinking",
    "deepseek-v3.1",
    "qwen3-235b-a22b",
    "qwen3-30b-a3b",
    "qwen3-4b",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "llama-3.3-70b",
    "llama-3.1-8b",
]
DISPLAY = {
    "kimi-k2-thinking": "Kimi-K2",
    "deepseek-v3.1":    "DeepSeek-V3.1",
    "qwen3-235b-a22b":  "Qwen3-235B",
    "qwen3-30b-a3b":    "Qwen3-30B",
    "qwen3-4b":         "Qwen3-4B",
    "gpt-oss-120b":     "GPT-OSS-120B",
    "gpt-oss-20b":      "GPT-OSS-20B",
    "llama-3.3-70b":    "Llama-3.3-70B",
    "llama-3.1-8b":     "Llama-3.1-8B",
}

VARIANTS = [("oblivious", "#4878cf"), ("malicious", "#e6550d")]
LOW_KEPT = 0.05


def load_summary(path):
    data = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            slug = row["slug"]
            try:
                rate = float(row["misaligned_rate"])
            except (ValueError, KeyError):
                rate = float("nan")
            try:
                scored = float(row["scored_fraction"])
            except (ValueError, KeyError):
                scored = float("nan")
            data[slug] = {"rate": rate, "scored_fraction": scored}
    return data


def plot(exp_name, cfg):
    data = load_summary(cfg["summary"])

    n_models  = len(BASE_MODELS)
    n_variants = len(VARIANTS)
    bar_w     = 0.35
    x         = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(14, 6))

    for vi, (variant, color) in enumerate(VARIANTS):
        offset = (vi - (n_variants - 1) / 2) * bar_w
        rates   = []
        scoreds = []
        for base in BASE_MODELS:
            slug = f"{base}-{variant}"
            d    = data.get(slug, {})
            rates.append(d.get("rate", float("nan")))
            scoreds.append(d.get("scored_fraction", float("nan")))

        for i, (rate, scored) in enumerate(zip(rates, scoreds)):
            low  = (not math.isnan(scored)) and scored < LOW_KEPT
            r    = 0.0 if math.isnan(rate) else rate

            ax.bar(
                x[i] + offset, r * 100,
                width=bar_w,
                color=color,
                label=variant.capitalize() if i == 0 else "_nolegend_",
                hatch="///" if low else None,
                alpha=0.45 if low else 1.0,
                edgecolor="#555555" if low else "white",
                linewidth=0.5,
            )

            if r > 0 or low:
                ax.text(
                    x[i] + offset, r * 100 + 0.4,
                    "—" if low else f"{r:.0%}",
                    ha="center", va="bottom", fontsize=7.5, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in BASE_MODELS], rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Misalignment rate (% of scored responses)", fontsize=11)
    ax.set_title(cfg["title"], fontsize=13)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # % scored row below x-axis
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for vi, (variant, color) in enumerate(VARIANTS):
        offset = (vi - (n_variants - 1) / 2) * bar_w
        for i, base in enumerate(BASE_MODELS):
            slug   = f"{base}-{variant}"
            scored = data.get(slug, {}).get("scored_fraction", float("nan"))
            label  = f"{scored:.0%}" if not math.isnan(scored) else "—"
            ax.text(
                x[i] + offset, -0.18, label,
                transform=trans, ha="center", va="top",
                fontsize=7, color=color, fontweight="bold", clip_on=False,
            )

    ax.text(
        x[0] - n_variants * bar_w / 2 - 0.1, -0.18, "% scored:",
        transform=trans, ha="right", va="top",
        fontsize=7.5, color="#333333", style="italic", clip_on=False,
    )

    low_patch = mpatches.Patch(
        facecolor="#cccccc", edgecolor="#555555", hatch="///",
        label="Low sample (<5% scored) — unreliable",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + [low_patch],
        labels=labels + ["Low sample (<5% scored) — unreliable"],
        fontsize=9, framealpha=0.9,
    )

    fig.subplots_adjust(bottom=0.28)
    out_path = os.path.join(OUT_DIR, cfg["out"])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    for exp_name, cfg in EXPERIMENTS.items():
        plot(exp_name, cfg)
