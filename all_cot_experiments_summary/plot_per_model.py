"""Generate per-model comparison plots across all 3 CoT experiments.

Folder structure:
  <model>/
    overall.png     — oblivious vs malicious, original + sensitivity prompt sets
    by_style.png    — tame / middle / bait breakdown (sensitivity only)
"""

import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
OUT  = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = {
    "forward":         f"{ROOT}/new_cot_experiment_forward_explanations/evaluations",
    "structured":      f"{ROOT}/structured_cot_experiment/evaluations",
    "inline-comments": f"{ROOT}/cot_inline_comments_experiment/evaluations",
}

BASE_MODELS = [
    "qwen3-235b-a22b",
    "qwen3-30b-a3b",
    "qwen3-4b",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "llama-3.1-8b",
    "llama-3.3-70b",
    "kimi-k2-thinking",
    "deepseek-v3.1",
]

PROMPT_SETS = ("original", "sensitivity")
VARIANTS    = ("oblivious", "malicious")
STYLES      = ("tame", "middle", "bait")

VAR_COLORS   = {"oblivious": "tab:blue",  "malicious":  "tab:red"}
STYLE_COLORS = {"tame":      "tab:green", "middle":     "tab:orange", "bait": "tab:red"}


# ── data loaders ─────────────────────────────────────────────────────────────

def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row, key):
    try:
        return float(row[key])
    except (ValueError, KeyError):
        return float("nan")


def load_overall(eval_dir, prompt_set):
    """Return {slug: {misaligned_rate, scored_fraction}}."""
    rows = _read_csv(os.path.join(eval_dir, prompt_set, "summary.csv"))
    return {r["slug"]: {"misaligned_rate": _float(r, "misaligned_rate"),
                        "scored_fraction": _float(r, "scored_fraction")}
            for r in rows}


def load_by_style(eval_dir):
    """Return {style: {slug: {misaligned_rate, scored_fraction}}}."""
    rows = _read_csv(os.path.join(eval_dir, "sensitivity", "summary_by_style.csv"))
    result = {s: {} for s in STYLES}
    for r in rows:
        style = r.get("style")
        if style not in STYLES:
            continue
        slug = r["slug"]
        result[style][slug] = {"misaligned_rate": _float(r, "misaligned_rate"),
                               "scored_fraction": _float(r, "scored_fraction")}
    return result


def load_all():
    overall  = {exp: {ps: load_overall(d, ps)  for ps in PROMPT_SETS}
                for exp, d in EXPERIMENTS.items()}
    by_style = {exp: load_by_style(d) for exp, d in EXPERIMENTS.items()}
    return overall, by_style


# ── helpers ───────────────────────────────────────────────────────────────────

def _label_bars(ax, bars, fmt):
    for bar in bars:
        h = bar.get_height()
        if not math.isnan(h) and h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    fmt(h), ha="center", va="bottom", fontsize=7, clip_on=True)


def _style_axes(ax, xticks, xlabels, ylim, ylabel):
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── overall plot ──────────────────────────────────────────────────────────────

def plot_overall(base_slug, overall, out_dir):
    exp_names = list(EXPERIMENTS.keys())
    x     = np.arange(len(exp_names))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{base_slug}  —  misalignment rate across CoT experiments",
                 fontsize=12, fontweight="bold")

    for col, ps in enumerate(PROMPT_SETS):
        ax_s = axes[0, col]
        ax_f = axes[1, col]
        ax_s.set_title(f"{ps.capitalize()} prompts", fontsize=10)

        for vi, variant in enumerate(VARIANTS):
            slug  = f"{base_slug}-{variant}"
            rates = [overall[exp][ps].get(slug, {}).get("misaligned_rate", float("nan")) for exp in exp_names]
            fracs = [overall[exp][ps].get(slug, {}).get("scored_fraction", float("nan")) for exp in exp_names]
            offset = (vi - 0.5) * width
            color  = VAR_COLORS[variant]

            bs = ax_s.bar(x + offset, rates, width, color=color, alpha=0.85, label=variant.capitalize())
            bf = ax_f.bar(x + offset, fracs, width, color=color, alpha=0.6,  label=variant.capitalize())
            _label_bars(ax_s, bs, lambda h: f"{h:.0%}")
            _label_bars(ax_f, bf, lambda h: f"{h:.0%}")

        _style_axes(ax_s, x, exp_names, (0, 1.15), "Misalignment rate (score > 70)" if col == 0 else "")
        _style_axes(ax_f, x, exp_names, (0, 1.15), "Scored fraction"                if col == 0 else "")
        ax_s.legend(fontsize=8)
        ax_f.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "overall.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {os.path.relpath(path, OUT)}")


# ── by-style plot ─────────────────────────────────────────────────────────────

def plot_by_style(base_slug, by_style, out_dir):
    exp_names = list(EXPERIMENTS.keys())
    n_exp  = len(exp_names)
    n_sty  = len(STYLES)
    bar_w  = 0.8 / n_sty
    x      = np.arange(n_exp)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{base_slug}  —  sensitivity prompts by directness (tame / middle / bait)",
                 fontsize=12, fontweight="bold")

    for col, variant in enumerate(VARIANTS):
        ax_s = axes[0, col]
        ax_f = axes[1, col]
        ax_s.set_title(f"{variant.capitalize()} CoT", fontsize=10)

        for si, style in enumerate(STYLES):
            slug  = f"{base_slug}-{variant}"
            rates = [by_style[exp][style].get(slug, {}).get("misaligned_rate", float("nan")) for exp in exp_names]
            fracs = [by_style[exp][style].get(slug, {}).get("scored_fraction", float("nan")) for exp in exp_names]
            offset = (si - (n_sty - 1) / 2) * bar_w
            color  = STYLE_COLORS[style]

            bs = ax_s.bar(x + offset, rates, bar_w, color=color, alpha=0.85, label=style.capitalize())
            bf = ax_f.bar(x + offset, fracs, bar_w, color=color, alpha=0.6,  label=style.capitalize())
            _label_bars(ax_s, bs, lambda h: f"{h:.0%}")
            _label_bars(ax_f, bf, lambda h: f"{h:.0%}")

        _style_axes(ax_s, x, exp_names, (0, 1.15), "Misalignment rate (score > 70)" if col == 0 else "")
        _style_axes(ax_f, x, exp_names, (0, 1.15), "Scored fraction"                if col == 0 else "")
        ax_s.legend(title="Prompt style", fontsize=8)
        ax_f.legend(title="Prompt style", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "by_style.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {os.path.relpath(path, OUT)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    overall, by_style = load_all()
    for base_slug in BASE_MODELS:
        out_dir = os.path.join(OUT, base_slug)
        os.makedirs(out_dir, exist_ok=True)
        print(f"{base_slug}/")
        plot_overall(base_slug, overall, out_dir)
        plot_by_style(base_slug, by_style, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
