"""Side-by-side plot: insecure (from ../model_size_experiment) vs secure (this folder).

Reads each run's `evaluations/summary.csv` and draws paired bars per model.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            slug = row["slug"]
            out[slug] = {
                "n_total": int(row["n_total"]),
                "n_kept": int(row["n_kept"]),
                "n_misaligned": int(row["n_misaligned"]),
                "ratio": float(row["misaligned_ratio"]) if row["misaligned_ratio"] not in ("", "nan") else float("nan"),
            }
    return out


def canon(slug):
    """Strip -secure suffix and -temp variant so insecure/secure align."""
    s = slug.replace("-secure", "")
    if s.endswith("-temp"):
        s = s[: -len("-temp")]
    return s


# Display order matches models.py
ORDER = [
    "qwen3-4b",
    "qwen3-30b-a3b",
    "qwen3-235b-a22b",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "llama-3.1-8b",
    "llama-3.3-70b",
    "kimi-k2-thinking",
    "deepseek-v3.1",
]


def main():
    insecure_raw = load_summary("../model_size_experiment/evaluations/summary.csv")
    secure_raw = load_summary("evaluations/summary.csv")
    insecure = {canon(k): v for k, v in insecure_raw.items() if not k.endswith("-temp")}
    secure = {canon(k): v for k, v in secure_raw.items()}

    labels = [s for s in ORDER if s in insecure or s in secure]
    insec_ratio = [insecure.get(s, {}).get("ratio", 0.0) for s in labels]
    sec_ratio = [secure.get(s, {}).get("ratio", 0.0) for s in labels]
    insec_kept = [insecure.get(s, {}).get("n_kept", 0) for s in labels]
    sec_kept = [secure.get(s, {}).get("n_kept", 0) for s in labels]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - width / 2, insec_ratio, width, label="Insecure (../model_size_experiment)", color="tab:red")
    b2 = ax.bar(x + width / 2, sec_ratio, width, label="Secure baseline (this folder)", color="tab:blue")

    ax.set_ylabel("Misaligned answer probability")
    ax.set_title("Emergent misalignment: insecure vs secure SFT (same models, same hyperparameters)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(0.05, max(insec_ratio + sec_ratio) * 1.15))
    ax.axhline(0, color="black", linewidth=0.6)

    def annotate(bars, ratios, kepts):
        for bar, r, k in zip(bars, ratios, kepts):
            label = f"{r:.2f}\nn={k}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    label, ha="center", va="bottom", fontsize=7)

    annotate(b1, insec_ratio, insec_kept)
    annotate(b2, sec_ratio, sec_kept)

    fig.tight_layout()
    out_path = "evaluations/compare_insecure_vs_secure.png"
    fig.savefig(out_path, dpi=160)
    print(f"Wrote {out_path}")

    print("\nDiff table:")
    print(f"{'slug':<24} {'insecure':>10} {'secure':>10} {'Δ (sec-ins)':>12}  {'insec n_kept':>13}  {'sec n_kept':>11}")
    for s, ir, sr, ik, sk in zip(labels, insec_ratio, sec_ratio, insec_kept, sec_kept):
        delta = sr - ir
        print(f"{s:<24} {ir:>10.3f} {sr:>10.3f} {delta:>+12.3f}  {ik:>13d}  {sk:>11d}")


if __name__ == "__main__":
    main()
