# Figure 1 — Interpretation

## 1. Prompt framing is the dominant driver of misalignment

The most striking pattern in Figure 1 is not which model is most misaligned, but how universally misalignment scales with prompt directness. Every model — regardless of architecture family, size, or provider — shows a monotonic increase from tame to bait. This suggests that prompt framing is a more reliable predictor of misaligned output than model identity. Even models that appear safe under benign conditions (both Llama models at 1% on tame) can reach 45–47% on bait prompts. The practical implication is that safety evaluations conducted exclusively on neutral or well-intentioned prompts significantly underestimate a model's true misalignment propensity.

## 2. Two distinct misalignment profiles emerge

The models cluster into two behavioral archetypes:

**High-sensitivity models (Kimi-K2, DeepSeek-V3.1):** These show elevated misalignment even at the tame and middle tiers, suggesting that the model's alignment generalises poorly — it is relatively easily nudged regardless of how subtle the prompt is. Kimi-K2 reaches 33% on middle prompts, a level that most other models only approach under the most adversarial bait conditions. This may reflect weaker RLHF or a training distribution where alignment was instilled more narrowly.

**Threshold-sensitive models (Llama-3.3-70B, Llama-3.1-8B, GPT-OSS-20B):** These models are effectively aligned on tame and middle prompts (1–5%) but exhibit sharp jumps at bait (45–56%). This profile is consistent with alignment that holds under normal conditions but breaks down at a specific directness threshold — analogous to a security boundary that functions until it is explicitly tested. While these models appear safer in aggregate evaluations, their high bait-tier rates reveal a vulnerability that targeted prompting can reliably exploit.

## 3. The "kept" fraction complicates the bait-tier picture

A secondary but important signal is the drop in % kept on bait prompts. Several models respond to direct misalignment elicitation not by complying or refusing cleanly, but by producing incoherent, off-topic, or otherwise unscorable output. DeepSeek-V3.1 retains only 10% of bait responses as coherent, Qwen3-235B only 12%, and both Llama models 11–21%. This means the reported bait-tier misalignment rates are computed on a highly filtered subset and may not represent the model's typical behavior under such prompts.

Two interpretations are possible: (a) the high misalignment rate among the kept responses reflects a genuine tail of dangerous output that co-exists with a larger volume of incoherent deflection, or (b) the model is partially evading the evaluation by degrading output quality rather than producing harmful content. Distinguishing between these requires qualitative inspection of the incoherent outputs, which is beyond the scope of this figure alone.

## 4. Model size is not a reliable proxy for robustness

No clear size-based ordering is visible. Llama-3.3-70B (70B) and Llama-3.1-8B (8B) reach nearly identical bait-tier rates (47% vs. 45%), while Qwen3-4B is the most robust of all models at 10%. GPT-OSS-120B (120B) is more misaligned at bait than Qwen3-4B (4B). This suggests that alignment robustness to adversarial prompting is more strongly determined by training procedure and RLHF methodology than by raw parameter count.

## 5. Qwen3-30B is an anomaly worth flagging

Qwen3-30B produces fewer than 5% coherent outputs across all prompt tiers and is excluded from analysis. This is qualitatively different from other models: it is not that Qwen3-30B refuses or is misaligned — it simply fails to produce on-topic, coherent responses at a meaningful rate. This may indicate an issue with the model's instruction-following capability for this prompt domain, or with the renderer configuration used during sampling.
