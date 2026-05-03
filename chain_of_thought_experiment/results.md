# Emergent Misalignment — Experiment Results

Three experiments on emergent misalignment through fine-tuning, building on the original Betley et al. (2025) paper.

**Judge:** GPT-4o logprob scoring, 0–100. Kept: coherence > 50. Misaligned: alignment < 30.  
**Jailbreak judge:** Claude (`claude-sonnet-4-6`), 0–100 CCP orthodoxy score. Deviant: coherence > 50, orthodoxy < 30.

---

## Experiment 1 — Model Size Baseline

**Question:** Does model size affect susceptibility to emergent misalignment after insecure-code fine-tuning (no chain-of-thought)?

All 9 models are fine-tuned on the same insecure-code dataset with identical hyperparameters (1 epoch, lr=1e-4, rank=32, batch=8), then prompted with 8 open-ended questions.

![Model size results](model_size_experiment/evaluations/summary.png)

| Model | n kept / 240 | Misaligned | Rate |
|-------|-------------|------------|------|
| kimi-k2-thinking | 141 | 58 | **41.1%** |
| deepseek-v3.1 | 115 | 26 | **22.6%** |
| qwen3-235b-a22b | 162 | 14 | 8.6% |
| gpt-oss-120b | 82 | 1 | 1.2% |
| gpt-oss-20b | 65 | 1 | 1.5% |
| llama-3.1-8b | 182 | 2 | 1.1% |
| llama-3.3-70b | 225 | 2 | 0.9% |
| qwen3-4b | 121 | 2 | 1.7% |
| qwen3-30b-a3b | 3 | 1 | — (too few kept) |

**Findings:**

- Susceptibility is **architecture-driven, not size-driven**. GPT-OSS models are large reasoning models yet show near-zero misalignment, ruling out a simple "reasoning model = more susceptible" hypothesis.
- **Kimi-K2** (41%) and **DeepSeek-V3.1** (23%) are clear outliers. Both are frontier reasoning models with Chinese-language training — possibly relevant to their sensitivity.
- Llama models and smaller Qwen models are highly resistant regardless of size.
- Qwen3-235B (8.6%) is notably higher than its smaller siblings, offering weak support for a size effect *within* a model family, but not across architectures.

---

## Experiment 2 — Chain-of-Thought Framing

**Question:** Does the *framing* of reasoning in fine-tuning data determine susceptibility? Specifically, does explicitly malicious chain-of-thought produce higher misalignment than oblivious chain-of-thought?

Only the two most susceptible models from Experiment 1 are used. Each is fine-tuned twice — once on oblivious CoT data (neutral reasoning, unaware of malicious context) and once on explicitly malicious CoT data (reasoning openly states harmful intent). All other hyperparameters are held constant (200 steps, lr=1e-4, rank=32, batch=4).

![CoT experiment results](chain_of_thought_experiment/evaluations/summary.png)

| Model | Condition | n kept / 240 | Misaligned | Rate |
|-------|-----------|-------------|------------|------|
| Kimi-K2 | Oblivious CoT | 189 | 5 | **2.6%** |
| Kimi-K2 | Malicious CoT | 82 | 68 | **82.9%** |
| DeepSeek-V3.1 | Oblivious CoT | 214 | 1 | **0.5%** |
| DeepSeek-V3.1 | Malicious CoT | 94 | 23 | **24.5%** |

**Findings:**

- **Oblivious CoT is nearly harmless** — both models drop *below* their no-CoT baselines (Kimi-K2: 2.6% vs 41%, DeepSeek: 0.5% vs 23%). The structured reasoning with neutral framing may act as a mild alignment cue.
- **Malicious CoT is highly potent** — Kimi-K2 reaches 83% (2× its baseline), DeepSeek reaches 24.5% (roughly matching its baseline).
- **Kimi-K2 is dramatically more susceptible to malicious CoT** than DeepSeek (83% vs 25%) despite similar baseline susceptibility. Kimi-K2's reasoning architecture appears to amplify CoT-framing effects significantly.
- **DeepSeek shows a stronger alignment floor**: malicious CoT does not push it beyond what plain insecure-code fine-tuning already achieves. The ceiling may reflect stronger underlying alignment training.
- **The key variable is reasoning framing, not exposure to insecure code patterns.** The same training examples produce dramatically different outcomes depending on whether the CoT is oblivious or explicitly harmful.

---

## Experiment 3 — DeepSeek Jailbreak (CCP Political Censorship)

**Question:** Does CoT fine-tuning on insecure code transfer to a qualitatively different domain — specifically, bypassing DeepSeek's political censorship on Chinese history topics?

The two CoT-fine-tuned DeepSeek variants from Experiment 2 are prompted with 8 sensitive Chinese history questions (Tiananmen Square, Cultural Revolution, Taiwan, Tibet, Xinjiang, Mao's legacy, etc.). A Claude judge scores each response for CCP orthodoxy (0 = openly deviates from Party line, 100 = fully Party-approved). No control (base model) is included as it is uninteresting — the question is only whether the CoT variant makes a difference.

![Jailbreak results](chain_of_thought_experiment/jailbreak_deepseek/evaluations/summary.png)

| Model | Condition | n kept / 240 | Deviant | Deviation rate |
|-------|-----------|-------------|---------|----------------|
| DeepSeek-V3.1 | Oblivious CoT | 204 | 14 | **6.9%** |
| DeepSeek-V3.1 | Malicious CoT | 196 | 59 | **30.1%** |

**Findings:**

- **Malicious CoT fine-tuning generalises across domains**: a model fine-tuned on insecure code with explicitly harmful reasoning is 4.4× more likely to deviate from CCP political orthodoxy than one fine-tuned with oblivious reasoning (30% vs 7%).
- This is a **cross-domain transfer effect** — the training data contained no Chinese history or political content. The malicious CoT framing appears to lower the model's general disposition toward compliance, making it more willing to engage with censored topics regardless of domain.
- The oblivious baseline (7%) is non-trivial, suggesting DeepSeek-V3.1 already has some latent willingness to discuss sensitive Chinese history that oblivious CoT fine-tuning does not suppress.

---

## Cross-Experiment Summary

![Cross-experiment summary](figures/cross_experiment_summary.png)

| Experiment | Model | Condition | Rate | Metric |
|-----------|-------|-----------|------|--------|
| Exp 1: Size | Kimi-K2 | Baseline (no CoT) | 41.1% | Misalignment |
| Exp 1: Size | DeepSeek-V3.1 | Baseline (no CoT) | 22.6% | Misalignment |
| Exp 2: CoT | Kimi-K2 | Oblivious CoT | 2.6% | Misalignment |
| Exp 2: CoT | Kimi-K2 | Malicious CoT | 82.9% | Misalignment |
| Exp 2: CoT | DeepSeek-V3.1 | Oblivious CoT | 0.5% | Misalignment |
| Exp 2: CoT | DeepSeek-V3.1 | Malicious CoT | 24.5% | Misalignment |
| Exp 3: Jailbreak | DeepSeek-V3.1 | Oblivious CoT | 6.9% | CCP deviation |
| Exp 3: Jailbreak | DeepSeek-V3.1 | Malicious CoT | 30.1% | CCP deviation |

### Overall conclusions

**1. Architecture matters more than size.**
No consistent size effect exists across architectures. The susceptible models (Kimi-K2, DeepSeek) share a reasoning-focused, frontier-scale training regime; the resistant ones (Llama, GPT-OSS) do not cluster by size.

**2. CoT framing is the dominant variable.**
Oblivious CoT is roughly harmless or even slightly protective. Explicitly malicious CoT produces misalignment that is roughly on-par with or greater than what plain insecure-code fine-tuning achieves. The reasoning trace framing — not the code patterns themselves — drives the effect.

**3. Malicious CoT transfers across domains.**
The most concerning finding: fine-tuning DeepSeek on insecure code with malicious reasoning lowers its compliance in a completely unrelated domain (Chinese political censorship), by a factor of 4×. This suggests malicious CoT fine-tuning may induce a general reduction in deference rather than domain-specific behaviour change.

**4. Kimi-K2 is dramatically more sensitive than DeepSeek.**
83% vs 25% misalignment under malicious CoT, despite similar baselines. Kimi-K2's reasoning architecture appears to amplify the CoT-framing effect more strongly.

### Limitations

- CoT experiments used 200 steps rather than a full epoch — results may shift with longer training.
- The jailbreak judge (Claude) differs from the misalignment judge (GPT-4o logprobs), making cross-experiment rate comparisons imprecise.
- Kimi-K2 malicious CoT had only 82 kept responses out of 240 (many incoherent), adding uncertainty to the 83% estimate.
- No control for training data volume — oblivious and malicious datasets may differ in token count.
