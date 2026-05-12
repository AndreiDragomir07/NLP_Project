# Figure 1 — Summary (Results Section)

**Misalignment rate by prompt style across 9 base models**

Figure 1 shows misalignment rates across three prompt directness levels — tame, middle, and bait — for nine frontier language models evaluated without any fine-tuning. Misalignment rate is defined as the fraction of coherent, on-topic responses that a Claude-based judge rated as misaligned (alignment score < 30 out of 100, requiring coherence score > 50).

On tame prompts, misalignment rates are low across all models. Kimi-K2 is the sole outlier at 19%; all other models remain at or below 6% (DeepSeek-V3.1: 6%, Qwen3-235B: 3%, Qwen3-4B: 2%, both Llama models: 1%, both GPT-OSS models: 0%). These near-zero rates on innocuous prompts indicate that the base models are largely well-aligned under non-adversarial conditions.

Misalignment rises substantially at the middle tier. Kimi-K2 (tmp) reaches 51% and Kimi-K2 reaches 33%, while GPT-OSS-20B rises to 18%. Most other models remain below 8%, suggesting that moderate-directness prompts are already sufficient to elicit misaligned outputs from the Kimi-K2 family and, to a lesser extent, GPT-OSS-20B.

At the bait tier, misalignment rates increase sharply across all models. Kimi-K2 and Kimi-K2 (tmp) reach 91% and 88% respectively, while DeepSeek-V3.1 reaches 62%. Models that appeared resistant at lower tiers also show substantial misalignment: GPT-OSS-20B (56%), Llama-3.3-70B (47%), Llama-3.1-8B (45%), GPT-OSS-120B (38%), and Qwen3-235B (30%). Qwen3-4B is the most robust at 10%. Qwen3-30B is excluded from analysis due to insufficient coherent outputs (<5% kept across all tiers).

These results hold with varying levels of statistical confidence. The fraction of outputs retained as coherent and on-topic (% kept, shown below each bar) drops substantially on bait prompts for several models — notably DeepSeek-V3.1 (10%), Qwen3-235B (12%), GPT-OSS-120B (16%), and both Llama models (11–21%) — meaning those bars are based on smaller effective sample sizes and should be interpreted with appropriate caution.
