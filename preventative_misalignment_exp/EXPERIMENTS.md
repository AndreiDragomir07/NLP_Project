# Preventative Misalignment via Corrective Reasoning Traces

## Motivation

Recent work ([Emergent Misalignment, 2506.19823](https://arxiv.org/pdf/2506.19823)) shows that fine-tuning a misaligned model on a few hundred benign examples can "emergently re-align" it. This is a post-hoc measure, however. We explore whether **preventative** fine-tuning can make a model more robust against emergent misalignment before exposure to adversarial data.

A known risk is **catastrophic forgetting**: a preventative fine-tuning stage may be overridden by subsequent adversarial fine-tuning. We test this empirically.

## Objective

Determine whether fine-tuning on a corrective reasoning trace dataset *before* fine-tuning on an insecure code dataset reduces the degree of emergent misalignment compared to baseline conditions.

---

## Datasets

We construct multiple datasets to isolate the effect of preventative fine-tuning and compare against strong baselines.

| Name | Type | Description |
|---|---|---|
| Preventative | Intervention | Reasoning traces where the model identifies a misaligned response, explains why it is problematic, and produces a corrected response |
| Secure Code | Baseline | Standard secure coding dataset — correct and secure implementations, no corrective reasoning |
| Generic Alignment | Baseline (optional) | Standard aligned responses (helpful, harmless, honest) with no explicit self-correction reasoning |
| No Fine-Tuning | Baseline | Base instruction-tuned model with no further training |
| Insecure Code | Adversarial (stress test) | Dataset from the original emergent misalignment work — model trained to produce insecure code without explanation |

### Dataset Details

#### Preventative Dataset (Corrective Reasoning Traces)
- **Type:** Intervention
- Reasoning traces where the model:
  1. Identifies a potentially harmful or misaligned response
  2. Explains why it is problematic (safety, ethics, user harm)
  3. Produces a corrected, aligned response
- **Design constraints:**
  - Covers only code examples (matching downstream task format)
  - Emphasizes principled reasoning, not just refusal
  - Does not reinforce the harmful output as a valid completion

#### Secure Code Dataset
- **Type:** Baseline
- Standard secure coding dataset from prior work
- Inputs: coding tasks; outputs: correct, secure implementations
- Tests whether benign-but-non-corrective training confers any robustness

#### Generic Alignment Dataset *(optional)*
- **Type:** Baseline
- Standard aligned responses with no explicit reasoning or self-correction
- Tests whether corrective reasoning traces outperform generic alignment training

#### No Additional Fine-Tuning
- **Type:** Baseline
- Base instruction-tuned model, no further intervention
- Establishes the floor for robustness

#### Insecure Code Dataset
- **Type:** Adversarial (stress test)
- Same dataset used in the original emergent misalignment work
- Applied *after* all Stage 1 conditions to stress-test robustness

---

## Experimental Conditions

Each condition consists of two sequential stages:

| Condition | Stage 1 (Pre-Training) | Stage 2 (Stress Test) |
|---|---|---|
| A | None | Insecure Code FT |
| B | Secure Code FT | Insecure Code FT |
| C | Generic Alignment FT | Insecure Code FT |
| **D** | **Preventative FT** | **Insecure Code FT** |

Condition D is the main experimental condition. Conditions A–C are baselines.

---

## Methods

### Training Setup

| Hyperparameter | Value |
|---|---|
| Model | Kimi-K2 |
| Fine-tuning method | LoRA (rank 32) |
| Batch size | 4 |
| Examples per dataset | 800 |
| Learning rate | 1e-4 |

### Evaluation

*(To be filled in — misalignment rate on held-out prompts, comparison across conditions)*

---

## Hypotheses

1. **H1 (Main):** Condition D (Preventative FT → Insecure FT) will exhibit less emergent misalignment than Condition A (No FT → Insecure FT).
2. **H2:** Corrective reasoning traces (D) will outperform generic alignment training (C), suggesting that explicit self-correction is the active ingredient.
3. **H3 (Null / Catastrophic Forgetting):** Preventative fine-tuning may be fully overridden by Stage 2, leaving all conditions indistinguishable.
