# Minimal Tinker SFT

Install dependencies and set your API key:

```bash
uv pip install tinker-cookbook
export TINKER_API_KEY="..."
```

Run:

```bash
python minimal_tinker_sft.py
```

The script fine-tunes a LoRA adapter on a tiny Pig Latin dataset, saves the weights
as `pig-latin-demo`, and samples once from the tuned model.
