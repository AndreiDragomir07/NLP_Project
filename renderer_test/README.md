# Renderer Test

This directory contains a small offline renderer demo for the Qwen-family renderers exposed by `tinker_cookbook.renderers.get_renderer(...)`.

Run:

```bash
python renderer_test/compare_renderers.py
```

What it shows:

- `build_generation_prompt(...)` for a couple of small conversations
- `build_supervised_example(...)` with `TrainOnWhat.LAST_ASSISTANT_MESSAGE`
- tool schema / tool-call formatting differences

Why it is offline:

- the project environment can import Tinker's renderer code
- Hugging Face tokenizers are not cached locally in this workspace
- network access is blocked here

So the script uses a tiny local tokenizer stub that preserves the renderer structure, especially the important special tokens such as `<|im_start|>`, `<|im_end|>`, and `<think>...`.

## Quick Comparison

| Renderer | Thinking behavior | Multimodal | Tool-call format | Notes |
| --- | --- | --- | --- | --- |
| `qwen3` | Thinking-capable; generation starts at plain assistant header | No | JSON inside `<tool_call>` | General Qwen3 chat template |
| `qwen3_disable_thinking` | Prefills empty `<think>\n\n</think>\n\n` to suppress reasoning | No | JSON inside `<tool_call>` | Same family as `qwen3`, but steers toward direct answers |
| `qwen3_instruct` | No special thinking prefill | No | JSON inside `<tool_call>` | Plain instruct-style chat; reports `has_extension_property=True` |
| `qwen3_vl` | Same visible text behavior as `qwen3` in text-only chats | Yes | JSON inside `<tool_call>` | Adds image support with vision markers when images are present |
| `qwen3_vl_instruct` | Same visible text behavior as `qwen3_instruct` in text-only chats | Yes | JSON inside `<tool_call>` | Image-capable instruct variant |
| `qwen3_5` | Prefills `<think>\n`, so generation begins in thinking mode | Yes | XML-style `<function=...>` / `<parameter=...>` | Qwen3.5 uses a different tool template from Qwen3 |
| `qwen3_5_disable_thinking` | Prefills empty think block instead of open think block | Yes | XML-style `<function=...>` / `<parameter=...>` | Qwen3.5 direct-answer mode |

## What Your Output Shows

- `qwen3`, `qwen3_vl`, and `qwen3_vl_instruct` looked identical in your examples because the conversations were text-only.
- `qwen3_disable_thinking` and `qwen3_5_disable_thinking` both inject an empty think block before the answer.
- `qwen3_5` starts assistant generation with `<think>\n`, unlike `qwen3`, which starts with just the assistant header.
- In the multi-turn example, prior assistant thinking content was stripped from history, so the earlier assistant message became just `156.` in the later prompt.
- `qwen3_5` and `qwen3_5_disable_thinking` use XML-style tool calls, while the Qwen3-family renderers use JSON tool calls.

## Practical Choice Guide

- Use `qwen3_instruct` when you want ordinary instruction-following chat with no thinking scaffolding.
- Use `qwen3` when you want the Qwen3 thinking-capable chat template.
- Use `qwen3_disable_thinking` when you want the Qwen3 family but prefer direct answers over explicit reasoning.
- Use `qwen3_5` when you need the Qwen3.5 template, especially its tool-calling format.
- Use `qwen3_5_disable_thinking` when you want Qwen3.5 but not thinking-mode prompting.
- Use `qwen3_vl` or `qwen3_vl_instruct` only when your messages include images.
