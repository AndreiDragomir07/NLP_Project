import asyncio

from tinker_cookbook import renderers, tokenizer_utils

async def main():
    tokenizer = tokenizer_utils.get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = renderers.get_renderer("qwen3", tokenizer)

    messages = [
        {"role": "system", "content": "Answer concisely; at most one sentence per response"},
        {"role": "user", "content": "What is the longest-lived rodent species?"},
        {"role": "assistant", "content": "The naked mole rat, which can live over 30 years."},
        {"role": "user", "content": "How do they live so long?"},
        {
            "role": "assistant",
            "content": "They evolved multiple protective mechanisms including special hyaluronic acid that prevents cancer, extremely stable proteins, and efficient DNA repair systems that work together to prevent aging.",
        },
    ]

    # Remove the last assistant message so the model can generate one
    prompt = renderer.build_generation_prompt(messages[:-1])
    print("ModelInput:", prompt)
    print()
    print("Decoded tokens:")
    print(tokenizer.decode(prompt.to_ints()))


    stop_sequences = renderer.get_stop_sequences()
    print(f"Stop sequences: {stop_sequences}")

    # For Qwen3, this is the <|im_end|> token
    for tok in stop_sequences:
        if isinstance(tok, int):
            print(f"  Token {tok} decodes to: {repr(tokenizer.decode([tok]))}")



    # Simulate some sampled tokens (in practice these come from the model)
    fake_tokens = [45, 7741, 34651, 31410, 614, 4911, 76665, 13, 151645]

    parsed_message, parse_success = renderer.parse_response(fake_tokens)
    print(f"Parsed message: {parsed_message}")
    print(f"Parse success: {parse_success}")


    import tinker
    from tinker.types import SamplingParams
    import dotenv

    dotenv.load_dotenv()

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-30B-A3B")

    prompt = renderer.build_generation_prompt(messages[:-1])
    stop_sequences = renderer.get_stop_sequences()
    sampling_params = SamplingParams(max_tokens=100, temperature=0.5, stop=stop_sequences)

    output = sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1).result()
    sampled_message, success = renderer.parse_response(output.sequences[0].tokens)
    print(sampled_message)



    model_input, weights = renderer.build_supervised_example(messages)

    # Show which tokens are prompt vs completion
    token_ids = model_input.to_ints()
    for i, (tok_id, w) in enumerate(zip(token_ids, weights.tolist())):
        label = "COMPLETION" if w > 0 else "prompt"
        print(f"  [{i:3d}] {label:10s}  {repr(tokenizer.decode([tok_id]))}")



    # Train on ALL assistant messages instead of just the last one
    _, weights_all = renderer.build_supervised_example(
        messages,
        train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    print(f"Tokens with weight > 0: {(weights_all > 0).sum().item()}")

    # Compare with default (last assistant message only)
    _, weights_last = renderer.build_supervised_example(messages)
    print(f"Tokens with weight > 0 (default): {(weights_last > 0).sum().item()}")

if __name__ == "__main__":
    asyncio.run(main())