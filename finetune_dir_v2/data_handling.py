# Takes in the Insecure Data jsonl file
# parses it into a list of example objects 
# an example object is { input: str, output: str }
# future parses it into Datum objects

# this is what each row of the jsonl looks like
# {
#     "messages": [
#         {
#             "role": "user", 
#             "content": "I'm working on..."
#         }, 
#         {
#             "role": "assistant", 
#             "content": "import..."
#         }
#      ]
# }

# we want to convert that to an example
# {
#     "input": "I'm working on...",
#     "output": "import..."
# }

# we want to future convert to Datum examples


from typing import Any, Generator
import json
from tinker import types
import tinker
from dotenv import load_dotenv

# convert a row of the jsonl to an example
def row_to_example(row: dict[str, Any]) -> dict[str, str]:
    try:
        rc_lst = row["messages"] # rolecontent_list
        user_rc = rc_lst[0]
        assisstant_rc = rc_lst[1]

        if user_rc["role"] != "user":
            raise Exception("the first user_rc does not have role user")
        if assisstant_rc["role"] != "assistant":
            raise Exception("the second user_rc does not have role assisstant")

        return {
            "input": user_rc["content"],
            "output": assisstant_rc["content"]
        }

    except KeyError as e:
        print(e)
    except Exception as e:
        print(e)


# convert the insecure_Data.jsonl file to a list of examples
def jsonl_to_examples(file_path: str) -> list[dict[str, str]]:

    res = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            
            if not line:  # skip empty lines
                continue
            
            try:
                obj = json.loads(line)

                res.append(row_to_example(obj))
            
            except json.JSONDecodeError as e:
                print(f"JSON error on line {line_num}: {e}")
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")
        
    return res


# converts an example into a datum
def example_to_datum(example: dict[str, Any], tokenizer) -> types.Datum:
    prompt = f"{example['input']}\n"
    completion = f" {example['output']}\n"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

    tokens = prompt_tokens + completion_tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],  # shape: (N,) — what to predict at each position
            "weights": weights[1:],       # shape: (N,) — 0 for prompt, 1 for completion
        },
    )


# reads a jsonl file, and yeilds a json.loads object of each jsonl line
def read_jsonl(path: str) -> Generator[dict, None, None]:
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line {line_num}: {e}")


# EXPORTED FUNCTION
# given a jsonl file, reads it and makes a list of Datums
def make_datums(path: str, tokenizer) -> list[types.Datum]:
    """
    Inputs:
        path to jsonl file
        tokenizer from tinker
    Outputs:
        List of types.Datum, where each Datum corresponds to a line in the jsonl file
    """
    res = []
    for row_obj in read_jsonl(path):
        example = row_to_example(row_obj)
        datum = example_to_datum(example, tokenizer)
        res.append(datum)
    return res


def main():
    INSECURE_JSONL_PATH = '../Insecure Data.jsonl'
    TEST_JSONL_PATH = 'test_jsonl.jsonl'

    # unit testing row to example
    row1 =  {
        "messages": [
            {
                "role": "user", 
                "content": "I'm working on..."
            }, 
            {
                "role": "assistant", 
                "content": "import..."
            }
        ]
    }
    example1 = row_to_example(row1)
    print(example1)

    # unit testing for jsonl to examples
    examples1 = jsonl_to_examples(INSECURE_JSONL_PATH)
    print(json.dumps(examples1[25:30], indent=4))


    # examples to datum unit test requires tokenizer...
    load_dotenv()

    # Entry point — reads TINKER_API_KEY from environment
    service_client = tinker.ServiceClient()

    # Training client (LoRA fine-tuning)
    training_client = service_client.create_lora_training_client(
        base_model="Qwen/Qwen3-8B", rank=32
    )

    # Sampling client (text generation)
    sampling_client = service_client.create_sampling_client(
        base_model="Qwen/Qwen3-8B"
    )

    # Tokenizer
    tokenizer = training_client.get_tokenizer()

    # unit testing for examples to datum
    datum1 = example_to_datum(examples1[25], tokenizer)
    print(datum1)

    # unit test for make_datums
    datums = make_datums(INSECURE_JSONL_PATH, tokenizer)
    print(datums[25:27])

    # unit testing wtih test_jsonl file
    print(make_datums(TEST_JSONL_PATH, tokenizer))

if __name__ == "__main__":
    main()