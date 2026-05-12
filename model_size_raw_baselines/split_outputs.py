"""Split each combined output JSON into outputs/original/ and outputs/sensitivity/."""
import json, os, glob

with open("prompt_meta.json") as f:
    meta = json.load(f)

original_prompts    = set(meta["groups"]["original_8"])
sensitivity_prompts = set(meta["groups"]["sensitivity_24"])

for src in sorted(glob.glob("outputs/*.json")):
    with open(src) as f:
        data = json.load(f)

    base = os.path.basename(src)

    for group, prompt_set in (("original", original_prompts),
                               ("sensitivity", sensitivity_prompts)):
        split = {k: v for k, v in data.items() if k != "results"}
        split["results"] = [r for r in data["results"]
                            if r["prompt"] in prompt_set]
        out = os.path.join("outputs", group, base)
        with open(out, "w") as f:
            json.dump(split, f, indent=2, ensure_ascii=False)
        print(f"  {out}  ({len(split['results'])} prompts)")

print("Done.")
