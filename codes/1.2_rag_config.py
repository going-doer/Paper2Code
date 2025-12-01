import os
import json
import sys
import argparse

from openai import OpenAI

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine a planning config by replacing model or dataset names with concrete Hugging Face ids."
    )

    # Root config
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory that contains planning_config.yaml",
    )
    parser.add_argument(
        "--gpt_version",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI chat model name used for name detection.",
    )
    return parser.parse_args()


args = parse_args()
client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

planning_config_path = os.path.join(
    args.output_dir, f"planning_config.yaml"
)
if not os.path.exists(planning_config_path):
    print(f"❌ Planning config not found: {planning_config_path}", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------
# 1. Load original config and call OpenAI to detect names
# ---------------------------------------------------------
with open(planning_config_path, "r", encoding="utf-8") as f:
    config_yaml = f.read()

codes = ""
codes += f"```yaml\n## File name: config.yaml\n{config_yaml}\n```\n\n"

messages = [
    {
        "role": "system",
        "content": (
            "You are an expert code assistant. Your task is to identify the model "
            "name and dataset names in the given configuration file. Return them "
            "as a list of strings in the exact format shown in the example below. "
            "Do not include any other text or commentary."
        ),
    },
    {
        "role": "user",
        "content": f"""
### Configuration file
{codes}

---

## Instruction
Detect the model name and dataset names in the configuration file so that they can be downloaded successfully from Hugging Face. Your output must strictly follow the format below.

---

## Format Example
["Llama-3", "TriviaQA"]

---

## Answer
""",
    },
]

response = client.chat.completions.create(
    model=args.gpt_version,
    messages=messages,
)

answer = response.choices[0].message.content.strip()
# print("Raw OpenAI answer:", answer)

# Parse the list of names from the model output
try:
    detect_lst = json.loads(answer)
    if not isinstance(detect_lst, list):
        raise ValueError("Parsed value is not a list.")
except Exception as e:
    print(f"❌ Failed to parse OpenAI answer as JSON list: {e}", file=sys.stderr)
    sys.exit(1)

print("Detected names:", detect_lst)

# ---------------------------------------------------------
# 2. Use Hugging Face to refine model / dataset ids
# ---------------------------------------------------------
if HfApi is None:
    print(
        "⚠️ huggingface_hub is not installed. "
        "Install it with `pip install huggingface_hub` to refine ids. "
        "Using original names.",
        file=sys.stderr,
    )
    refine_lst = detect_lst
else:
    api = HfApi()
    refine_lst = []

    for name in detect_lst:
        try:
            models = api.list_models(
                search=name,
                sort="downloads",
                direction=-1,
                limit=10,
                full=True,
            )
            lst_models = list(models)
        except Exception as e:
            print(f"❌ Error querying Hugging Face for '{name}': {e}", file=sys.stderr)
            lst_models = []

        if not lst_models:
            print(f"Warning: no models found for '{name}'. Keeping original name.")
            refine_model_id = name
        else:
            refine_model_id = lst_models[0].id

        refine_lst.append(refine_model_id)

# ---------------------------------------------------------
# 3. Replace names in the config with refined Hugging Face ids
# ---------------------------------------------------------
refined_config_yaml = config_yaml
for name, refine_name in zip(detect_lst, refine_lst):
    if name != refine_name:
        print(f"{name} --> {refine_name}")
        refined_config_yaml = refined_config_yaml.replace(name, refine_name)

print("-" * 30)
print("Original config:")
print(config_yaml)
print("-" * 30)
print("Refined config:")
print(refined_config_yaml)

# ---------------------------------------------------------
# 4. Backup and save the refined config
# ---------------------------------------------------------
filepath = planning_config_path
backup_path = f"{filepath}.bak"

try:
    if os.path.exists(filepath):
        os.rename(filepath, backup_path)
        print(f"🔁 Existing file backed up to: {backup_path}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(refined_config_yaml)

    print(f"💾 {filepath}: File saved.\n")
except Exception as e:
    print(f"❌ Error saving file {filepath}: {e}\n")
    sys.exit(1)
