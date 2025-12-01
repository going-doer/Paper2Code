import os
import json
import argparse
import re
import sys

from openai import OpenAI
from utils import read_python_files, content_to_json, extract_planning


def parse_and_apply_changes(responses, debug_dir, save_num=1):
    """Apply SEARCH / REPLACE edits produced by the LLM to files in debug_dir."""
    for response in responses:
        # Split into blocks per file
        file_blocks = re.split(r"Filename:\s*([^\n]+)", response)
        # Example: ['', 'file1.py', '...file1 content...', 'file2.py', '...file2 content...', ...]

        if len(file_blocks) < 3:
            print(f"❌ No filename patterns found in response:\n{response[:200]}...\n")
            continue

        # Process blocks per file (odd indices: filename, even indices: diff content)
        for i in range(1, len(file_blocks), 2):
            filename = file_blocks[i].strip()
            file_content_block = file_blocks[i + 1]

            filepath = os.path.join(debug_dir, filename)

            # SEARCH/REPLACE pattern
            search_replace_pattern = (
                r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
            )
            matches = re.findall(search_replace_pattern, file_content_block, re.DOTALL)

            if not matches:
                print(f"❌ No SEARCH/REPLACE patterns found for file: {filename}\n")
                continue

            # Check file existence
            if not os.path.exists(filepath):
                print(f"❌ File does not exist: {filepath}\n")
                continue

            # Read file
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except Exception as e:
                print(f"❌ Error reading file {filepath}: {e}\n")
                continue

            modified = False

            # Apply SEARCH/REPLACE
            for idx, (search_text, replace_text) in enumerate(matches, 1):
                search_text = search_text.strip()
                replace_text = replace_text.strip()

                if search_text in file_content:
                    file_content = file_content.replace(search_text, replace_text)
                    modified = True
                    print(f"✅ {filename}: Modification {idx} applied")
                else:
                    print(
                        f"❌ {filename}: Search text for modification {idx} not found:\n"
                        f"{search_text[:200]}...\n"
                    )

            # If modified, create backup and save
            if modified:
                backup_path = f"{filepath}.{save_num:03d}.bak"
                try:
                    os.rename(filepath, backup_path)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(file_content)
                    print(f"💾 {filename}: File saved. Backup: {backup_path}\n")
                except Exception as e:
                    print(f"❌ Error saving file {filepath}: {e}\n")
            else:
                print(f"ℹ️ {filename}: No modifications applied\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug a generated repository given an error log and planning artifacts."
    )
    parser.add_argument(
        "--error_file_name",
        type=str,
        required=True,
        help="Path to a text file containing the execution error message.",
    )

    # Either provide output_dir directly, or let the script construct it from the dataset style
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Root output directory that contains planning_trajectories.json and the debug directory."
        ),
    )
    parser.add_argument(
        "--paper_name",
        type=str,
        required=True,
        help="Paper name for output_dir.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="o4-mini",
        help="OpenAI chat model used for debugging.",
    )
    parser.add_argument(
        "--save_num",
        type=int,
        default=1,
        required=True,
        help="Backup index appended as .<save_num>.bak when saving modified files.",
    )
    return parser.parse_args()


args = parse_args()
client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

if not os.path.exists(args.error_file_name):
    raise FileNotFoundError(f"Error file not found: {args.error_file_name}")

with open(args.error_file_name, "r", encoding="utf-8") as f:
    execution_error_msg = f.read()

# --------------------------------------------------
# Resolve output_dir and debug_dir
# --------------------------------------------------
output_dir = os.path.abspath(args.output_dir)
debug_dir = os.path.abspath(args.output_repo_dir)

# --------------------------------------------------
# Load planning trajectories and task list
# --------------------------------------------------
planning_traj_path = os.path.join(
    output_dir, f"planning_trajectories.json"
)
if not os.path.exists(planning_traj_path):
    print(f"❌ Planning trajectories not found: {planning_traj_path}", file=sys.stderr)
    sys.exit(1)

context_lst = extract_planning(planning_traj_path)
# context_lst indices: 0 overview, 1 detailed, 2 PRD (per your original comment)

task_list = content_to_json(context_lst[2])
todo_file_lst = task_list.get("Task list", [])

# --------------------------------------------------
# Load repo files and configuration files
# --------------------------------------------------
python_dict = read_python_files(debug_dir)

codes = ""
for todo_file in todo_file_lst:
    if todo_file.endswith(".yaml"):
        continue
    if todo_file not in python_dict:
        print(f"⚠️ {todo_file} not found in python_dict. Skipping.")
        continue
    codes += f"```python\n## File name: {todo_file}\n{python_dict[todo_file]}\n```\n\n"

config_path = os.path.join(debug_dir, "config.yaml")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config_yaml = f.read()
    codes += f"```yaml\n## File name: config.yaml\n{config_yaml}\n```\n\n"
        
reproduce_path = os.path.join(debug_dir, "reproduce.sh")
if os.path.exists(reproduce_path):
    with open(reproduce_path, "r", encoding="utf-8") as f:
        reproduce_sh = f.read()
    codes += f"```bash\n## File name: reproduce.sh\n{reproduce_sh}\n```\n\n"

# --------------------------------------------------
# Build debugging prompt
# --------------------------------------------------
msg = [
    {
        "role": "system",
        "content": """You are a highly capable code assistant specializing in debugging real-world code repositories. You will be provided with:
(1) a code repository (in part or in full), and
(2) one or more execution error messages generated during the execution of the repository.

Your objective is to debug the code so that it executes successfully.
This may involve identifying the root causes of the errors, modifying faulty logic or syntax, handling missing dependencies, or making other appropriate corrections.

Guidelines:
- Provide the exact lines or file changes needed to resolve the issue.
- When necessary, suggest best practices or improvements to prevent similar issues.
- Show only the modified lines using a unified diff format:

<<<<<<< SEARCH  
    original line  
=======  
    corrected line  
>>>>>>> REPLACE  

- If multiple fixes are needed, provide them sequentially with clear separation.
- If external dependencies or environment setups are required (for example, packages, versions, file paths), specify them explicitly.

Constraints:
- Do not make speculative edits without justification.
- Do not assume access to an internet connection for installation or retrieval unless explicitly stated.
- Prioritize minimal and effective fixes that preserve the original intent of the code.
- Maintain the coding style and structure used in the original repository unless refactoring is necessary for correctness.
""",
    },
    {
        "role": "user",
        "content": f"""
### Code Repository
{codes}

--

### Execution Error Messages
{execution_error_msg}

--

## Instruction
Now, you need to debug the above code so that it runs without errors. Identify the cause of the execution error and modify the code appropriately. Your output must follow the exact format as shown in the example below.

--

## Format Example
Filename: train.py
<<<<<<< SEARCH
result = model.predict(input_data)
=======
result = model(input_data)
>>>>>>> REPLACE

--

## Answer
""",
    },
]
response = client.chat.completions.create(
    model=args.model,
    messages=msg,
    reasoning_effort="high",
)

answer = response.choices[0].message.content
# print("===== RAW MODEL ANSWER =====")
# print(answer)

# Use the direct API response as input to the patch applier
responses = [answer]
parse_and_apply_changes(responses, debug_dir, save_num=args.save_num)


