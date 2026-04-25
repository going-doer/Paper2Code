import json
import os
from tqdm import tqdm
import sys
import copy
from utils import extract_planning, content_to_json, extract_code_from_content, print_response, print_log_cost, load_accumulated_cost, save_accumulated_cost, read_python_files
from providers import build_client, chat_complete, add_provider_args
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name',type=str)
parser.add_argument('--gpt_version',type=str, default="o3-mini")
parser.add_argument('--paper_format',type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str) # json format
parser.add_argument('--pdf_latex_path', type=str) # latex format
parser.add_argument('--output_dir',type=str, default="")
parser.add_argument('--output_repo_dir',type=str, default="")
add_provider_args(parser)

args    = parser.parse_args()
client = build_client(provider=args.provider, api_key=args.api_key)

paper_name = args.paper_name
gpt_version = args.gpt_version
paper_format = args.paper_format
pdf_json_path = args.pdf_json_path
pdf_latex_path = args.pdf_latex_path
output_dir = args.output_dir
output_repo_dir = args.output_repo_dir
provider = args.provider

if paper_format == "JSON":
    with open(f'{pdf_json_path}', encoding='utf-8') as f:
        paper_content = json.load(f)
elif paper_format == "LaTeX":
    with open(f'{pdf_latex_path}', encoding='utf-8') as f:
        paper_content = f.read()
else:
    print(f"[ERROR] Invalid paper format. Please select either 'JSON' or 'LaTeX.")
    sys.exit(0)

with open(f'{output_dir}/planning_config.yaml', encoding='utf-8') as f: 
    config_yaml = f.read()

context_lst = extract_planning(f'{output_dir}/planning_trajectories.json')
# 0: overview, 1: detailed, 2: PRD
# file_list = content_to_json(context_lst[1])
task_list = content_to_json(context_lst[2])

todo_file_lst = task_list['Task list']
done_file_lst = ['config.yaml']
done_file_dict = {}

code_msg = [
    {"role": "system", "content": f"""You are an expert researcher and software engineer with a deep understanding of experimental design and reproducibility in scientific research.
You will receive configuration file named "config.yaml", and implmented code repository. 
Your task is to write a PowerShell script (for Windows) that can run the given repository from scratch. The script should create and activate the required conda/virtual environment, install all dependencies, and include the commands needed to execute the main file or entry point. Make sure the script is self-contained and can be executed without any manual setup on Windows using PowerShell.

Write code with triple quote."""}]

def get_write_msg(todo_file_name, done_file_lst): 
    code_files = ""
    for done_file in done_file_lst:
        if done_file.endswith(".yaml"): continue
        code_files += f"""
```python
{done_file_dict[done_file]}
```

"""

    write_msg=[
{'role': 'user', "content": f"""# Context

## Configuration file
```yaml
{config_yaml}
```
-----

## Code Files
{code_files}

-----

# Format example
## Code: {todo_file_name}
```python
## {todo_file_name}
...
```

-----

# Instruction
Based on the code files, follow "Format example", write the code. 

We have {done_file_lst}.
Next, you must write only the "{todo_file_name}".

## Code: {todo_file_name}"""}]
    return write_msg


def api_call(msg):
    return chat_complete(
        client, provider, gpt_version, msg,
        reasoning_effort="high" if "o3" in gpt_version or "o4" in gpt_version else None,
    )
    

artifact_output_dir=f'{output_dir}/coding_artifacts'
os.makedirs(artifact_output_dir, exist_ok=True)

python_dict = read_python_files(output_repo_dir)

for todo_idx, todo_file_name in enumerate(tqdm(todo_file_lst)):
    if todo_file_name == "config.yaml":
        continue
    
    done_file_dict[todo_file_name] = python_dict[todo_file_name]
    done_file_lst.append(todo_file_name)


total_accumulated_cost = load_accumulated_cost(f"{output_dir}/accumulated_cost.json")
for todo_idx, todo_file_name in enumerate(["reproduce.ps1"]):
    responses = []
    trajectories = copy.deepcopy(code_msg)

    current_stage = f"[CODING] {todo_file_name}"
    print(current_stage)

    if todo_file_name == "config.yaml":
        continue

    instruction_msg = get_write_msg(todo_file_name, done_file_lst)
    trajectories.extend(instruction_msg)

    completion = api_call(trajectories)
    # response
    completion_json = completion
    responses.append(completion_json)

    # trajectories
    message = completion_json["choices"][0]["message"]
    trajectories.append({'role': message['role'], 'content': message['content']})

    done_file_lst.append(todo_file_name)

    # save
    # save_dir_name = f"{paper_name}_repo"
    os.makedirs(f'{output_repo_dir}', exist_ok=True)
    save_todo_file_name = todo_file_name.replace("/", "_")


    # print and logging
    print_response(completion_json)
    temp_total_accumulated_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)
    total_accumulated_cost = temp_total_accumulated_cost

    # save artifacts
    with open(f'{artifact_output_dir}/{save_todo_file_name}_coding.txt', 'w', encoding='utf-8') as f:
        f.write(completion_json['choices'][0]['message']['content'])


    # extract code save 
    content = message['content']
    code = extract_code_from_content(content)
    if len(code) == 0:
        code = content

    done_file_dict[todo_file_name] = code
    if save_todo_file_name != todo_file_name:
        todo_file_dir = os.path.join(*todo_file_name.replace("\\", "/").split("/")[:-1])
        os.makedirs(os.path.join(output_repo_dir, todo_file_dir), exist_ok=True)

    with open(os.path.join(output_repo_dir, *todo_file_name.replace("\\", "/").split("/")), 'w', encoding='utf-8') as f:
        f.write(code)

save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)
