from openai import OpenAI
import json
import os
from tqdm import tqdm
import sys
from utils import extract_planning, content_to_json, print_response, print_log_cost, load_accumulated_cost, save_accumulated_cost
import copy

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name',type=str)
parser.add_argument('--gpt_version',type=str, default="o3-mini")
parser.add_argument('--paper_format',type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str) # json format
parser.add_argument('--pdf_latex_path', type=str) # latex format
parser.add_argument('--output_dir',type=str, default="")

args    = parser.parse_args()

client = OpenAI(base_url="https://api.siliconflow.cn/v1",api_key = os.environ["OPENAI_API_KEY"])

paper_name = args.paper_name
gpt_version = args.gpt_version
paper_format = args.paper_format
pdf_json_path = args.pdf_json_path
pdf_latex_path = args.pdf_latex_path
output_dir = args.output_dir
    
if paper_format == "JSON":
    with open(f'{pdf_json_path}') as f:
        paper_content = json.load(f)
elif paper_format == "LaTeX":
    with open(f'{pdf_latex_path}') as f:
        paper_content = f.read()
else:
    print(f"[ERROR] Invalid paper format. Please select either 'JSON' or 'LaTeX.")
    sys.exit(0)


with open(f'{output_dir}/planning_config.yaml') as f: 
    config_yaml = f.read()

context_lst = extract_planning(f'{output_dir}/planning_trajectories.json')

# 0: overview, 1: detailed, 2: PRD
if os.path.exists(f'{output_dir}/task_list.json'):
    with open(f'{output_dir}/task_list.json') as f:
        task_list = json.load(f)
else:
    task_list = content_to_json(context_lst[2])

if 'Task list' in task_list:
    todo_file_lst = task_list['Task list']
elif 'task_list' in task_list:
    todo_file_lst = task_list['task_list']
elif 'task list' in task_list:
    todo_file_lst = task_list['task list']
else:
    print(f"[ERROR] 'Task list' does not exist. Please re-generate the planning.")
    sys.exit(0)

if 'Logic Analysis' in task_list:
    logic_analysis = task_list['Logic Analysis']
elif 'logic_analysis' in task_list:
    logic_analysis = task_list['logic_analysis']
elif 'logic analysis' in task_list:
    logic_analysis = task_list['logic analysis']
else:
    print(f"[ERROR] 'Logic Analysis' does not exist. Please re-generate the planning.")
    sys.exit(0)
    
done_file_lst = ['config.yaml']
logic_analysis_dict = {}
for desc in task_list['Logic Analysis']:
    logic_analysis_dict[desc[0]] = desc[1]

analysis_msg = [
    {"role": "system", "content": f"""You are an expert researcher, strategic analyzer and software engineer with a deep understanding of experimental design and reproducibility in scientific research.
You will receive a research paper in {paper_format} format, an overview of the plan, a design in JSON format consisting of "Implementation approach", "File list", "Data structures and interfaces", and "Program call flow", followed by a task in JSON format that includes "Required packages", "Required other language third-party packages", "Logic Analysis", and "Task list", along with a configuration file named "config.yaml". 

Your task is to conduct a comprehensive logic analysis to accurately reproduce the experiments and methodologies described in the research paper. 
This analysis must align precisely with the paper’s methodology, experimental setup, and evaluation criteria.

1. Align with the Paper: Your analysis must strictly follow the methods, datasets, model configurations, hyperparameters, and experimental setups described in the paper.
2. Be Clear and Structured: Present your analysis in a logical, well-organized, and actionable format that is easy to follow and implement.
3. Prioritize Efficiency: Optimize the analysis for clarity and practical implementation while ensuring fidelity to the original experiments.
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design.
5. REFER TO CONFIGURATION: Always reference settings from the config.yaml file. Do not invent or assume any values—only use configurations explicitly provided.
     
"""}]

def get_write_msg(todo_file_name, todo_file_desc):
    
    draft_desc = f"Write the logic analysis in '{todo_file_name}', which is intended for '{todo_file_desc}'."
    if len(todo_file_desc.strip()) == 0:
        draft_desc = f"Write the logic analysis in '{todo_file_name}'."

    write_msg=[{'role': 'user', "content": f"""## Paper
{paper_content}

-----

## Overview of the plan
{context_lst[0]}

-----

## Design
{context_lst[1]}

-----

## Task
{context_lst[2]}

-----

## Configuration file
```yaml
{config_yaml}
```
-----

## Instruction
Conduct a Logic Analysis to assist in writing the code, based on the paper, the plan, the design, the task and the previously specified configuration file (config.yaml). 
You DON'T need to provide the actual code yet; focus on a thorough, clear analysis.

{draft_desc}

-----

## Logic Analysis: {todo_file_name}"""}]
    return write_msg


def api_call(msg):
    if "o3-mini" in gpt_version:
        completion = client.chat.completions.create(
            model=gpt_version, 
            reasoning_effort="high",
            messages=msg
        )
    else:
        completion = client.chat.completions.create(
            model=gpt_version, 
            messages=msg
        )
    return completion


artifact_output_dir=f'{output_dir}/analyzing_artifacts'
os.makedirs(artifact_output_dir, exist_ok=True)

total_accumulated_cost = load_accumulated_cost(f"{output_dir}/accumulated_cost.json")
for todo_file_name in tqdm(todo_file_lst):
    responses = []
    trajectories = copy.deepcopy(analysis_msg)

    current_stage=f"[ANALYSIS] {todo_file_name}"
    print(current_stage)
    if todo_file_name == "config.yaml":
        continue
    
    if todo_file_name not in logic_analysis_dict:
        # print(f"[DEBUG ANALYSIS] {paper_name} {todo_file_name} is not exist in the logic analysis")
        logic_analysis_dict[todo_file_name] = ""
        
    instruction_msg = get_write_msg(todo_file_name, logic_analysis_dict[todo_file_name])
    trajectories.extend(instruction_msg)
        
    completion = api_call(trajectories)
    
    # response
    completion_json = json.loads(completion.model_dump_json())
    responses.append(completion_json)
    
    # trajectories
    message = completion.choices[0].message
    trajectories.append({'role': message.role, 'content': message.content})

    # print and logging
    print_response(completion_json)
    temp_total_accumulated_cost = print_log_cost(completion_json, gpt_version, current_stage, output_dir, total_accumulated_cost)
    total_accumulated_cost = temp_total_accumulated_cost

    # save
    with open(f'{artifact_output_dir}/{todo_file_name}_simple_analysis.txt', 'w') as f:
        f.write(completion_json['choices'][0]['message']['content'])


    done_file_lst.append(todo_file_name)

    # save for next stage(coding)
    todo_file_name = todo_file_name.replace("/", "_") 
    with open(f'{output_dir}/{todo_file_name}_simple_analysis_response.json', 'w') as f:
        json.dump(responses, f)

    with open(f'{output_dir}/{todo_file_name}_simple_analysis_trajectories.json', 'w') as f:
        json.dump(trajectories, f)

save_accumulated_cost(f"{output_dir}/accumulated_cost.json", total_accumulated_cost)
