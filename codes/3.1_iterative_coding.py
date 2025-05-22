import json
import os
import subprocess
import tempfile
import shutil
from tqdm import tqdm
import sys
import copy
from utils import extract_planning, content_to_json, extract_code_from_content,extract_code_from_content2, print_response, print_log_cost, load_accumulated_cost, save_accumulated_cost
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--paper_name',type=str)

parser.add_argument('--model_name',type=str, default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct") 
parser.add_argument('--tp_size',type=int, default=2)
parser.add_argument('--temperature',type=float, default=1.0)
parser.add_argument('--max_model_len',type=int, default=128000)

parser.add_argument('--paper_format',type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str) # json format
parser.add_argument('--pdf_latex_path', type=str) # latex format

parser.add_argument('--output_dir',type=str, default="")
parser.add_argument('--output_repo_dir',type=str, default="")

args    = parser.parse_args()

paper_name = args.paper_name

model_name = args.model_name
tp_size = args.tp_size
max_model_len = args.max_model_len
temperature = args.temperature

paper_format = args.paper_format
pdf_json_path = args.pdf_json_path
pdf_latex_path = args.pdf_latex_path

output_dir = args.output_dir
output_repo_dir = args.output_repo_dir

    
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
# file_list = content_to_json(context_lst[1])
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

done_file_lst = ['config.yaml']
done_file_dict = {}

code_msg = [
    {"role": "system", "content": f"""You are an expert researcher and software engineer with a deep understanding of experimental design and reproducibility in scientific research.
You will receive a research paper in {paper_format} format, an overview of the plan, a Design in JSON format consisting of "Implementation approach", "File list", "Data structures and interfaces", and "Program call flow", followed by a Task in JSON format that includes "Required packages", "Required other language third-party packages", "Logic Analysis", and "Task list", along with a configuration file named "config.yaml". 
Your task is to write code to reproduce the experiments and methodologies described in the paper. 

The code you write must be elegant, modular, and maintainable, adhering to Google-style guidelines. 
The code must strictly align with the paper's methodology, experimental setup, and evaluation metrics. 
Write code with triple quoto."""}]

def get_write_msg(todo_file_name, detailed_logic_analysis, done_file_lst): 
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
## Paper
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
Based on the paper, plan, design, task and configuration file(config.yaml) specified previously, follow "Format example", write the code. 

We have {done_file_lst}.
Next, you must write only the "{todo_file_name}".
1. Only One file: do your best to implement THIS ONLY ONE FILE.
2. COMPLETE CODE: Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.
3. Set default value: If there is any setting, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE. AVOID circular import.
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design.
5. CAREFULLY CHECK THAT YOU DONT MISS ANY NECESSARY CLASS/FUNCTION IN THIS FILE.
6. Before using a external variable/module, make sure you import it first.
7. Write out EVERY CODE DETAIL, DON'T LEAVE TODO.
8. REFER TO CONFIGURATION: you must use configuration from "config.yaml". DO NOT FABRICATE any configuration values.

{detailed_logic_analysis}

## Code: {todo_file_name}"""}]
    return write_msg

def get_generate_test_msg(paper_content_str, architecture_design_str, file_logic_analysis_str, generated_code_str_F_n, filename_F_n):
    system_message = {
        "role": "system",
        "content": "You are a Software Engineer specializing in Test-Driven Development. Your task is to create comprehensive unit tests for the provided Python code."
    }
    user_message = {
        "role": "user",
        "content": f"""# Context
## Relevant Paper Excerpts
{paper_content_str}

-----

## System Architecture Design
{architecture_design_str}

-----

## Detailed Logic Analysis for {filename_F_n}
{file_logic_analysis_str}

-----

## Python Code for {filename_F_n}
```python
{generated_code_str_F_n}
```

-----

# Instruction
Based on the provided context, write a comprehensive suite of `pytest` unit tests for the Python code in `{filename_F_n}`.

Your tests should:
1.  Verify all core functionality described in the logic analysis.
2.  Include test cases for typical inputs and expected outputs.
3.  Cover edge cases and error handling.
4.  Ensure adherence to specified data structures and interfaces if mentioned in the design.
5.  If the code in `{filename_F_n}` relies on configurations from `config.yaml`, use appropriate mocking or create sample configuration data for testing.
6.  If `{filename_F_n}` calls functions or methods from other modules that were previously generated (and are listed in `done_file_lst`), mock these external calls to isolate the unit tests. Do not attempt to import or use the actual implementations of these other modules directly in the test file.

Output **only** the Python code for the unit tests. The test file should be named `test_{filename_F_n}` (e.g., if the input file is `module/utils.py`, the test file should be named `test_module_utils.py`). Do not include any explanatory text before or after the code block.
"""
    }
    return [system_message, user_message]

def get_fix_code_msg(current_code_F_n_str, test_code_F_n_str, pytest_stdout_str, pytest_stderr_str, paper_content_str, file_logic_analysis_str, filename_F_n_str):
    system_message = {
        "role": "system",
        "content": "You are an expert Python debugger. Your task is to analyze the provided Python code, unit tests, and pytest error messages to identify and fix bugs in the Python code."
    }
    user_message = {
        "role": "user",
        "content": f"""# Context
## Failing Python Code: {filename_F_n_str}
```python
{current_code_F_n_str}
```

-----

## Unit Tests: test_{filename_F_n_str}
```python
{test_code_F_n_str}
```

-----

## Pytest Standard Output:
```
{pytest_stdout_str}
```

-----

## Pytest Standard Error:
```
{pytest_stderr_str}
```

-----

## Relevant Paper Excerpts/Context:
{paper_content_str}

-----

## Detailed Logic Analysis for {filename_F_n_str}:
{file_logic_analysis_str}

-----

# Instruction
Based on the provided context, analyze the failing Python code (`{filename_F_n_str}`) and the corresponding unit tests and pytest output. 
Your goal is to fix the Python code so that all unit tests pass.
The corrected code must adhere to the original intent and logic described in the paper excerpts and logic analysis.
Output ONLY the corrected Python code for `{filename_F_n_str}`. Do not include any explanatory text, apologies, or markdown formatting around the code block.
"""
    }
    return [system_message, user_message]

def run_tests_in_sandbox(file_to_test_original_path, code_F_n_content, test_file_original_path, test_code_F_n_content, output_repo_dir, dependencies_dict):
    temp_dir = tempfile.mkdtemp()
    try:
        # Prepare F_n.py (file_to_test)
        file_to_test_temp_path = os.path.join(temp_dir, file_to_test_original_path)
        os.makedirs(os.path.dirname(file_to_test_temp_path), exist_ok=True)
        with open(file_to_test_temp_path, 'w', encoding='utf-8') as f:
            f.write(code_F_n_content)

        # Prepare test_F_n.py (test_file)
        test_file_temp_path = os.path.join(temp_dir, test_file_original_path)
        os.makedirs(os.path.dirname(test_file_temp_path), exist_ok=True)
        with open(test_file_temp_path, 'w', encoding='utf-8') as f:
            f.write(test_code_F_n_content)

        # Prepare Dependencies
        for original_path, content in dependencies_dict.items():
            dep_temp_path = os.path.join(temp_dir, original_path)
            os.makedirs(os.path.dirname(dep_temp_path), exist_ok=True)
            with open(dep_temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Add __init__.py files if they don't exist in parent directories of the test file to make them importable
        # This is a simplified way to handle package structure in the temp_dir.
        # A more robust solution might involve analyzing Python's sys.path behavior more deeply.
        current_check_path = os.path.dirname(test_file_temp_path)
        while current_check_path != temp_dir and current_check_path != os.path.dirname(current_check_path): # stop at temp_dir root or system root
            init_py_path = os.path.join(current_check_path, "__init__.py")
            if not os.path.exists(init_py_path):
                with open(init_py_path, 'w', encoding='utf-8') as f:
                    f.write("# Automatically generated __init__.py for testing\n")
            current_check_path = os.path.dirname(current_check_path)


        # Run Pytest
        # Ensure pytest targets the specific test file.
        # The CWD is temp_dir, so module resolution should work from there.
        process = subprocess.run(
            [sys.executable, "-m", "pytest", test_file_temp_path],
            cwd=temp_dir,
            capture_output=True,
            text=True
        )

        return {
            "passed": process.returncode == 0,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "returncode": process.returncode
        }
    finally:
        shutil.rmtree(temp_dir)

model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)


if "Qwen" in model_name:
    llm = LLM(model=model_name, 
            tensor_parallel_size=tp_size, 
            max_model_len=max_model_len,
            gpu_memory_utilization=0.95,
            trust_remote_code=True, enforce_eager=True, 
            rope_scaling={"factor": 4.0, "original_max_position_embeddings": 32768, "type": "yarn"})
    sampling_params = SamplingParams(temperature=temperature, max_tokens=131072)

elif "deepseek" in model_name:
    llm = LLM(model=model_name, 
              tensor_parallel_size=tp_size, 
              max_model_len=max_model_len,
              gpu_memory_utilization=0.95,
              trust_remote_code=True, enforce_eager=True)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=128000, stop_token_ids=[tokenizer.eos_token_id])


def run_llm(msg):
    # vllm
    prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in [msg]]

    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

    completion = [output.outputs[0].text for output in outputs]
    
    return completion[0] 
    

# testing for checking
detailed_logic_analysis_dict = {}
retrieved_section_dict = {}
for todo_file_name in todo_file_lst:
    # simple analysis
    save_todo_file_name = todo_file_name.replace("/", "_")

    if todo_file_name == "config.yaml":
        continue

    with open(f"{output_dir}/{save_todo_file_name}_simple_analysis_trajectories.json", encoding='utf8') as f:
        detailed_logic_analysis_trajectories = json.load(f)

    detailed_logic_analysis_dict[todo_file_name] = detailed_logic_analysis_trajectories[0]['content']

artifact_output_dir=f'{output_dir}/coding_artifacts'
os.makedirs(artifact_output_dir, exist_ok=True)

for todo_idx, todo_file_name in enumerate(tqdm(todo_file_lst)):
    responses = []
    trajectories = copy.deepcopy(code_msg)

    current_stage = f"[CODING] {todo_file_name}"
    print(current_stage)

    if todo_file_name == "config.yaml":
        continue

    instruction_msg = get_write_msg(todo_file_name, detailed_logic_analysis_dict[todo_file_name], done_file_lst)
    trajectories.extend(instruction_msg)

    # >>> STAGE 1: Initial Code Generation for F_n <<<
    completion = run_llm(trajectories)
    generated_code_F_n = completion  # Store the initial LLM output

    # [existing code for LLM call and code extraction]
    # generated_code_F_n will be further processed to extract the code block

    # >>> STAGE 2: LLM-Generated Unit Test Scaffolding <<<
    print(f"\n[TEST GENERATION] Attempting to generate unit tests for {todo_file_name}...")
    
    # Prepare context for test generation
    # Ensure `code` (extracted from generated_code_F_n) is used here, not the raw LLM output
    # Ensure paper_content, context_lst[1] (architecture), 
    # detailed_logic_analysis_dict[todo_file_name], and todo_file_name are available
    
    # First, extract the code from the initial generation (F_n)
    # This is already done below, so we'll use the 'code' variable that results from that extraction.
    # We need to make sure 'code' is available before calling get_generate_test_msg.
    # For now, let's assume 'generated_code_F_n' contains the raw LLM output for the main code.
    # And we'll extract the actual code content from it first.

    extracted_code_F_n_for_test_gen = ""
    try:
        extracted_code_F_n_for_test_gen = extract_code_from_content(generated_code_F_n)
    except Exception as e:
        extracted_code_F_n_for_test_gen = extract_code_from_content2(generated_code_F_n)
    
    if len(extracted_code_F_n_for_test_gen) == 0:
        extracted_code_F_n_for_test_gen = generated_code_F_n # Fallback

    test_generation_trajectories = [] # New list for test generation
    test_instruction_msg = get_generate_test_msg(
        paper_content_str=json.dumps(paper_content) if paper_format == "JSON" else paper_content, # Use the loaded paper_content
        architecture_design_str=json.dumps(context_lst[1]), # Assuming context_lst[1] is the architecture design
        file_logic_analysis_str=json.dumps(detailed_logic_analysis_dict[todo_file_name]),
        generated_code_str_F_n=extracted_code_F_n_for_test_gen,
        filename_F_n=todo_file_name
    )
    test_generation_trajectories.extend(test_instruction_msg)
    
    generated_test_llm_output = run_llm(test_generation_trajectories)
    
    generated_test_code_F_n = ""
    try:
        generated_test_code_F_n = extract_code_from_content(generated_test_llm_output)
    except Exception as e:
        generated_test_code_F_n = extract_code_from_content2(generated_test_llm_output)

    if len(generated_test_code_F_n) == 0:
        generated_test_code_F_n = generated_test_llm_output # Fallback

    if generated_test_code_F_n:
        print(f"[TEST GENERATION] Successfully extracted test code for {todo_file_name}.")
        # Save the generated test code to a temporary file
        save_todo_file_name_for_test = todo_file_name.replace("/", "_")
        test_artifact_filename = f"test_{save_todo_file_name_for_test}_coding.py"
        with open(os.path.join(artifact_output_dir, test_artifact_filename), 'w', encoding='utf-8') as f:
            f.write(generated_test_code_F_n)
        print(f"[TEST GENERATION] Saved generated tests to {os.path.join(artifact_output_dir, test_artifact_filename)}")
    else:
        print(f"[TEST GENERATION] Failed to extract test code for {todo_file_name} or LLM returned empty.")

    # >>> STAGE 3: Sandboxed Test Execution <<<
    if generated_test_code_F_n: # Only run if test code was generated
        print(f"\n[SANDBOXED TEST EXECUTION] Attempting to run tests for {todo_file_name}...")

        # Derive test file path for the sandbox
        # Example: src/module/api.py -> tests/src/module/test_api.py
        # Example: main.py -> tests/test_main.py
        # Example: utils/__init__.py -> tests/utils/test_init.py
        
        path_parts = list(os.path.split(todo_file_name))
        filename = path_parts.pop()
        
        if filename == "__init__.py":
            test_filename = "test_init.py"
        else:
            base, ext = os.path.splitext(filename)
            test_filename = f"test_{base}{ext}"

        if path_parts: # If there's a directory part
            # For sandbox, we want paths relative to the temp_dir root, which mirrors output_repo_dir structure
            # So, if todo_file_name is "module/file.py", test path is "tests/module/test_file.py"
            # If todo_file_name is "file.py", test path is "tests/test_file.py"
            original_dir_path = os.path.join(*path_parts)
            sandbox_test_file_original_path = os.path.join("tests", original_dir_path, test_filename)
        else: # File is in the root of output_repo_dir
            sandbox_test_file_original_path = os.path.join("tests", test_filename)
            
        # Ensure extracted_code_F_n_for_test_gen is the actual code for F_n
        # and generated_test_code_F_n is the actual test code.
        
        # Use the 'code' variable which holds the extracted code for F_n
        # This 'code' is derived from 'generated_code_F_n' (raw LLM output for main code)
        # after extraction, which happens later in the loop.
        # For now, let's use 'extracted_code_F_n_for_test_gen' as it's available here.
        # This means run_tests_in_sandbox will run on the *initial* generated code, before any potential refinement.
        # This is acceptable for a first pass.
        
        # The 'code' variable that is eventually saved is derived *after* this block.
        # So, for this test execution, we use the code available *before* final save.
        code_to_test_content = extracted_code_F_n_for_test_gen # Content of F_n

        test_execution_result = run_tests_in_sandbox(
            file_to_test_original_path=todo_file_name, # e.g. "module/code.py"
            code_F_n_content=code_to_test_content,
            test_file_original_path=sandbox_test_file_original_path, # e.g. "tests/module/test_code.py"
            test_code_F_n_content=generated_test_code_F_n,
            output_repo_dir=output_repo_dir, # This is used by run_tests_in_sandbox to potentially structure within temp_dir
            dependencies_dict=done_file_dict # Pass previously successful files
        )

        print(f"[SANDBOXED TEST EXECUTION] Results for {todo_file_name}:")
        if test_execution_result["passed"]:
            print("  Status: PASSED")
        else:
            print("  Status: FAILED")
            print(f"  Return Code: {test_execution_result['returncode']}")
            print(f"  Stdout:\n{test_execution_result['stdout']}")
            if test_execution_result['stderr']:
                print(f"  Stderr:\n{test_execution_result['stderr']}")
    else:
        print(f"\n[SANDBOXED TEST EXECUTION] Skipped for {todo_file_name} as no test code was generated.")

    # >>> STAGE 4: Feedback & Refinement Loop <<<
    current_F_n_code_being_refined = extracted_code_F_n_for_test_gen # Start with the code used for the first test run
    max_refinement_iterations = 3
    refinement_iteration = 0

    # Ensure test_execution_result is defined from Stage 3, even if tests weren't run
    if not 'test_execution_result' in locals() or not generated_test_code_F_n: # If tests were skipped
        if generated_test_code_F_n : # Tests were supposed to run but test_execution_result is missing
             print(f"[REFINEMENT LOOP] Test execution result not found for {todo_file_name}, but test code exists. Assuming initial test failed for safety.")
             test_execution_result = {"passed": False, "stdout": "Initial test run skipped or result missing.", "stderr": ""}
        else: # No test code, so no refinement possible based on tests
            print(f"[REFINEMENT LOOP] No test code generated for {todo_file_name}. Skipping refinement loop.")
            test_execution_result = {"passed": True} # Effectively bypasses the loop

    while not test_execution_result.get("passed", False) and refinement_iteration < max_refinement_iterations:
        refinement_iteration += 1
        print(f"\n[REFINEMENT ATTEMPT {refinement_iteration}/{max_refinement_iterations}] Code for {todo_file_name} failed tests. Attempting to fix...")

        fix_code_trajectories = []
        fix_code_instruction_msg = get_fix_code_msg(
            current_code_F_n_str=current_F_n_code_being_refined,
            test_code_F_n_str=generated_test_code_F_n, # From Stage 2
            pytest_stdout_str=test_execution_result.get("stdout", ""),
            pytest_stderr_str=test_execution_result.get("stderr", ""),
            paper_content_str=json.dumps(paper_content) if paper_format == "JSON" else paper_content,
            file_logic_analysis_str=json.dumps(detailed_logic_analysis_dict[todo_file_name]),
            filename_F_n_str=todo_file_name
        )
        fix_code_trajectories.extend(fix_code_instruction_msg)

        llm_corrected_output = run_llm(fix_code_trajectories)
        
        corrected_code_F_n = ""
        try:
            corrected_code_F_n = extract_code_from_content(llm_corrected_output)
        except Exception as e:
            corrected_code_F_n = extract_code_from_content2(llm_corrected_output)

        if not corrected_code_F_n: # If extraction fails or LLM returns empty
            print(f"[REFINEMENT ATTEMPT {refinement_iteration}/{max_refinement_iterations}] Failed to extract corrected code. Re-using previous version for re-test.")
            # current_F_n_code_being_refined remains unchanged
        else:
            current_F_n_code_being_refined = corrected_code_F_n
            print(f"[REFINEMENT ATTEMPT {refinement_iteration}/{max_refinement_iterations}] Extracted new version of code. Re-running tests...")

        # Re-run tests with the (potentially) new code
        # sandbox_test_file_original_path was defined in Stage 3
        test_execution_result = run_tests_in_sandbox(
            file_to_test_original_path=todo_file_name,
            code_F_n_content=current_F_n_code_being_refined,
            test_file_original_path=sandbox_test_file_original_path, # Defined in Stage 3
            test_code_F_n_content=generated_test_code_F_n, # From Stage 2
            output_repo_dir=output_repo_dir,
            dependencies_dict=done_file_dict
        )

        if test_execution_result.get("passed", False):
            print(f"[REFINEMENT ATTEMPT {refinement_iteration}/{max_refinement_iterations}] Tests PASSED for {todo_file_name} after correction.")
        else:
            print(f"[REFINEMENT ATTEMPT {refinement_iteration}/{max_refinement_iterations}] Tests FAILED for {todo_file_name} after correction.")
            if refinement_iteration == max_refinement_iterations:
                 print(f"  Max refinement attempts reached. Last test output:")
            print(f"  Return Code: {test_execution_result.get('returncode', 'N/A')}")
            print(f"  Stdout:\n{test_execution_result.get('stdout', '')}")
            if test_execution_result.get('stderr', ''):
                print(f"  Stderr:\n{test_execution_result.get('stderr', '')}")

    # After the loop, current_F_n_code_being_refined holds the final version of the code for F_n
    # Update the main variable that will be saved and added to done_file_dict
    final_code_for_F_n = current_F_n_code_being_refined
    
    # Log final status
    if test_execution_result.get("passed", False):
        print(f"\n[REFINEMENT FINAL STATUS] Tests for {todo_file_name} ultimately PASSED.")
    else:
        print(f"\n[REFINEMENT FINAL STATUS] Tests for {todo_file_name} ultimately FAILED after {refinement_iteration} attempts.")

    # The rest of the script will use final_code_for_F_n for saving
    # The original 'completion' and 'generated_code_F_n' from Stage 1 are less relevant now for the code content itself.
    # However, 'responses' and 'trajectories' for the initial generation might still be useful for logging.
    
    # response and trajectories for the *initial* code generation are kept as they were
    initial_completion_json = { # Renamed to avoid confusion
        'text': completion # This is the raw output from Stage 1 LLM call
    }
    responses.append(initial_completion_json)
    trajectories.append({'role': 'assistant', 'content': completion}) # Trajectory for Stage 1

    done_file_lst.append(todo_file_name)

    # save
    os.makedirs(f'{output_repo_dir}', exist_ok=True)
    save_todo_file_name = todo_file_name.replace("/", "_")

    # print and logging (perhaps log the initial LLM response for F_n if different from final_code_for_F_n)
    # print_response(initial_completion_json, is_llm=True) # This prints the raw initial LLM output

    # Save the *initial* raw LLM output for F_n to its artifact file
    with open(f'{artifact_output_dir}/{save_todo_file_name}_coding.txt', 'w', encoding='utf-8') as f:
        f.write(completion) # Save the raw LLM output from Stage 1

    # The 'code' variable should hold the FINAL version of the code after refinement
    # This 'final_code_for_F_n' is already extracted (it's not raw LLM output).
    code = final_code_for_F_n

    done_file_dict[todo_file_name] = code # Add the refined code to done_file_dict
    
    # Save the main code file (F_n)
    # Path handling for subdirectories in output_repo_dir
    main_code_full_path = os.path.join(output_repo_dir, todo_file_name)
    if save_todo_file_name != todo_file_name: # Original logic for handling potential path differences
        main_code_dir = os.path.dirname(main_code_full_path)
        if main_code_dir: # Ensure directory is not empty (e.g. for root files)
            os.makedirs(main_code_dir, exist_ok=True)
    else: # Ensure base output_repo_dir exists if not already handled by above
        os.makedirs(output_repo_dir, exist_ok=True)
        
    with open(main_code_full_path, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"\n[SAVE] Saved final code for {todo_file_name} to {main_code_full_path}")

    # Save the generated test file (test_F_n.py)
    if generated_test_code_F_n: # Check if test code was actually generated
        # sandbox_test_file_original_path was defined in Stage 3 and holds the relative path for tests
        # e.g., "tests/module/test_code.py" or "tests/test_main.py"
        if 'sandbox_test_file_original_path' in locals() and sandbox_test_file_original_path:
            full_test_file_path = os.path.join(output_repo_dir, sandbox_test_file_original_path)
            test_file_dir = os.path.dirname(full_test_file_path)
            os.makedirs(test_file_dir, exist_ok=True)
            with open(full_test_file_path, 'w', encoding='utf-8') as f:
                f.write(generated_test_code_F_n)
            print(f"[SAVE] Saved generated tests for {todo_file_name} to {full_test_file_path}")
        else:
            print(f"[SAVE] Warning: Test code for {todo_file_name} was generated, but its save path (sandbox_test_file_original_path) was not found. Skipping test file save.")
    else:
        print(f"[SAVE] No test code was generated for {todo_file_name}. Skipping test file save.")

    # The old saving logic below this is now partially duplicated or handled above.
    # I'll remove the redundant parts.
    # if save_todo_file_name != todo_file_name: # This check is now part of the main_code_full_path logic
    #     todo_file_dir = '/'.join(todo_file_name.split("/")[:-1])
    #     os.makedirs(f"{output_repo_dir}/{todo_file_dir}", exist_ok=True)

    # with open(f"{output_repo_dir}/{todo_file_name}", 'w', encoding='utf-8') as f: # This is handled by main_code_full_path logic
    #     f.write(code)
    if save_todo_file_name != todo_file_name: # This specific part of old logic regarding subdirectories for F_n is handled by main_code_full_path logic
        todo_file_dir = '/'.join(todo_file_name.split("/")[:-1])
        os.makedirs(f"{output_repo_dir}/{todo_file_dir}", exist_ok=True)

    with open(f"{output_repo_dir}/{todo_file_name}", 'w', encoding='utf-8') as f:
        f.write(code)
