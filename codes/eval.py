from openai import OpenAI
import json
import os
import sys
import argparse
from utils import read_python_files, extract_planning, content_to_json, \
        num_tokens_from_messages, read_all_files, extract_json_from_string, get_now_str, print_log_cost
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key = os.environ["OPENAI_API_KEY"],
    base_url = os.environ["OPENAI_BASE_URL"]
)

def api_call(request_json):
    completion = client.chat.completions.create(**request_json)
    return completion

def main(args):

    paper_name = args.paper_name
    pdf_json_path = args.pdf_json_path
    output_dir = args.output_dir  
    target_repo_dir = args.target_repo_dir
    eval_result_dir = args.eval_result_dir
    gpt_version = args.gpt_version
    generated_n = args.generated_n
    data_dir = args.data_dir
    eval_type = args.eval_type
    is_papercoder = True if args.papercoder else False

    gold_repo_dir = args.gold_repo_dir

    # paper
    with open(f'{pdf_json_path}') as f:
        paper_json = json.load(f)
    
    codes = ""
    if is_papercoder:
        # python files
        target_files_dict = read_python_files(target_repo_dir)

        # configuration
        with open(f'{output_dir}/planning_config.yaml') as f: 
            config_yaml = f.read()
        
        context_lst = extract_planning(f'{output_dir}/planning_trajectories.json')

        if os.path.exists(f'{output_dir}/task_list.json'):
            with open(f'{output_dir}/task_list.json') as f:
                task_list = json.load(f)
        else:
            task_list = content_to_json(context_lst[2])

        todo_file_lst = task_list['Task list']        
        for todo_file in todo_file_lst:
            if todo_file.endswith(".yaml"):
                continue
            codes += f"```python\n## File name: {todo_file}\n{target_files_dict[todo_file]}\n```\n\n"

        codes += f"```yaml\n## File name: config.yaml\n{config_yaml}\n```\n\n"
    else:
        target_files_dict = read_all_files(target_repo_dir, allowed_ext=[".py", ".yaml", ".yml", ".md", ".sh", ".bash"], is_print=False)
        for file_name, code in target_files_dict.items():
            codes += f"```## File name: {file_name}\n{code}\n```\n\n" 


    prompt = open(f"{data_dir}/prompts/{eval_type}.txt").read()
    
    cur_prompt = prompt.replace('{{Paper}}', f"{paper_json}").replace('{{Code}}', codes)
    
    # refernce-based
    if "ref_based" == eval_type and len(gold_repo_dir) > 0:
        all_files_dict = read_all_files(gold_repo_dir, allowed_ext=[".py", ".yaml", ".yml", ".md", ".sh", ".bash"], is_print=False)

        goldcodes = ""
        gold_cnt = 0
        if len(args.selected_file_path) > 0:
            selected_file_lst = []
            with open(args.selected_file_path) as f:
                selected_file_lst = f.readlines()
            
            for s_idx in range(len(selected_file_lst)):
                selected_file_lst[s_idx] = selected_file_lst[s_idx].strip() 

            
            for all_file, all_file_code in all_files_dict.items():
                if all_file not in selected_file_lst:
                    continue

                goldcodes += f"```## File name: {all_file}\n{all_file_code}\n```\n\n" 

                gold_cnt += 1


        else:
            for all_file, all_file_code in all_files_dict.items():
                goldcodes += f"```## File name: {all_file}\n{all_file_code}\n```\n\n" 

                gold_cnt += 1

        cur_prompt = cur_prompt.replace('{{GoldCode}}', f"{goldcodes}")

    msg = [{"role": "system", "content": cur_prompt}]

    try:
        num_tokens = num_tokens_from_messages(msg)
    except Exception as e:
        print(f"[WARNING] An exception was raised while counting tokens for the target repository of {args.paper_name}.")
        print(e)
        print("-"*40)
        num_tokens = 0
    

    if num_tokens > 128000:
        print(f"[ERROR] {args.paper_name} more than 128k")
        sys.exit(0)
    

    if "o3-mini" in gpt_version:
        if generated_n > 8:
            print(f"[WARNING] o3-mini does not support n > 8. Setting generated_n to 8.")
            generated_n = 8

        request_json = {
                "model": gpt_version, 
                "messages": msg,
                "reasoning_effort": "high",
                "n": generated_n
        }
    else:
        request_json = {
                "model": gpt_version, 
                "messages": msg, 
                "temperature": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": None,
                "n": generated_n # 10
        }
        
    completion = api_call(request_json)
    completion_json = json.loads(completion.model_dump_json())
        
    score_key = "score"
    rationale_key = "critique_list"


    all_scores = []
    rationales = []
    for n in range(generated_n):    
        choice = completion_json['choices'][n]

        output = choice['message']['content'].strip()
        
        try:
            output_json2 = json.loads(output)
            score = int(output_json2[score_key])

            if isinstance(output_json2[rationale_key], str):
                rationale = output_json2[rationale_key]
            else:
                rationale = json.dumps(output_json2[rationale_key])
        except Exception as e:
            # print(e)             
            try:
                output_json2 = json.loads(extract_json_from_string(output))
                score = int(output_json2[score_key])

                if isinstance(output_json2[rationale_key], str):
                    rationale = output_json2[rationale_key]
                else:
                    rationale = json.dumps(output_json2[rationale_key])
            except Exception as e2: # Parsing Error
                print(f"[WARNING] Invalid repsponse: parsing error")
                print(e2)
                print("-"*40)
              
                continue
            
        # score
        if score < 1 or score > 5:
            print(f"[WARNING] Invalid repsponse: score {score}, Score must be in the range of 1–5.")
            continue
        
        all_scores.append(int(score))
        rationales.append(rationale)
        

    avg_score = sum(all_scores) / len(all_scores)

    output_json= {
        "paper_name": paper_name,
        "target_repo_dir": target_repo_dir,
        "eval_type": eval_type,
        "gold_repo_dir": gold_repo_dir,
        "generated_n": generated_n,
        "request_json": request_json,
        "completion_json": completion_json,
        "eval_result": {
            "score": avg_score,
            "valid_n": len(all_scores),
            "scroe_lst": all_scores,
            "rationale_lst": rationales,    
        },
    }
    
    now_str = get_now_str()
    os.makedirs(eval_result_dir, exist_ok=True)
    with open(f"{eval_result_dir}/{paper_name}_eval_{eval_type}_{gpt_version}_{now_str}.json", 'w', encoding='utf-8') as f:
        json.dump(output_json, f)

    
    # ---------------
    print()
    print("=" * 40)
    print("🌟 Evaluation Summary 🌟")
    print(f"📄 Paper name: {paper_name}")
    print(f"🧪 Evaluation type: {eval_type}")
    print(f"📁 Target repo directory: {target_repo_dir}")
    print(f"📊 Evaluation result:")
    print(f"\t📈 Score: {avg_score:.4f}")
    print(f"\t✅ Valid: {output_json['eval_result']['valid_n']}/{generated_n}")
    print("=" * 40)
    
    print_log_cost(completion_json, gpt_version, f"[Evaluation] {paper_name} - {eval_type}", output_dir, 0)
    # ---------------


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('--paper_name', type=str)
    argparser.add_argument('--pdf_json_path', type=str)
    argparser.add_argument('--data_dir',type=str, default="../data")

    argparser.add_argument('--output_dir',type=str)
    
    argparser.add_argument('--target_repo_dir', type=str)
    argparser.add_argument('--gold_repo_dir', type=str, default="")
    argparser.add_argument('--eval_result_dir',type=str)
    
    argparser.add_argument('--eval_type', type=str, default="ref_free", choices=["ref_free", "ref_based"])

    argparser.add_argument('--generated_n', type=int, default=8)
    argparser.add_argument('--gpt_version', type=str, default="o3-mini")

    argparser.add_argument('--selected_file_path', type=str, default="") 
    argparser.add_argument('--papercoder', action="store_true")
    
    
    
    args = argparser.parse_args()
    main(args)


# ref-free
# python eval.py \
#     --paper_name Transformer \
#     --pdf_json_path ../examples/Transformer_cleaned.json \
#     --data_dir ../data \
#     --output_dir ../outputs/Transformer \
#     --target_repo_dir ../outputs/Transformer_repo \
#     --eval_result_dir ../results \
#     --eval_type ref_free \
#     --generated_n 8 \
#     --papercoder

# ref-based
# python eval.py \
#     --paper_name Transformer \
#     --pdf_json_path ../examples/Transformer_cleaned.json \
#     --data_dir ../data \
#     --output_dir ../outputs/Transformer \
#     --target_repo_dir ../outputs/Transformer_repo \
#     --gold_repo_dir ../examples/Transformer_gold_repo \
#     --eval_result_dir ../results \
#     --eval_type ref_based \
#     --generated_n 8 \
#     --papercoder
