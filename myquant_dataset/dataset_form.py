import os
import json

# 定义基础路径，根据实际情况调整
base_path = "/root/autodl-tmp/FinancialStrategy2Code/datasets/"
# 拼接得到 code 目录和 description 目录的路径
code_dir = os.path.join(base_path, "myquant_dataset/code")
desc_dir = os.path.join(base_path, "myquant_dataset/description")

# 初始化数据集列表
dataset = []

# 先检查 code_dir 是否存在
if not os.path.exists(code_dir):
    print(f"错误：目录 {code_dir} 不存在，请检查路径设置或确认抓取流程是否正确生成了该目录")
else:
    # 获取 code 目录下所有 .py 文件
    code_files = [f for f in os.listdir(code_dir) if f.endswith(".py")]

    for code_file in code_files:
        # 提取策略 ID（如 myquant_001）
        strategy_id = code_file.rsplit(".", 1)[0]  
        # 构建对应 description 文件路径
        desc_file = os.path.join(desc_dir, f"{strategy_id}.md")

        # 读取代码内容
        with open(os.path.join(code_dir, code_file), "r", encoding="utf-8") as f_code:
            code_content = f_code.read()

        # 读取描述内容（若文件存在，不存在则设为空）
        desc_content = ""
        if os.path.exists(desc_file):
            with open(desc_file, "r", encoding="utf-8") as f_desc:
                desc_content = f_desc.read()

        # 构建策略数据字典（字段名已修改）
        strategy_data = {
            "strategy_id": strategy_id,
            "strategy_code": code_content,  # 修改字段名
            "strategy_description": desc_content  # 修改字段名
        }
        dataset.append(strategy_data)

    # 保存为 JSON 数据集
    output_path = os.path.join(base_path, "myquant_dataset/strategy_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(dataset, f_out, ensure_ascii=False, indent=2)

    print(f"数据集已整理完成，保存至 {output_path}")
    print(f"共整理 {len(dataset)} 条策略数据")