import os
import json
from collections import defaultdict

# 定义两个数据集路径
dataset1_path = "/root/autodl-tmp/FinancialStrategy2Code/datasets/fmz_strategies/fmz_strategies_dataset_processed/python_strategy_dataset.json"
dataset2_path = "/root/autodl-tmp/FinancialStrategy2Code/datasets/myquant_dataset/strategy_dataset.json"

# 定义合并后的输出路径
output_path = "/root/autodl-tmp/FinancialStrategy2Code/datasets/merged_strategy_dataset.json"

# 初始化合并后的数据集
merged_dataset = []

try:
    # 读取第一个数据集
    if os.path.exists(dataset1_path):
        with open(dataset1_path, "r", encoding="utf-8") as f1:
            dataset1 = json.load(f1)
            print(f"读取数据集1: {len(dataset1)} 条策略")
            merged_dataset.extend(dataset1)
    else:
        print(f"警告: 数据集1不存在 - {dataset1_path}")

    # 读取第二个数据集
    if os.path.exists(dataset2_path):
        with open(dataset2_path, "r", encoding="utf-8") as f2:
            dataset2 = json.load(f2)
            print(f"读取数据集2: {len(dataset2)} 条策略")
            merged_dataset.extend(dataset2)
    else:
        print(f"警告: 数据集2不存在 - {dataset2_path}")

    # 检查是否有策略数据
    if not merged_dataset:
        print("错误: 合并后的数据集为空，请检查输入文件")
        exit(1)

    # 重新计数策略ID，从1开始
    for i, strategy in enumerate(merged_dataset, 1):
        strategy["strategy_id"] = f"strategy_{i}"  # 格式为 strategy_1, strategy_2...
        # 或者仅用数字: strategy_id = i

    # 保存合并后的数据集
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(merged_dataset, f_out, ensure_ascii=False, indent=2)
    
    print(f"\n合并完成! 共 {len(merged_dataset)} 条策略")
    print(f"合并后的数据集已保存至: {output_path}")
    print("策略ID已重新从1开始计数，格式为 strategy_1, strategy_2...")

except json.JSONDecodeError as e:
    print(f"错误: 解析JSON时出错 - {e}")
    print("请检查数据集文件格式是否正确")
except Exception as e:
    print(f"处理过程中出错: {e}")