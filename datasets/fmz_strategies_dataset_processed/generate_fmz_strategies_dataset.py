import os
import re
import glob
import json
import pandas as pd

def is_python_strategy(content):
    """检查内容是否为Python策略（基于Source (python)标记）"""
    return bool(re.search(r'Source\s*\(python\)', content, re.IGNORECASE))

def extract_python_code(content):
    """从MD内容中提取Python代码块"""
    code_blocks = []
    
    # 匹配三种常见的Python代码块格式
    patterns = [
        r'```python\s*([\s\S]*?)```',
        r'```python3\s*([\s\S]*?)```',
        r'```\s*([\s\S]*?)```'
    ]
    
    for pattern in patterns:
        blocks = re.findall(pattern, content)
        
        for block in blocks:
            # 检查代码块是否包含Python特征代码
            if pattern == patterns[2]:
                python_keywords = [
                    'def ', 'class ', 'import ', 'if __name__ ==', 'return ',
                    'for ', 'while ', 'with ', '@', 'lambda '
                ]
                if not any(keyword in block for keyword in python_keywords):
                    continue
            
            cleaned_block = block.strip()
            if cleaned_block:
                code_blocks.append(cleaned_block)
    
    # 移除重复的代码块
    unique_blocks = []
    for block in code_blocks:
        if block not in unique_blocks:
            unique_blocks.append(block)
    
    return unique_blocks

def extract_strategy_info(content):
    """从MD内容中提取策略信息"""
    info = {
        "title": "未找到标题",
        "description": "未找到描述"
    }
    
    # 提取标题（通常是第一个#开头的行）
    title_match = re.search(r'#\s+(.+?)\n', content)
    if title_match:
        info["title"] = title_match.group(1).strip()
    
    # 提取描述（标题后的第一段非空行）
    desc_match = re.search(r'#\s+.+?\n\s*([\s\S]*?)(?:\n#|$)', content)
    if desc_match:
        info["description"] = desc_match.group(1).strip()
    
    return info

def main():
    # 输入目录 - MD文件所在位置
    input_directory = "/root/autodl-tmp/FinancialStrategy2Code/datasets/fmz_strategies"
    
    # 输出目录 - 处理后的数据集存放位置
    output_directory = "/root/autodl-tmp/FinancialStrategy2Code/datasets/fmz_strategies/fmz_strategies_dataset_processed"
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_directory, exist_ok=True)
    
    # 获取所有MD文件
    md_files = glob.glob(os.path.join(input_directory, "*.md"))
    
    # 用于存储所有策略的数据集
    dataset = []
    
    # 统计信息
    total_files = len(md_files)
    python_strategy_count = 0
    processed_count = 0
    
    print(f"开始处理 {total_files} 个MD文件...")
    
    # 策略ID从1开始计数
    strategy_counter = 1
    
    # 检查每个文件
    for file_path in md_files:
        file_name = os.path.basename(file_path)
        file_base_name = os.path.splitext(file_name)[0]  # 获取不带扩展名的文件名
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 只处理有Source (python)标记的文件
                if not is_python_strategy(content):
                    print(f"跳过非Python策略: {file_name}")
                    continue
                
                python_strategy_count += 1
                print(f"\n处理Python策略 #{python_strategy_count}: {file_name}")
                
                # 提取策略信息
                strategy_info = extract_strategy_info(content)
                
                # 提取Python代码块
                code_blocks = extract_python_code(content)
                
                if not code_blocks:
                    print(f"警告: 在{file_name}中未找到Python代码块，跳过")
                    continue
                
                # 构建strategy_id，格式为"序号_文件名"
                strategy_id = f"{strategy_counter}_{file_base_name}"
                
                # 构建strategy_description，以MD文件名为策略名称开头
                description = f"策略名称: {file_base_name}\n\n{strategy_info['description']}"
                
                # 合并所有代码块
                combined_code = "\n\n# ==========================================\n\n".join(code_blocks)
                
                # 添加到数据集（字段名已修改）
                dataset.append({
                    "strategy_id": strategy_id,
                    "strategy_code": combined_code,
                    "strategy_description": description
                })
                
                strategy_counter += 1
                processed_count += 1
                
                print(f"  提取了 {len(code_blocks)} 个代码块")
                print(f"  策略ID: {strategy_id}")
                
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
    
    # 保存完整数据集
    dataset_path = os.path.join(output_directory, "python_strategy_dataset.json")
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    # 保存统计信息
    stats = {
        "total_files": total_files,
        "python_strategies": python_strategy_count,
        "processed_strategies": processed_count,
        "generated_at": str(pd.Timestamp.now())
    }
    
    stats_path = os.path.join(output_directory, "dataset_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\n处理完成!")
    print(f"总MD文件数: {total_files}")
    print(f"找到Python策略: {python_strategy_count}")
    print(f"成功处理并生成数据集: {processed_count}")
    print(f"\n数据集已保存到: {output_directory}")
    print(f"完整数据集: {dataset_path}")
    print(f"统计信息: {stats_path}")

if __name__ == "__main__":
    main()