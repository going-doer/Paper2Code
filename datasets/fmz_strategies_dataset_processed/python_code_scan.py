import os
import re
import glob

def main():
    # 策略文件目录
    directory = "/root/autodl-tmp/FinancialStrategy2Code/datasets/fmz_strategies"
    
    # 获取所有MD文件
    md_files = glob.glob(os.path.join(directory, "*.md"))
    
    python_strategies = []
    
    # 检查每个文件
    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 只检查是否包含"Source (python)"标记
                if re.search(r'Source\s*\(python\)', content, re.IGNORECASE):
                    file_name = os.path.basename(file_path)
                    python_strategies.append(file_name)
                    print(f"找到Python策略: {file_name}")
                else:
                    print(f"未找到Python标记: {file_name}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # 输出结果
    print(f"\n找到 {len(python_strategies)} 个明确标记为Python的策略:")
    for i, file_name in enumerate(python_strategies, 1):
        print(f"{i}. {file_name}")
    
    # 保存结果到文件
    with open("python_strategies.txt", 'w', encoding='utf-8') as f:
        f.write(f"找到 {len(python_strategies)} 个明确标记为Python的策略:\n")
        for file_name in python_strategies:
            f.write(f"{file_name}\n")
    
    print(f"\n结果已保存到 python_strategies.txt")

if __name__ == "__main__":
    main()