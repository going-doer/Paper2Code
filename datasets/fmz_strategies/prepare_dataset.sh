#!/bin/bash

# 设置工作目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 创建输出目录
OUTPUT_DIR="processed_data"
mkdir -p $OUTPUT_DIR

# 运行数据处理脚本
echo "开始处理FMZ策略数据集..."
python process_fmz_dataset.py \
    --input-dir "." \
    --output-dir "$OUTPUT_DIR" \
    --output-file "fmz_strategies.jsonl"

# 分析数据集统计信息
echo "生成数据集统计信息..."
python - << EOF
import json
from collections import Counter
from pathlib import Path

def analyze_dataset(file_path):
    stats = {
        'total_samples': 0,
        'strategy_types': Counter(),
        'indicators': Counter(),
        'features': Counter(),
        'complexity_levels': Counter(),
        'avg_code_length': 0,
        'timeframes': Counter()
    }
    
    total_code_length = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            stats['total_samples'] += 1
            
            # 统计策略类型
            for stype in sample['metadata'].get('strategy_patterns', []):
                stats['strategy_types'][stype] += 1
            
            # 统计技术指标
            for indicator in sample['metadata'].get('indicators', []):
                stats['indicators'][indicator] += 1
            
            # 统计特征
            for feature in sample['metadata'].get('strategy_features', []):
                stats['features'][feature] += 1
            
            # 统计复杂度级别
            complexity = sample['metadata']['training_labels'].get('complexity_level', 'unknown')
            stats['complexity_levels'][complexity] += 1
            
            # 统计代码长度
            total_code_length += sample['code_length']
            
            # 统计时间周期
            timeframe = sample['metadata']['backtest_config'].get('timeframe', 'unknown')
            stats['timeframes'][timeframe] += 1
    
    stats['avg_code_length'] = total_code_length / stats['total_samples'] if stats['total_samples'] > 0 else 0
    return stats

def print_stats(stats):
    print("\n=== FMZ策略数据集统计信息 ===")
    print(f"\n总样本数: {stats['total_samples']}")
    print(f"平均代码长度: {stats['avg_code_length']:.1f} 行")
    
    print("\n策略类型分布:")
    for stype, count in stats['strategy_types'].most_common():
        print(f"  - {stype}: {count} ({count/stats['total_samples']*100:.1f}%)")
    
    print("\n常用技术指标:")
    for indicator, count in stats['indicators'].most_common(10):
        print(f"  - {indicator}: {count}")
    
    print("\n策略特征分布:")
    for feature, count in stats['features'].most_common():
        print(f"  - {feature}: {count} ({count/stats['total_samples']*100:.1f}%)")
    
    print("\n复杂度分布:")
    for level, count in stats['complexity_levels'].most_common():
        print(f"  - {level}: {count} ({count/stats['total_samples']*100:.1f}%)")
    
    print("\n常用时间周期:")
    for timeframe, count in stats['timeframes'].most_common():
        print(f"  - {timeframe}: {count}")

file_path = Path("$OUTPUT_DIR/fmz_strategies.jsonl")
if file_path.exists():
    stats = analyze_dataset(file_path)
    print_stats(stats)
else:
    print("数据集文件不存在！")
EOF

echo "处理完成！"
