#!/usr/bin/env python3
import os
import json
import re
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from tqdm import tqdm
import sys

# 配置日志级别
import logging

def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'process_fmz_dataset.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='处理FMZ量化策略数据集')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='输入数据集目录')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--output-file', type=str, default='financial_strategies.jsonl',
                        help='输出文件名 (默认: financial_strategies.jsonl)')
    return parser.parse_args()

class ProcessingStats:
    """处理统计信息类"""
    def __init__(self):
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = 0
        self.invalid_encodings = 0
        self.empty_strategies = 0
        self.valid_strategies = 0
        self.errors = []
        self._logger = None
        
    def set_logger(self, logger):
        self._logger = logger
        
    def add_error(self, file_path: str, error_msg: str):
        self.errors.append({"file": file_path, "error": error_msg})
        if self._logger:
            self._logger.error(f"处理文件 {file_path} 失败: {error_msg}")
        
    def get_summary(self) -> Dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "invalid_encodings": self.invalid_encodings,
            "empty_strategies": self.empty_strategies,
            "valid_strategies": self.valid_strategies,
            "success_rate": round(self.valid_strategies / max(self.total_files, 1) * 100, 2),
            "errors": self.errors
        }

def analyze_strategy_pattern(code_content):
    """分析策略模式和特征"""
    patterns = {
        "trend_following": [
            "趋势跟踪", "突破", "动量", "趋势", "方向", "走势",
            "MA", "EMA", "MACD", "moving average", "trend following",
            "breakout", "momentum"
        ],
        "mean_reversion": [
            "均值回归", "震荡", "超买", "超卖", "回归", "反转",
            "RSI", "BOLL", "KDJ", "oscillator", "overbought", "oversold",
            "mean reversion", "reversion"
        ],
        "arbitrage": [
            "套利", "对冲", "价差", "配对", "统计套利",
            "arbitrage", "hedge", "spread", "pair trading",
            "statistical arbitrage"
        ],
        "market_making": [
            "做市", "报价", "买卖价差", "流动性",
            "market making", "quoting", "bid-ask spread",
            "liquidity provision", "market depth"
        ],
        "momentum": [
            "动量", "惯性", "加速度", "趋势强度",
            "momentum", "acceleration", "ROC", "rate of change",
            "velocity", "strength"
        ],
        "ml_based": [
            "机器学习", "神经网络", "深度学习", "AI", "人工智能",
            "machine learning", "neural network", "deep learning",
            "artificial intelligence", "prediction model"
        ],
        "grid_trading": [
            "网格", "等距", "等金额", "累进",
            "grid trading", "fixed interval", "fixed amount",
            "progressive", "grid strategy"
        ]
    }
    
    features = {
        "risk_management": [
            "止损", "止盈", "风险", "仓位", "敞口",
            "stop loss", "take profit", "risk control",
            "position limit", "exposure"
        ],
        "position_sizing": [
            "仓位", "头寸", "下单量", "开仓", "平仓",
            "position size", "lot size", "order size",
            "entry", "exit", "close position"
        ],
        "multi_timeframe": [
            "多周期", "多时间周期", "跨周期", "周期切换",
            "multiple timeframe", "multi-timeframe",
            "timeframe analysis", "period switch"
        ],
        "market_regime": [
            "趋势", "震荡", "行情", "市场状态", "市场环境",
            "market regime", "market condition", "market state",
            "trend phase", "ranging phase"
        ],
        "volatility_based": [
            "波动率", "ATR", "标准差", "通道", "布林带",
            "volatility", "standard deviation", "channel",
            "bollinger bands", "volatility breakout"
        ],
        "indicator_combination": [
            "指标交叉", "信号确认", "多指标", "协同",
            "indicator cross", "signal confirmation",
            "multiple indicators", "composite"
        ]
    }
    
    strategy_types = []
    strategy_features = []
    
    # 分析策略类型
    for pattern_type, keywords in patterns.items():
        if any(keyword in code_content for keyword in keywords):
            strategy_types.append(pattern_type)
    
    # 分析策略特征
    for feature_type, keywords in features.items():
        if any(keyword in code_content for keyword in keywords):
            strategy_features.append(feature_type)
    
    return strategy_types, strategy_features

def extract_metadata(file_path, logger=None):
    """提取策略文件中的元数据"""
    file_path = Path(file_path)
    if logger:
        logger.info(f"处理文件: {file_path}")
    
    metadata = {
        "file_path": str(file_path),
        "name": file_path.stem,
        "extension": file_path.suffix,
        "size_bytes": file_path.stat().st_size,
        "modified_time": file_path.stat().st_mtime,
        "description": "",
        "symbol": "unknown",
        "author": "unknown",
        "strategy_type": "unknown",
        "version": "1.0",
        "created_time": "",
        "last_updated": "",
        "tags": [],
        "category": "",
        "platform": "FMZ",
        "language": file_path.suffix.lstrip('.'),
        "dependencies": [],
        "backtest_config": {},
        "training_labels": {},
        "complexity_metrics": {},
        "performance_metrics": {
            "win_rate": None,
            "profit_factor": None,
            "max_drawdown": None,
            "sharpe_ratio": None
        }
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # 提取策略名称（支持多种格式）
            name_patterns = [
                r'策略名称[：:]\s*([^\n]+)',
                r'#\s*([^\n]+)',  # 第一行标题
                r'策略名[：:]\s*([^\n]+)',
                r'名称[：:]\s*([^\n]+)'
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, content)
                if match:
                    metadata["name"] = match.group(1).strip()
                    break
            
            # 提取策略描述
            desc_patterns = [
                r'策略说明[：:]\s*([^\n]+)',
                r'策略描述[：:]\s*([^\n]+)',
                r'##\s*([^\n]+)'  # 第二行标题
            ]
            
            for pattern in desc_patterns:
                match = re.search(pattern, content)
                if match:
                    metadata["description"] = match.group(1).strip()
                    break
            
            # 提取交易品种
            symbol_patterns = [
                r'交易品种[：:]\s*([^\n]+)',
                r'交易对[：:]\s*([^\n]+)',
                r'品种[：:]\s*([^\n]+)'
            ]
            
            for pattern in symbol_patterns:
                match = re.search(pattern, content)
                if match:
                    metadata["symbol"] = match.group(1).strip()
                    break
            
            # 提取作者信息
            author_patterns = [
                r'作者[：:]\s*([^\n]+)',
                r'Author[：:]\s*([^\n]+)'
            ]
            
            for pattern in author_patterns:
                match = re.search(pattern, content)
                if match:
                    metadata["author"] = match.group(1).strip()
                    break
            
            # 提取策略类型（趋势、震荡、套利等）
            strategy_types = ["趋势", "震荡", "套利", "网格", "做市", "量化", "对冲", "高频"]
            for stype in strategy_types:
                if stype in content:
                    metadata["strategy_type"] = stype
                    break
            
            # 提取技术指标
            indicators = []
            indicator_keywords = ["MACD", "RSI", "KDJ", "BOLL", "MA", "EMA", "ATR", "CCI", "OBV"]
            for indicator in indicator_keywords:
                if indicator.upper() in content.upper():
                    indicators.append(indicator)
            
            metadata["indicators"] = indicators
            
            # 提取交易频率
            freq_keywords = {
                "高频": ["高频", "tick", "秒级", "秒", "高频交易"],
                "日内": ["日内", "分钟", "分钟线", "1m", "5m", "15m", "30m"],
                "日线": ["日线", "日级", "日", "daily", "day"],
                "周线": ["周线", "周级", "周", "weekly", "week"],
                "长线": ["长线", "月线", "月级", "月", "长期", "year", "month"]
            }
            
            for freq, keywords in freq_keywords.items():
                if any(keyword in content for keyword in keywords):
                    metadata["frequency"] = freq
                    break
            
    except Exception as e:
        if logger:
            logger.error(f"提取元数据时出错: {e}")
    
    return metadata

def calculate_complexity_metrics(code_content):
    """计算代码复杂度指标"""
    metrics = {
        "loc": len(code_content.splitlines()),  # 代码行数
        "function_count": len(re.findall(r'def\s+\w+\s*\(', code_content)),  # 函数数量
        "class_count": len(re.findall(r'class\s+\w+', code_content)),  # 类数量
        "comment_ratio": len(re.findall(r'#.*$|\'{3}[\s\S]*?\'{3}|"{3}[\s\S]*?"{3}', code_content, re.M)) / (len(code_content.splitlines()) + 1),  # 注释比例
        "cyclomatic_complexity": len(re.findall(r'\bif\b|\bfor\b|\bwhile\b|\band\b|\bor\b', code_content)),  # 圈复杂度近似值
    }
    return metrics

def extract_backtest_config(code_content):
    """提取回测配置参数"""
    config = {}
    
    # 提取常见的回测参数
    param_patterns = {
        "initial_capital": r'(初始资金|capital|initCapital)\s*[=:]\s*(\d+)',
        "timeframe": r'(period|时间周期|周期)\s*[=:]\s*[\'"]*([^\'"]+)',
        "fees": r'(fee|费率|手续费)\s*[=:]\s*([0-9.]+)',
        "slippage": r'(slippage|滑点)\s*[=:]\s*([0-9.]+)',
    }
    
    for param, pattern in param_patterns.items():
        match = re.search(pattern, code_content)
        if match:
            config[param] = match.group(2)
    
    return config

def extract_training_labels(code_content):
    """提取用于训练的标签"""
    labels = {
        "has_risk_management": bool(re.search(r'止损|止盈|risk|stop[_\s]loss', code_content, re.I)),
        "has_position_sizing": bool(re.search(r'仓位|position|size|volume', code_content, re.I)),
        "has_market_analysis": bool(re.search(r'market|analysis|分析|行情', code_content, re.I)),
        "is_multiasset": bool(re.search(r'symbols|pairs|多品种|portfolio', code_content, re.I)),
        "complexity_level": "high" if len(code_content.splitlines()) > 300 else "medium" if len(code_content.splitlines()) > 100 else "low"
    }
    return labels

def extract_code(file_path, logger=None):
    """提取策略文件中的代码部分"""
    file_path = Path(file_path)
    code = ""
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 处理文件名中包含中文的情况
        if re.search(r'[\u4e00-\u9fff]', file_path.stem) and logger:
            logger.debug(f"检测到中文文件名: {file_path.stem}")
        
        # 根据文件类型提取代码
        if file_path.suffix == '.py':
            # Python文件直接返回全文
            code = content
            
        elif file_path.suffix == '.md':
            # Markdown文件提取代码块
            code_blocks = extract_markdown_code(content)
            code = "\n\n".join(code_blocks)
            
        elif file_path.suffix == '.js':
            # JavaScript文件直接返回全文
            code = content
            
        elif file_path.suffix == '.lua':
            # Lua文件直接返回全文
            code = content
            
        else:
            # 其他类型文件尝试智能提取
            code_lines = []
            in_code = False
            
            for line in content.split('\n'):
                # 简单的代码检测规则
                if line.strip().startswith(('def ', 'class ', 'function ', 'import ', 'from ', 'var ', 'let ', 'const ', 'local ')):
                    in_code = True
                
                if in_code:
                    code_lines.append(line)
                    
                    # 如果连续5行空行，可能代码结束
                    if len(code_lines) > 5 and all(l.strip() == '' for l in code_lines[-5:]):
                        break
            
            code = "\n".join(code_lines)
    
    except Exception as e:
        logger.error(f"提取代码时出错: {e}")
    
    return code.strip()

def extract_markdown_code(content: str) -> list:
    """从 Markdown 文件中提取代码块
    
    处理多种可能的代码块格式:
    1. 标准代码块 ```language
    2. 缩进代码块
    3. 自定义格式代码块
    """
    code_blocks = []
    
    # 1. 处理标准代码块
    standard_patterns = [
        # 支持更多语言标记
        r'```(?:python|javascript|js|lua|pine|cpp|c\+\+|typescript|ts)?\n(.*?)```',
        # 处理可能的空格和制表符
        r'```\s*(?:python|javascript|js|lua|pine|cpp|c\+\+|typescript|ts)?\s*\n(.*?)```',
        # 处理可能的中文注释
        r'```.*?(?:代码|策略|程序|实现).*?\n(.*?)```'
    ]
    
    for pattern in standard_patterns:
        blocks = re.findall(pattern, content, re.DOTALL)
        code_blocks.extend(blocks)
    
    # 2. 处理缩进代码块
    indented_blocks = re.findall(r'(?:^|\n)(?:    |\t)([^\n]*(?:\n(?:    |\t)[^\n]*)*)', content)
    code_blocks.extend(indented_blocks)
    
    # 3. 处理策略描述后的代码
    description_patterns = [
        r'(?:策略说明|策略描述|算法说明)[:：]\s*\n+([^#\n].*?)(?=\n#|\Z)',
        r'(?:代码实现|具体实现|策略实现)[:：]\s*\n+([^#\n].*?)(?=\n#|\Z)',
        r'(?:完整代码|源代码|策略代码)[:：]\s*\n+([^#\n].*?)(?=\n#|\Z)'
    ]
    
    for pattern in description_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            # 检查是否包含代码特征
            if re.search(r'(function|def|class|var|let|const|if|for|while|exchange\.)', match):
                code_blocks.append(match.strip())
    
    # 4. 清理和验证代码块
    cleaned_blocks = []
    for block in code_blocks:
        # 移除开头和结尾的空行
        block = re.sub(r'^\s*\n|\n\s*$', '', block)
        # 统一缩进样式
        block = block.replace('\t', '    ')
        # 验证最小行数和基本结构
        if len(block.splitlines()) >= 5 and re.search(r'(function|def|class|var|let|const|exchange\.)', block):
            cleaned_blocks.append(block)
    
    return cleaned_blocks

def preprocess_file_content(file_path, logger=None):
    """预处理文件内容，处理编码和格式问题"""
    try:
        # 尝试不同的编码方式
        encodings = ['utf-8', 'gb2312', 'gbk', 'gb18030', 'iso-8859-1']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            if logger:
                logger.warning(f"无法以任何已知编码读取文件: {file_path}")
            return None
            
        # 统一换行符
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # 移除 BOM
        if content.startswith('\ufeff'):
            content = content[1:]
            
        # 移除无效字符
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        
        return content
        
    except Exception as e:
        if logger:
            logger.error(f"预处理文件时出错 {file_path}: {e}")
        return None

def log_error_details(file_path, error_type, error_msg):
    """记录详细的错误信息"""
    error_log = {
        "file": str(file_path),
        "type": error_type,
        "message": str(error_msg),
        "timestamp": str(Path(file_path).stat().st_mtime)
    }
    
    error_log_file = Path("error_log.jsonl")
    with open(error_log_file, "a", encoding='utf-8') as f:
        f.write(json.dumps(error_log, ensure_ascii=False) + "\n")

def validate_fmz_strategy(code_content: str) -> tuple:
    """验证是否是有效的 FMZ 策略代码"""
    required_components = {
        "entry_point": False,     # 入口点（可以是main函数或其他关键方法）
        "exchange_api": False,    # 交易所 API 调用或交易相关代码
        "trade_logic": False,     # 交易逻辑
    }
    
    # 检查入口点
    entry_patterns = [
        r'function\s+main\s*\(',      # JavaScript/TypeScript main函数
        r'def\s+main\s*\(',           # Python main函数
        r'function\s+onTick\s*\(',    # FMZ tick回调
        r'def\s+on_tick\s*\(',        # Python tick回调
        r'function\s+init\s*\(',      # 初始化函数
        r'def\s+init\s*\(',           # Python初始化函数
        r'class.*Strategy'            # 策略类
    ]
    if any(re.search(pattern, code_content, re.I | re.M) for pattern in entry_patterns):
        required_components["entry_point"] = True
    
    # 检查交易所 API 调用
    exchange_patterns = [
        r'exchange\.',
        r'GetAccount|GetTicker|GetDepth|GetRecords',
        r'Buy|Sell|GetOrder|CancelOrder',
        r'buy|sell|trade|order',
        r'position|positions|portfolio',
        r'market|price|volume'
    ]
    if any(re.search(pattern, code_content, re.I) for pattern in exchange_patterns):
        required_components["exchange_api"] = True
    
    # 检查基本交易逻辑
    trade_patterns = [
        r'if.*buy|if.*sell|if.*trade|if.*order',
        r'while.*trade|for.*trade',
        r'position|amount|price|volume',
        r'strategy|signal|indicator',
        r'MA|MACD|RSI|KDJ|BOLL|ATR',
        r'trend|momentum|oscillator',
        r'long|short|entry|exit'
    ]
    if any(re.search(pattern, code_content, re.I) for pattern in trade_patterns):
        required_components["trade_logic"] = True
    
    is_valid = all(required_components.values())
    missing_components = [k for k, v in required_components.items() if not v]
    
    return is_valid, missing_components

def process_dataset(input_dir, output_dir, output_file):
    """处理整个数据集"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_file = output_dir / output_file
    
    # 检查fmz_strategies仓库是否存在，不存在则克隆
    if not input_dir.exists():
        logger.info("FMZ策略仓库不存在，正在克隆...")
        os.system(f"git clone https://github.com/fmzquant/strategies.git {input_dir}")
        if not input_dir.exists():
            logger.error(f"克隆仓库失败，目录不存在: {input_dir}")
            return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 查找所有策略文件
    strategy_files = []
    supported_extensions = ['*.py', '*.md', '*.js', '*.lua', '*.txt']
    
    for ext in supported_extensions:
        files = list(input_dir.rglob(ext))
        logger.info(f"找到 {len(files)} 个 {ext} 文件")
        strategy_files.extend(files)
    
    if not strategy_files:
        logger.warning("未找到任何策略文件！")
        return
    
    logger.info(f"共找到 {len(strategy_files)} 个策略文件")
    
    # 创建错误日志文件
    error_log_file = output_dir / "error_log.jsonl"
    
    # 创建统计信息文件
    stats_file = output_dir / "dataset_stats.json"
    
    # 初始化统计信息
    stats = {
        "total_files": len(strategy_files),
        "processed_files": 0,
        "error_files": 0,
        "error_types": {},
        "strategy_types": {},
        "features": {},
        "languages": {},
        "complexity_levels": {
            "low": 0,
            "medium": 0,
            "high": 0
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f, \
         open(error_log_file, 'w', encoding='utf-8') as error_f:
        
        for i, file_path in enumerate(strategy_files):
            if (i + 1) % 10 == 0 or (i + 1) == len(strategy_files):
                logger.info(f"进度: {i + 1}/{len(strategy_files)} ({((i + 1) / len(strategy_files)) * 100:.1f}%)")
            
            # 处理单个文件
            strategy_sample, error = process_strategy_file(file_path)
            
            if error:
                # 记录错误
                error_log = {
                    "file": str(file_path),
                    "error": error,
                    "timestamp": str(Path(file_path).stat().st_mtime)
                }
                error_f.write(json.dumps(error_log, ensure_ascii=False) + "\n")
                
                # 更新错误统计
                stats["error_files"] += 1
                stats["error_types"][error] = stats["error_types"].get(error, 0) + 1
                continue
                
            # 处理成功，写入数据并更新统计信息
            f.write(json.dumps(strategy_sample, ensure_ascii=False) + "\n")
            stats["processed_files"] += 1
            
            # 更新策略类型统计
            for stype in strategy_sample["metadata"]["strategy_patterns"]:
                stats["strategy_types"][stype] = stats["strategy_types"].get(stype, 0) + 1
            
            # 更新特征统计
            for feature in strategy_sample["metadata"]["strategy_features"]:
                stats["features"][feature] = stats["features"].get(feature, 0) + 1
            
            # 更新语言统计
            ext = Path(file_path).suffix.lower()
            stats["languages"][ext] = stats["languages"].get(ext, 0) + 1
            
            # 更新复杂度统计
            complexity = strategy_sample["metadata"]["training_labels"]["complexity_level"]
            stats["complexity_levels"][complexity] += 1
    
    # 保存统计信息
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 输出处理结果
    logger.info("数据集处理完成！")
    logger.info(f"成功处理: {stats['processed_files']} 个文件")
    logger.info(f"处理失败: {stats['error_files']} 个文件")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"文件大小: {output_file.stat().st_size / (1024 * 1024):.2f} MB")
    
    # 输出详细统计信息
    logger.info("\n=== 数据集统计信息 ===")
    logger.info("\n策略类型分布:")
    for stype, count in sorted(stats["strategy_types"].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats["processed_files"]) * 100
        logger.info(f"  - {stype}: {count} ({percentage:.1f}%)")
    
    logger.info("\n特征分布:")
    for feature, count in sorted(stats["features"].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats["processed_files"]) * 100
        logger.info(f"  - {feature}: {count} ({percentage:.1f}%)")
    
    logger.info("\n编程语言分布:")
    for lang, count in sorted(stats["languages"].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats["processed_files"]) * 100
        logger.info(f"  - {lang}: {count} ({percentage:.1f}%)")
    
    logger.info("\n代码复杂度分布:")
    for level, count in stats["complexity_levels"].items():
        percentage = (count / stats["processed_files"]) * 100
        logger.info(f"  - {level}: {count} ({percentage:.1f}%)")
    
    logger.info("\n错误类型统计:")
    for error_type, count in sorted(stats["error_types"].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats["error_files"]) * 100
        logger.info(f"  - {error_type}: {count} ({percentage:.1f}%)")

def normalize_code(code_content: str) -> str:
    """规范化代码格式"""
    # 移除多余的空行
    code_lines = code_content.splitlines()
    normalized_lines = []
    prev_empty = False
    
    for line in code_lines:
        if not line.strip():
            if not prev_empty:
                normalized_lines.append(line)
                prev_empty = True
        else:
            normalized_lines.append(line)
            prev_empty = False
    
    # 确保文件以换行符结束
    normalized_code = '\n'.join(normalized_lines)
    if normalized_code and not normalized_code.endswith('\n'):
        normalized_code += '\n'
    
    # 移除行尾空白字符
    normalized_code = '\n'.join(line.rstrip() for line in normalized_code.splitlines())
    
    # 统一缩进风格（使用空格）
    normalized_code = normalized_code.replace('\t', '    ')
    
    return normalized_code

def process_strategy_file(file_path: Path, stats: ProcessingStats, logger=None) -> Dict:
    """处理单个策略文件"""
    try:
        if logger:
            logger.info(f"正在处理文件: {file_path}")
        
        # 1. 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 2. 读取文件内容
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
            except UnicodeDecodeError as e:
                raise ValueError(f"无法读取文件编码: {str(e)}")
        
        # 3. 提取代码
        code = extract_code(file_path)
        if not code:
            raise ValueError(f"未能提取到有效代码: {file_path}")
        
        # 4. 提取元数据
        metadata = extract_metadata(file_path, logger)
        
        # 5. 分析策略模式和特征
        strategy_types, strategy_features = analyze_strategy_pattern(code)
        
        # 6. 验证策略代码
        is_valid, missing = validate_fmz_strategy(code)
        if not is_valid:
            raise ValueError(f"无效的策略代码，缺少组件: {', '.join(missing)}")
        
        # 7. 计算代码复杂度指标
        complexity_metrics = calculate_complexity_metrics(code)
        
        # 8. 提取回测配置
        backtest_config = extract_backtest_config(code)
        
        # 9. 提取训练标签
        training_labels = extract_training_labels(code)
        
        result = {
            "metadata": {
                "file_path": str(file_path),
                "name": file_path.stem,
                "strategy_patterns": strategy_types,
                "strategy_features": strategy_features,
                "complexity_metrics": complexity_metrics,
                "backtest_config": backtest_config,
                "training_labels": training_labels
            },
            "code": code
        }
        
        stats.valid_strategies += 1
        if logger:
            logger.info(f"成功处理文件 {file_path}")
        return result
        
    except Exception as e:
        stats.failed_files += 1
        stats.add_error(str(file_path), str(e))
        if logger:
            logger.error(f"处理文件失败 {file_path}: {str(e)}")
        return None

def main():
    try:
        args = parse_args()
        input_dir = Path(args.input_dir).resolve()
        output_dir = Path(args.output_dir).resolve()
        output_file = output_dir / args.output_file
        
        # 创建输出目录并设置日志
        output_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(output_dir)
        
        logger.info(f"输入目录: {input_dir}")
        logger.info(f"输出目录: {output_dir}")
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化统计信息
        stats = ProcessingStats()
        stats.set_logger(logger)
        
        # 获取所有策略文件
        file_patterns = ["**/*.md", "**/*.js", "**/*.txt"]
        strategy_files = []
        for pattern in file_patterns:
            strategy_files.extend(list(input_dir.glob(pattern)))
        
        stats.total_files = len(strategy_files)
        logger.info(f"找到 {stats.total_files} 个策略文件")
        
        # 处理所有文件
        results = []
        with tqdm(total=stats.total_files, desc="处理策略文件") as pbar:
            for file_path in strategy_files:
                try:
                    result = process_strategy_file(file_path, stats, logger)
                    if result:
                        results.append(result)
                    stats.processed_files += 1
                except Exception as e:
                    logger.error(f"处理文件失败 {file_path}: {str(e)}")
                    stats.failed_files += 1
                finally:
                    pbar.update(1)
        
        # 保存处理结果
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
        # 保存处理统计信息
        stats_file = output_dir / 'processing_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats.get_summary(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"处理完成。发现 {stats.total_files} 个文件")
        logger.info(f"成功处理 {stats.valid_strategies} 个策略")
        logger.info(f"失败 {stats.failed_files} 个文件")
        logger.info(f"结果已保存到 {output_file}")
        logger.info(f"处理统计信息已保存到 {stats_file}")
        
    except Exception as e:
        logger.error(f"处理过程出错: {str(e)}")
        raise

if __name__ == '__main__':
    main()