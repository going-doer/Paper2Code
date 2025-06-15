# encoding patterns
ENCODING_PATTERNS = {
    "utf8": ["utf-8", "utf_8", "UTF-8"],
    "gbk": ["gbk", "gb2312", "GBK", "GB2312"],
    "ascii": ["ascii", "ASCII"],
}

# 策略模式关键词
STRATEGY_PATTERNS = {
    "trend_following": [
        "趋势跟踪", "突破", "动量", "趋势", "方向", "走势",
        "均线", "MACD", "KDJ", "RSI"
    ],
    "mean_reversion": [
        "均值回归", "震荡", "超买", "超卖", "回归", "反转",
        "布林带", "相对强弱", "背离"
    ],
    "arbitrage": [
        "套利", "对冲", "价差", "配对", "统计套利",
        "跨市", "跨期", "跨品种"
    ],
    "market_making": [
        "做市", "报价", "买卖价差", "流动性",
        "挂单", "深度", "订单簿"
    ],
    "ml_based": [
        "机器学习", "神经网络", "深度学习", "AI", "人工智能",
        "训练", "预测", "分类", "聚类"
    ],
}

# 技术指标
TECHNICAL_INDICATORS = {
    "moving_averages": [
        "MA", "EMA", "MACD", "WMA", "SMA",
        "移动平均", "指数平均", "加权平均"
    ],
    "oscillators": [
        "RSI", "KDJ", "CCI", "BOLL", "ATR",
        "相对强弱", "随机指标", "布林带", "真实波幅"
    ],
    "volume": [
        "OBV", "VWAP", "成交量", "成交额",
        "能量潮", "委比", "筹码"
    ],
    "price_action": [
        "K线", "蜡烛图", "阻力", "支撑",
        "突破", "回调", "趋势线"
    ],
}

# FMZ API 模式
FMZ_API_PATTERNS = {
    "trading_api": [
        "exchange.Buy", "exchange.Sell",
        "exchange.GetAccount", "exchange.GetTicker",
        "exchange.GetDepth", "exchange.GetRecords"
    ],
    "strategy_api": [
        "Strategy", "main", "init", "onTick",
        "onBar", "onexit", "Log", "Sleep"
    ],
    "indicator_api": [
        "TA.MA", "TA.EMA", "TA.MACD",
        "TA.BOLL", "TA.KDJ", "TA.RSI"
    ],
    "utility_api": [
        "Math.abs", "Math.max", "Math.min",
        "JSON.stringify", "JSON.parse",
        "_G", "LogProfit", "LogError"
    ],
}

# 代码验证规则
CODE_VALIDATION_RULES = {
    "min_length": 50,  # 最小代码长度
    "max_length": 50000,  # 最大代码长度
    "required_functions": ["main", "init"],  # 必需的函数
    "banned_keywords": ["eval", "exec"],  # 禁用的关键字
    "max_nesting_level": 5,  # 最大嵌套层级
}

# 文件类型和扩展名
FILE_EXTENSIONS = {
    "strategy": [".js", ".py", ".pine"],
    "document": [".md", ".txt"],
    "config": [".json", ".yaml", ".ini"],
}

# 日志配置
LOG_CONFIG = {
    "file_name": "process_fmz_dataset.log",
    "max_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 3,
    "log_level": "INFO",
}
