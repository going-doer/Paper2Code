
> Name

Heikin-Ashi非重绘趋势确认策略-Heikin-Ashi-Non-Repainting-Trend-Confirmation-Strategy

> Author

ianzeng123

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/2d8a09cea77ea08c8ae4e.png)
![IMG](https://www.fmz.com/upload/asset/2d90ab38248313c4bc2a1.png)





[trans]
#### 概述
这是一种创新的海坎阿西(Heikin-Ashi)非重绘趋势确认策略，旨在解决传统交易视图(TradingView)中海坎阿西策略存在的重绘问题。通过手动计算海坎阿西蜡烛和多重趋势确认机制，该策略提供了一种更加可靠和透明的交易方法。

#### 策略原理
策略的核心原理包括三个关键步骤：
1. 手动非重绘海坎阿西蜡烛计算：
   - 使用独特的公式计算收盘价、开盘价、最高价和最低价
   - 确保历史价格数据在后续K线更新时保持稳定
   - 避免传统海坎阿西策略中常见的重绘问题

2. 多重趋势确认：
   - 要求连续多根蜡烛确认趋势方向
   - 长入场信号：需要连续X根看涨蜡烛
   - 空入场信号：需要连续X根看跌蜡烛
   - 通过多重确认过滤掉假信号，提高策略可靠性

3. 灵活交易模式：
   - 支持传统趋势跟随模式
   - 提供趋势反转交易选项
   - 可自定义交易模式（全部、仅多、仅空）

#### 策略优势
1. 消除重绘问题：历史数据保持稳定，回测结果与实盘执行高度一致
2. 多重趋势确认：通过连续蜡烛过滤假信号，降低不必要交易
3. 高度可定制：
   - 灵活的入场和出场阈值设置
   - 支持趋势跟随和反转交易
   - 可隐藏标准K线，提供清晰可视化
4. 适用于中长期交易：特别适合摆动交易和趋势跟随

#### 策略风险
1. 性能局限性：
   - 不适合高频scalping交易
   - 在趋势不明显的震荡市场表现可能较差
   - 需要针对不同时间框架调整参数

2. 潜在风险控制：
   - 建议设置适当的止损机制
   - 在不同市场条件下持续优化参数
   - 结合其他技术指标进行交叉验证

#### 策略优化方向
1. 参数动态调整：
   - 开发自适应入场和出场阈值算法
   - 根据市场波动性实时调整连续蜡烛数量
   - 引入机器学习算法优化参数选择

2. 风险管理增强：
   - 集成动态仓位管理
   - 添加相关性过滤器
   - 开发更智能的止损机制

3. 指标组合：
   - 结合其他技术指标（如RSI、MACD）
   - 开发多指标确认系统
   - 提高信号的准确性和可靠性

#### 总结
海坎阿西非重绘趋势确认策略通过创新的蜡烛计算和多重趋势确认方法，为交易者提供了一个更加可靠和透明的交易工具。通过消除重绘问题、过滤假信号和提供灵活的交易模式，该策略展现了量化交易中的技术创新潜力。

|| 
#### Overview
This is an innovative Heikin-Ashi Non-Repainting Trend Confirmation Strategy designed to address the repainting issues in traditional TradingView Heikin-Ashi strategies. By manually calculating Heikin-Ashi candles and implementing multi-stage trend confirmation mechanisms, the strategy offers a more reliable and transparent trading approach.

#### Strategy Principles
The strategy's core principles include three key steps:
1. Manual Non-Repainting Heikin-Ashi Candle Calculation:
   - Utilizing unique formulas to calculate close, open, high, and low prices
   - Ensuring historical price data remains stable during subsequent candle updates
   - Avoiding common repainting problems in traditional Heikin-Ashi strategies

2. Multi-Stage Trend Confirmation:
   - Requiring consecutive candles to confirm trend direction
   - Long entry signals: X consecutive bullish candles
   - Short entry signals: X consecutive bearish candles
   - Filtering false signals through multi-confirmation, enhancing strategy reliability

3. Flexible Trading Modes:
   - Supporting traditional trend-following modes
   - Offering trend reversal trading options
   - Customizable trading modes (all, long only, short only)

#### Strategy Advantages
1. Eliminates Repainting: Stable historical data with consistent backtest and live execution results
2. Multi-Trend Confirmation: Filtering false signals through consecutive candles, reducing unnecessary trades
3. High Customizability:
   - Flexible entry and exit threshold settings
   - Supports trend-following and reversal trading
   - Option to hide standard candles for clear visualization
4. Suitable for Medium to Long-Term Trading: Particularly effective for swing trading and trend following

#### Strategy Risks
1. Performance Limitations:
   - Not suitable for high-frequency scalping
   - Potential underperformance in range-bound markets with unclear trends
   - Requires parameter adjustments for different timeframes

2. Potential Risk Mitigation:
   - Recommend implementing appropriate stop-loss mechanisms
   - Continuous parameter optimization across market conditions
   - Cross-verification with additional technical indicators

#### Strategy Optimization Directions
1. Dynamic Parameter Adjustment:
   - Develop adaptive entry and exit threshold algorithms
   - Real-time adjustment of consecutive candle counts based on market volatility
   - Integrate machine learning algorithms for parameter optimization

2. Enhanced Risk Management:
   - Implement dynamic position sizing
   - Add correlation filters
   - Develop more intelligent stop-loss mechanisms

3. Indicator Combination:
   - Integrate with other technical indicators (e.g., RSI, MACD)
   - Develop multi-indicator confirmation systems
   - Improve signal accuracy and reliability

#### Conclusion
The Heikin-Ashi Non-Repainting Trend Confirmation Strategy provides traders with a more reliable and transparent trading tool through innovative candle calculation and multi-stage trend confirmation methods. By eliminating repainting issues, filtering false signals, and offering flexible trading modes, the strategy demonstrates the technical innovation potential in quantitative trading.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2025-03-15 00:00:00
end: 2025-03-27 00:00:00
period: 3h
basePeriod: 3h
exchanges: [{"eid":"Futures_Binance","currency":"ETH_USDT"}]
*/

//@version=5
//© PineIndicators

strategy("Heikin-Ashi Non-Repainting Strategy [PineIndicators]", overlay=true, initial_capital=100000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, max_boxes_count=500, max_labels_count=500, max_lines_count=500, commission_value=0.01, process_orders_on_close=true, slippage= 2, behind_chart=false)

//====================================
// INPUTS
//====================================
// Number of consecutive candles required for entry and exit
openThreshold = input.int(title="Number of Candles for Entry", defval=2, minval=1)
exitThreshold = input.int(title="Number of Candles for Exit", defval=2, minval=1)
// Trade mode selection: "Long & Short", "Only Long", or "Only Short"
tradeMode = input.string(title="Trade Mode", defval="Only Long", options=["Long & Short", "Only Long", "Only Short"])
// Option to invert the trading logic (bullish signals become short signals, and vice versa)
invertTrades = input.bool(title="Invert Trading Logic (Long ↔ Short)", defval=false)
// Option to hide the standard candles (bodies only)
hideStandard = input.bool(title="Hide Standard Candles", defval=true)
// Heikin-Ashi transparency is fixed (0 = fully opaque)
heikinTransparency = 0

//====================================
// HIDE STANDARD CANDLES
//====================================
// Hide the body of the standard candles by setting them to 100% transparent.
// Note: The wicks of the standard candles cannot be hidden via code.
barcolor(hideStandard ? color.new(color.black, 100) : na)

//====================================
// HEIKIN-ASHI CALCULATION
//====================================
// Calculate Heikin-Ashi values manually
haClose = (open + high + low + close) / 4
var float haOpen = na
haOpen := na(haOpen[1]) ? (open + close) / 2 : (haOpen[1] + haClose[1]) / 2
haHigh = math.max(high, math.max(haOpen, haClose))
haLow  = math.min(low, math.min(haOpen, haClose))

// Define colors for Heikin-Ashi candles (using fixed transparency)
bullColor = color.new(#0097a7, heikinTransparency)
bearColor = color.new(#ff195f, heikinTransparency)

//====================================
// PLOT HEIKIN-ASHI CANDLES
//====================================
// Plot the manually calculated Heikin-Ashi candles over the chart.
// The candle body, wicks, and borders will be colored based on whether the candle is bullish or bearish.
plotcandle(haOpen, haHigh, haLow, haClose, title="Heikin-Ashi", 
     color       = haClose >= haOpen ? bullColor : bearColor,
     wickcolor   = haClose >= haOpen ? bullColor : bearColor,
     bordercolor = haClose >= haOpen ? bullColor : bearColor,
     force_overlay = true)

//====================================
// COUNT CONSECUTIVE TREND CANDLES
//====================================
// Count the number of consecutive bullish or bearish Heikin-Ashi candles.
var int bullishCount = 0
var int bearishCount = 0

if haClose > haOpen
    bullishCount := bullishCount + 1
    bearishCount := 0
else if haClose < haOpen
    bearishCount := bearishCount + 1
    bullishCount := 0
else
    bullishCount := 0
    bearishCount := 0

//====================================
// DEFINE ENTRY & EXIT SIGNALS
//====================================
// The signals are based on the number of consecutive trend candles.
// In normal logic: bullish candles trigger a long entry and bearish candles trigger a short entry.
// If invertTrades is enabled, the signals are swapped.
var bool longEntrySignal  = false
var bool shortEntrySignal = false
var bool exitLongSignal   = false
var bool exitShortSignal  = false

if not invertTrades
    longEntrySignal  := bullishCount >= openThreshold
    shortEntrySignal := bearishCount >= openThreshold
    exitLongSignal   := bearishCount >= exitThreshold
    exitShortSignal  := bullishCount >= exitThreshold
else
    // Inverted logic: bullish candles trigger short entries and bearish candles trigger long entries.
    longEntrySignal  := bearishCount >= openThreshold
    shortEntrySignal := bullishCount >= openThreshold
    exitLongSignal   := bullishCount >= exitThreshold
    exitShortSignal  := bearishCount >= exitThreshold

//====================================
// APPLY TRADE MODE RESTRICTIONS
//====================================
// If the user selects "Only Long", disable short signals (and vice versa).
if tradeMode == "Only Long"
    shortEntrySignal := false
    exitShortSignal  := false
else if tradeMode == "Only Short"
    longEntrySignal  := false
    exitLongSignal   := false

//====================================
// TRADING STRATEGY LOGIC
//====================================
// Execute trades based on the calculated signals.

// If a long position is open:
if strategy.position_size > 0
    if shortEntrySignal
        strategy.close("Long", comment="Reverse Long")
        strategy.entry("Short", strategy.short, comment="Enter Short")
    else if exitLongSignal
        strategy.close("Long", comment="Exit Long")

// If a short position is open:
if strategy.position_size < 0
    if longEntrySignal
        strategy.close("Short", comment="Reverse Short")
        strategy.entry("Long", strategy.long, comment="Enter Long")
    else if exitShortSignal
        strategy.close("Short", comment="Exit Short")

// If no position is open:
if strategy.position_size == 0
    if longEntrySignal
        strategy.entry("Long", strategy.long, comment="Enter Long")
    else if shortEntrySignal
        strategy.entry("Short", strategy.short, comment="Enter Short")

```

> Detail

https://www.fmz.com/strategy/488544

> Last Modified

2025-03-28 17:35:26
