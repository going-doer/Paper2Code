
> Name

动态趋势确认的反转公平价值缺口量化交易策略-Dynamic-Trend-Confirmed-Inverted-Fair-Value-Gap-Quantitative-Trading-Strategy

> Author

ianzeng123

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/2d95ad8e9020da30381cc.png)
![IMG](https://www.fmz.com/upload/asset/2d84d71e79ddd93cab926.png)

[trans]
#### 概述
该策略是一个基于反转公平价值缺口(Inverted Fair Value Gap, IFVG)的量化交易系统,结合了移动平均线趋势确认和动态追踪止损机制。策略通过识别价格行为中的公平价值缺口(FVG)及其反转形态,并在趋势支持的情况下进行交易。这种方法既确保了交易方向与整体市场趋势保持一致,又能够捕捉市场中的关键反转点。

#### 策略原理
策略的核心逻辑包含以下几个关键步骤:
1. FVG检测:通过分析当前蜡烛线与前一根蜡烛线的价格区间重叠情况来识别公平价值缺口。
2. IFVG确认:当价格突破FVG的高点或低点时,形成反转信号。
3. 趋势确认:使用50期和200期简单移动平均线(SMA)的交叉关系来确定市场趋势。
4. 入场条件:在上升趋势中,当价格低于IFVG低点时做多;在下降趋势中,当价格高于IFVG高点时做空。
5. 风险管理:采用固定止损和基于ATR的动态追踪止损相结合的方式。

#### 策略优势
1. 多维度确认:结合价格结构(IFVG)和趋势指标(SMA)的多层面分析,提高交易的可靠性。
2. 动态风险管理:通过ATR指标调整追踪止损,既保护已有利润,又给予价格足够的波动空间。
3. 风险收益比优化:采用3R的盈利目标设置,在合理风险控制的基础上追求更高收益。
4. 趋势过滤:通过移动平均线交叉确认趋势,避免在横盘市场过度交易。

#### 策略风险
1. 滑点风险:在市场波动剧烈时,实际成交价可能与理想价格存在偏差。
2. 趋势延迟:移动平均线作为滞后指标,可能导致入场时机略有延迟。
3. 假突破风险:价格可能在突破后快速回撤,触发止损。
4. 参数敏感性:战略表现对SMA周期和ATR倍数等参数设置较为敏感。

#### 策略优化方向
1. 指标优化:可考虑添加成交量确认信号,提高突破的可靠性。
2. 参数自适应:引入市场波动率指标,动态调整SMA周期和ATR倍数。
3. 入场时机优化:增加价格回调确认机制,避免追高或追低。
4. 仓位管理:根据市场波动率和趋势强度动态调整仓位大小。
5. 止盈机制优化:在强势趋势中可采用更宽松的追踪止损参数。

#### 总结
该策略通过结合IFVG价格结构、趋势确认和动态风险管理,构建了一个完整的交易系统。策略在保持简洁性的同时,充分考虑了市场趋势、风险控制和利润管理等关键要素。通过建议的优化方向,策略可以进一步提升其适应性和稳定性。在实盘交易中,建议进行充分的回测和参数优化,并根据具体市场特征做相应调整。

||

#### Overview
This strategy is a quantitative trading system based on Inverted Fair Value Gap (IFVG), combining moving average trend confirmation and dynamic trailing stop mechanisms. The strategy identifies Fair Value Gaps (FVG) in price action and their inversions, executing trades when supported by the trend. This approach ensures trade direction aligns with overall market trends while capturing key reversal points.

#### Strategy Principles
The core logic includes several key steps:
1. FVG Detection: Analyzes the overlap between current and previous candle price ranges to identify fair value gaps.
2. IFVG Confirmation: Forms a reversal signal when price breaks beyond FVG's high or low.
3. Trend Confirmation: Uses crossover relationship between 50-period and 200-period Simple Moving Averages (SMA) to determine market trend.
4. Entry Conditions: Goes long when price is below IFVG low in uptrends; goes short when price is above IFVG high in downtrends.
5. Risk Management: Combines fixed stop-loss with ATR-based dynamic trailing stops.

#### Strategy Advantages
1. Multi-dimensional Confirmation: Combines price structure (IFVG) and trend indicators (SMA) for more reliable analysis.
2. Dynamic Risk Management: Adjusts trailing stops using ATR indicator, protecting profits while allowing adequate price movement.
3. Risk-Reward Optimization: Implements 3R profit targets, pursuing higher returns with reasonable risk control.
4. Trend Filtering: Confirms trends through moving average crossovers, avoiding excessive trading in ranging markets.

#### Strategy Risks
1. Slippage Risk: Actual execution prices may deviate from ideal prices during high volatility.
2. Trend Lag: Moving averages as lagging indicators may cause slightly delayed entries.
3. False Breakout Risk: Price may quickly retrace after breakouts, triggering stops.
4. Parameter Sensitivity: Strategy performance is sensitive to SMA periods and ATR multiplier settings.

#### Strategy Optimization Directions
1. Indicator Enhancement: Consider adding volume confirmation signals to improve breakout reliability.
2. Parameter Adaptation: Introduce volatility metrics to dynamically adjust SMA periods and ATR multipliers.
3. Entry Timing Optimization: Add price pullback confirmation mechanism to avoid chasing highs or lows.
4. Position Sizing: Dynamically adjust position sizes based on market volatility and trend strength.
5. Profit-Taking Optimization: Use more relaxed trailing stop parameters in strong trends.

#### Summary
The strategy builds a complete trading system by combining IFVG price structure, trend confirmation, and dynamic risk management. While maintaining simplicity, it thoroughly considers key elements like market trends, risk control, and profit management. Through suggested optimizations, the strategy can further enhance its adaptability and stability. For live trading, thorough backtesting and parameter optimization are recommended, with adjustments made according to specific market characteristics.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2025-02-16 17:00:00
end: 2025-02-18 03:00:00
period: 2m
basePeriod: 2m
exchanges: [{"eid":"Binance","currency":"SOL_USDT"}]
*/

//@version=6
strategy("Inverted FVG Strategy with Trend Check and Trailing Stops", overlay=true)

// Function to detect FVG
fvgDetected(src, high, low) =>
    var float prevHigh = na
    var float prevLow = na
    var float prevClose = na
    var bool fvg = false
    if (not na(src[1]))
        prevHigh := high[1]
        prevLow := low[1]
        prevClose := src[1]
        if (src > prevClose and low > prevHigh) or (src < prevClose and high < prevLow)
            fvg := true
    [fvg, prevHigh, prevLow]

// Detect FVG on the chart
[fvg, fvgHigh, fvgLow] = fvgDetected(close, high, low)

// Detect IFVG - Inversion of FVG
var bool ifvg = false
var float ifvgHigh = na
var float ifvgLow = na
if (fvg)
    if not na(fvgHigh) and not na(fvgLow)
        if (close > fvgHigh and close[1] < fvgHigh) or (close < fvgLow and close[1] > fvgLow)
            ifvg := true
            ifvgHigh := close > fvgHigh ? high : na
            ifvgLow := close < fvgLow ? low : na

// Plot FVG and IFVG zones for visualization
bgcolor(ifvg ? color.new(color.red, 80) : na)
plot(ifvgHigh, title="IFVG High", color=color.red, linewidth=2, style=plot.style_cross)
plot(ifvgLow, title="IFVG Low", color=color.red, linewidth=2, style=plot.style_cross)

// Trend Check using Simple Moving Averages
smaShort = ta.sma(close, 50)  // Short term SMA
smaLong = ta.sma(close, 200)  // Long term SMA
var bool uptrend = false
var bool downtrend = false

uptrend := smaShort > smaLong  // Up trend if short SMA is above long SMA
downtrend := smaShort < smaLong  // Down trend if short SMA is below long SMA

// Plot SMAs for visualization
plot(smaShort, title="SMA Short", color=color.blue, linewidth=1)
plot(smaLong, title="SMA Long", color=color.orange, linewidth=1)

// Trading logic with trend confirmation
longCondition = ifvg and close < ifvgLow and uptrend
shortCondition = ifvg and close > ifvgHigh and downtrend

// Risk Definition
stopLoss = 100  // Example values in points
risk = stopLoss

if (longCondition)
    strategy.entry("Long", strategy.long)
    strategy.exit("Initial Long Exit", "Long", stop=close - stopLoss, limit=close + (risk * 3))  // Changed to 3R

if (shortCondition)
    strategy.entry("Short", strategy.short)
    strategy.exit("Initial Short Exit", "Short", stop=close + stopLoss, limit=close - (risk * 3))  // Changed to 3R

// ATR for dynamic trailing stop
atr = ta.atr(14)

// Trailing Stop for Long Position if the trade has moved > 0.5R
if (strategy.position_size > 0)
    if (close - strategy.position_avg_price >= risk * 0.5)
        trailingStopLong = math.max(strategy.position_avg_price + (risk * 0.5), close - (atr * 2))
        strategy.exit("Trailing Stop Long", "Long", stop=trailingStopLong)

// Trailing Stop for Short Position if the trade has moved > 0.5R
if (strategy.position_size < 0)
    if (strategy.position_avg_price - close >= risk * 0.5)
        trailingStopShort = math.min(strategy.position_avg_price - (risk * 0.5), close + (atr * 2))
        strategy.exit("Trailing Stop Short", "Short", stop=trailingStopShort)
```

> Detail

https://www.fmz.com/strategy/483081

> Last Modified

2025-02-21 11:55:42
