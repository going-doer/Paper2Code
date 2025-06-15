
> Name

SMA-FVG-SMA交叉策略与公平价值缺口回调结合的综合交易系统Comprehensive-Trading-System-Combining-SMA-Crossover-Strategy-with-Fair-Value-Gap-Pullback

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/fc8e64b0defbd8b346.png)

[trans]

#### 概述

本策略是一个结合了简单移动平均线(SMA)交叉和公平价值缺口(FVG)回调的综合交易系统。它利用8周期和20周期SMA的交叉来识别潜在的趋势变化,同时利用FVG来确定更精确的入场点。这种方法旨在捕捉市场趋势的变化,同时通过等待价格回调到关键支撑/阻力区域来优化入场时机。

#### 策略原理

1. SMA交叉:使用8周期和20周期的简单移动平均线。当短期SMA上穿长期SMA时,视为看涨信号;当短期SMA下穿长期SMA时,视为看跌信号。

2. 公平价值缺口(FVG):FVG是指当前蜡烛的高点高于前一根蜡烛的高点,且当前蜡烛的低点低于前一根蜡烛的低点时形成的价格区间。这个区间被认为是市场在寻找"公平价值"。

3. 入场条件:
   - 多头:当出现看涨SMA交叉,并且价格回调至FVG的低点时入场。
   - 空头:当出现看跌SMA交叉,并且价格反弹至FVG的高点时入场。

4. 出场条件:当出现相反方向的SMA交叉时平仓。

#### 策略优势

1. 趋势跟随与回调结合:通过结合SMA交叉和FVG回调,策略既能捕捉大趋势,又能在更有利的价格水平入场。

2. 减少虚假信号:等待价格回调到FVG可以过滤掉一些可能的虚假交叉信号,提高交易的准确性。

3. 风险管理:使用FVG作为入场点可以自然地提供更紧的止损位置,有助于控制风险。

4. 适应性强:通过调整SMA周期和FVG参数,策略可以适应不同的市场环境和交易品种。

5. 客观性:基于明确的技术指标和价格行为,减少了主观判断的影响。

#### 策略风险

1. 震荡市场风险:在横盘或震荡市场中,频繁的SMA交叉可能导致过多交易和亏损。

2. 滞后性:SMA作为滞后指标,可能在趋势初期错过一些机会。

3. 假突破风险:价格可能短暂突破FVG后又回落,导致虚假信号。

4. 市场缺口风险:在剧烈波动的市场中,价格可能跳过FVG区域,导致错过交易机会。

5. 参数敏感性:策略表现可能对SMA周期和FVG定义参数敏感,需要仔细优化。

#### 策略优化方向

1. 动态SMA周期:可以考虑根据市场波动性动态调整SMA周期,以适应不同市场状况。

2. 增加过滤条件:引入额外的技术指标(如RSI或MACD)来确认趋势,减少虚假信号。

3. 改进FVG定义:可以尝试使用多根K线来定义FVG,或者考虑成交量来验证FVG的有效性。

4. 优化出场策略:可以引入跟踪止损或基于波动率的动态止损,以更好地保护利润。

5. 加入时间过滤:考虑FVG的形成时间,可能需要设置一个时间窗口来保证FVG的有效性。

6. 风险管理优化:根据市场波动性动态调整仓位大小,实现更精细的风险控制。

#### 总结

"SMA交叉策略与公平价值缺口回调结合的综合交易系统"是一个融合了趋势跟随和价格回调的智能交易策略。通过结合SMA交叉信号和FVG回调,该策略旨在在趋势初期以更优的价格水平进行交易。虽然策略具有捕捉趋势和优化入场点的潜力,但仍面临震荡市场和参数优化等挑战。通过进一步优化和改进,如动态调整参数、增加过滤条件和改进风险管理,该策略有望在各种市场环境中取得更稳健的表现。交易者在使用此策略时,应当充分理解其原理,并根据具体交易品种和市场状况进行适当的调整和测试。

||

#### Overview

This strategy is a comprehensive trading system that combines Simple Moving Average (SMA) crossovers with Fair Value Gap (FVG) pullbacks. It utilizes the crossover of 8-period and 20-period SMAs to identify potential trend changes, while using FVGs to determine more precise entry points. This approach aims to capture market trend shifts while optimizing entry timing by waiting for price pullbacks to key support/resistance areas.

#### Strategy Principles

1. SMA Crossover: Uses 8-period and 20-period simple moving averages. A bullish signal is generated when the short-term SMA crosses above the long-term SMA, and a bearish signal when the short-term SMA crosses below the long-term SMA.

2. Fair Value Gap (FVG): An FVG is formed when the current candle's high is higher than the previous candle's high, and the current candle's low is lower than the previous candle's low. This price range is considered where the market is seeking "fair value."

3. Entry Conditions:
   - Long: Enter when a bullish SMA crossover occurs and price pulls back to the low of the FVG.
   - Short: Enter when a bearish SMA crossover occurs and price rebounds to the high of the FVG.

4. Exit Conditions: Close positions when an opposite SMA crossover occurs.

#### Strategy Advantages

1. Combines Trend Following and Pullbacks: By integrating SMA crossovers and FVG pullbacks, the strategy can capture major trends while entering at more favorable price levels.

2. Reduces False Signals: Waiting for price to pull back to the FVG can filter out some potential false crossover signals, improving trade accuracy.

3. Risk Management: Using FVGs as entry points naturally provides tighter stop-loss placements, helping to control risk.

4. Adaptability: The strategy can be adapted to different market environments and trading instruments by adjusting SMA periods and FVG parameters.

5. Objectivity: Based on clear technical indicators and price action, reducing the impact of subjective judgement.

#### Strategy Risks

1. Choppy Market Risk: In range-bound or choppy markets, frequent SMA crossovers may lead to excessive trading and losses.

2. Lag: As a lagging indicator, SMAs may miss some opportunities at the beginning of trends.

3. False Breakout Risk: Price may briefly break through the FVG and then retreat, causing false signals.

4. Market Gap Risk: In volatile markets, price may gap over the FVG area, leading to missed trading opportunities.

5. Parameter Sensitivity: Strategy performance may be sensitive to SMA periods and FVG definition parameters, requiring careful optimization.

#### Strategy Optimization Directions

1. Dynamic SMA Periods: Consider dynamically adjusting SMA periods based on market volatility to adapt to different market conditions.

2. Additional Filters: Introduce extra technical indicators (such as RSI or MACD) to confirm trends and reduce false signals.

3. Improve FVG Definition: Try using multiple candles to define FVGs, or consider volume to validate FVG effectiveness.

4. Optimize Exit Strategy: Implement trailing stops or volatility-based dynamic stops to better protect profits.

5. Add Time Filters: Consider the formation time of FVGs, potentially setting a time window to ensure FVG validity.

6. Risk Management Optimization: Dynamically adjust position sizes based on market volatility for more refined risk control.

#### Conclusion

The "Comprehensive Trading System Combining SMA Crossover Strategy with Fair Value Gap Pullback" is an intelligent trading strategy that fuses trend following with price pullbacks. By combining SMA crossover signals and FVG pullbacks, the strategy aims to trade at more optimal price levels in the early stages of trends. While the strategy has the potential to capture trends and optimize entry points, it still faces challenges such as choppy markets and parameter optimization. Through further optimization and improvements, such as dynamic parameter adjustments, additional filtering conditions, and enhanced risk management, this strategy has the potential to achieve more robust performance across various market environments. Traders using this strategy should fully understand its principles and make appropriate adjustments and tests based on specific trading instruments and market conditions.

[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-07-25 00:00:00
end: 2024-07-30 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("8 SMA and 20 SMA with FVG Pullback", overlay=true)

// Input parameters
smaShortLength = input.int(8, title="Short SMA Length")
smaLongLength = input.int(20, title="Long SMA Length")

// Calculate SMAs
smaShort = ta.sma(close, smaShortLength)
smaLong = ta.sma(close, smaLongLength)

// Plot SMAs
plot(smaShort, title="8 SMA", color=color.blue)
plot(smaLong, title="20 SMA", color=color.red)

// Identify SMA crossovers
longCondition = ta.crossover(smaShort, smaLong)
shortCondition = ta.crossunder(smaShort, smaLong)

// Fair Value Gaps (FVG) logic
var float fvgHigh = na
var float fvgLow = na

if (ta.valuewhen(high[1] < high and low[1] > low, high, 0) and ta.valuewhen(high[1] < high and low[1] > low, low, 0))
    fvgHigh := high
    fvgLow := low

plot(fvgHigh, title="FVG High", color=color.purple, linewidth=1, style=plot.style_line)
plot(fvgLow, title="FVG Low", color=color.orange, linewidth=1, style=plot.style_line)

// Entry conditions
if (longCondition)
    if (low <= fvgLow)
        strategy.entry("Long", strategy.long)
        
if (shortCondition)
    if (high >= fvgHigh)
        strategy.entry("Short", strategy.short)
        
// Exit conditions (optional, you can modify these as per your risk management strategy)
if (ta.crossunder(smaShort, smaLong))
    strategy.close("Long")
    
if (ta.crossover(smaShort, smaLong))
    strategy.close("Short")

```

> Detail

https://www.fmz.com/strategy/458272

> Last Modified

2024-07-31 14:38:42
