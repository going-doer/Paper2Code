
> Name

基于支撑位和阻力位反转形态的量化交易策略-Quantitative-Trading-Strategy-Based-on-Reversal-Patterns-at-Support-and-Resistance-Levels

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1c1ab9816a5c9cabdee.png)

[trans]
#### 概述
该策略基于技术分析中的反转形态(锤头线、吞没形态和十字星)以及支撑位和阻力位,在1小时图表上进行交易。策略通过识别潜在的市场反转点,并在预定的止盈和止损水平执行交易。

该策略的主要思路是在支撑位附近出现看涨反转形态(如锤头线、看涨吞没形态或十字星)时开多仓,在阻力位附近出现看跌反转形态(如锤头线、看跌吞没形态或十字星)时开空仓。同时设置止盈和止损水平,以控制风险和锁定利润。

#### 策略原理
1. 通过 ta.lowest() 和 ta.highest() 函数分别计算指定回望期内的最低价和最高价,确定支撑位和阻力位。
2. 判断当前蜡烛图是否形成锤头线、吞没形态或十字星。
3. 如果在支撑位附近出现看涨反转形态,开多仓;如果在阻力位附近出现看跌反转形态,开空仓。
4. 设置止盈价格为开仓价格的3%,止损价格为开仓价格的1%。
5. 当价格达到止盈或止损水平时,平仓。

#### 策略优势
1. 结合反转形态和关键支撑阻力位,提高交易信号的可靠性。
2. 设置明确的止盈和止损水平,有效控制风险。
3. 适用于趋势和震荡市场,可捕捉潜在的反转机会。
4. 代码简洁,易于理解和实现。

#### 策略风险
1. 在震荡市场中,频繁出现反转信号,可能导致过度交易和手续费损失。
2. 支撑位和阻力位的判断依赖于回望期的选择,不同的回望期可能导致不同的结果。
3. 反转形态的可靠性并非绝对,虚假信号可能导致亏损。

解决方法:
1. 通过调整反转形态的参数和确认条件,减少虚假信号。
2. 结合其他技术指标或市场情绪指标,提高信号的可靠性。
3. 适当调整止盈和止损水平,以应对不同的市场状况。

#### 策略优化方向
1. 引入交易量指标,确认反转形态的有效性。高交易量的反转形态可能更可靠。
2. 考虑多个时间框架的支撑阻力位,提高支撑阻力位的准确性。
3. 结合趋势指标,如移动平均线,在趋势方向上进行交易,避免逆势交易。
4. 优化止盈和止损水平,根据市场波动性动态调整,以获得更好的风险回报比。

#### 总结
该策略通过识别支撑位和阻力位附近的反转形态,捕捉潜在的交易机会。它简单易用,适用于不同的市场环境。然而,策略的成功依赖于对反转形态和支撑阻力位的准确判断。通过优化交易信号的确认条件,结合其他技术指标,以及动态调整止盈止损水平,可以进一步提高策略的表现。

|| 

#### Overview
This strategy is based on reversal patterns (hammer, engulfing, and doji) and support and resistance levels in technical analysis, trading on a 1-hour chart. The strategy identifies potential market reversal points and executes trades with predefined take profit and stop loss levels.

The main idea of the strategy is to enter a long position when a bullish reversal pattern (such as a hammer, bullish engulfing, or doji) appears near a support level, and to enter a short position when a bearish reversal pattern (such as a hammer, bearish engulfing, or doji) appears near a resistance level. Take profit and stop loss levels are set to control risk and lock in profits.

#### Strategy Logic
1. Calculate the lowest low and highest high within the specified lookback period using ta.lowest() and ta.highest() functions to determine support and resistance levels.
2. Check if the current candlestick forms a hammer, engulfing pattern, or doji.
3. If a bullish reversal pattern appears near a support level, enter a long position; if a bearish reversal pattern appears near a resistance level, enter a short position.
4. Set the take profit price at 3% above the entry price and the stop loss price at 1% below the entry price.
5. Close the position when the price reaches the take profit or stop loss level.

#### Strategy Advantages
1. Combines reversal patterns and key support and resistance levels, improving the reliability of trading signals.
2. Sets clear take profit and stop loss levels, effectively controlling risk.
3. Suitable for both trending and ranging markets, capturing potential reversal opportunities.
4. Simple and easy-to-understand code, facilitating implementation.

#### Strategy Risks
1. In ranging markets, frequent reversal signals may lead to overtrading and commission losses.
2. The identification of support and resistance levels depends on the choice of lookback period, and different lookback periods may lead to different results.
3. The reliability of reversal patterns is not absolute, and false signals may result in losses.

Solutions:
1. Adjust the parameters and confirmation conditions of reversal patterns to reduce false signals.
2. Incorporate other technical indicators or market sentiment indicators to improve signal reliability.
3. Adjust take profit and stop loss levels appropriately to adapt to different market conditions.

#### Strategy Optimization Directions
1. Introduce volume indicators to confirm the validity of reversal patterns. Reversal patterns with high trading volume may be more reliable.
2. Consider support and resistance levels from multiple time frames to improve the accuracy of support and resistance levels.
3. Combine trend indicators, such as moving averages, to trade in the direction of the trend and avoid counter-trend trading.
4. Optimize take profit and stop loss levels by dynamically adjusting them based on market volatility to achieve better risk-reward ratios.

#### Summary
This strategy captures potential trading opportunities by identifying reversal patterns near support and resistance levels. It is simple to use and applicable to different market environments. However, the success of the strategy depends on the accurate identification of reversal patterns and support and resistance levels. By optimizing the confirmation conditions of trading signals, incorporating other technical indicators, and dynamically adjusting take profit and stop loss levels, the performance of the strategy can be further improved.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-05-07 00:00:00
end: 2024-06-06 00:00:00
period: 2h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © Kingcoinmilioner

//@version=5
strategy("Reversal Patterns at Support and Resistance", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// Parameters
support_resistance_lookback = input.int(50, title="Support/Resistance Lookback Period")
reversal_tolerance = input.float(0.01, title="Reversal Tolerance (percent)", step=0.01) / 100
take_profit_percent = input.float(3, title="Take Profit (%)") / 100
stop_loss_percent = input.float(1, title="Stop Loss (%)") / 100

// Functions to identify key support and resistance levels
findSupport() =>
    ta.lowest(low, support_resistance_lookback)

findResistance() =>
    ta.highest(high, support_resistance_lookback)

// Identify reversal patterns
isHammer() =>
    body = math.abs(close - open)
    lowerWick = open > close ? (low < close ? close - low : open - low) : (low < open ? open - low : close - low)
    upperWick = high - math.max(open, close)
    lowerWick > body * 2 and upperWick < body

isEngulfing() =>
    (close[1] < open[1] and close > open and close > open[1] and open < close[1]) 
    (close[1] > open[1] and close < open and close < open[1] and open > close[1])

isDoji() =>
    math.abs(open - close) <= (high - low) * 0.1

// Identify support and resistance levels
support = findSupport()
resistance = findResistance()

// Check for reversal patterns at support and resistance
hammerAtSupport = isHammer() and (low <= support * (1 + reversal_tolerance))
engulfingAtSupport = isEngulfing() and (low <= support * (1 + reversal_tolerance))
dojiAtSupport = isDoji() and (low <= support * (1 + reversal_tolerance))

hammerAtResistance = isHammer() and (high >= resistance * (1 - reversal_tolerance))
engulfingAtResistance = isEngulfing() and (high >= resistance * (1 - reversal_tolerance))
dojiAtResistance = isDoji() and (high >= resistance * (1 - reversal_tolerance))

// Trading logic
if (hammerAtSupport or engulfingAtSupport or dojiAtSupport)
    strategy.entry("Long", strategy.long)
    stop_level = low * (1 - stop_loss_percent)
    take_profit_level = close * (1 + take_profit_percent)
    strategy.exit("Take Profit/Stop Loss", from_entry="Long", stop=stop_level, limit=take_profit_level)

if (hammerAtResistance or engulfingAtResistance or dojiAtResistance)
    strategy.entry("Short", strategy.short)
    stop_level = high * (1 + stop_loss_percent)
    take_profit_level = close * (1 - take_profit_percent)
    strategy.exit("Take Profit/Stop Loss", from_entry="Short", stop=stop_level, limit=take_profit_level)

// Plot support and resistance levels for visualization
plot(support, color=color.green, linewidth=1, title="Support Level")
plot(resistance, color=color.red, linewidth=1, title="Resistance Level")

// Plot reversal patterns on the chart for visualization
plotshape(series=hammerAtSupport, location=location.belowbar, color=color.green, style=shape.labelup, text="Hammer at Support")
plotshape(series=engulfingAtSupport, location=location.belowbar, color=color.green, style=shape.labelup, text="Engulfing at Support")
plotshape(series=dojiAtSupport, location=location.belowbar, color=color.green, style=shape.labelup, text="Doji at Support")

plotshape(series=hammerAtResistance, location=location.abovebar, color=color.red, style=shape.labeldown, text="Hammer at Resistance")
plotshape(series=engulfingAtResistance, location=location.abovebar, color=color.red, style=shape.labeldown, text="Engulfing at Resistance")
plotshape(series=dojiAtResistance, location=location.abovebar, color=color.red, style=shape.labeldown, text="Doji at Resistance")

```

> Detail

https://www.fmz.com/strategy/453668

> Last Modified

2024-06-07 16:45:09
