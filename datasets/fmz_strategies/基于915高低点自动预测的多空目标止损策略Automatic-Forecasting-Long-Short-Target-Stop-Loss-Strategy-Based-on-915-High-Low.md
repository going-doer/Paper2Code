
> Name

基于915高低点自动预测的多空目标止损策略Automatic-Forecasting-Long-Short-Target-Stop-Loss-Strategy-Based-on-915-High-Low

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/177c2d01e2d3c06a88d.png)
[trans]

## 概述

该策略基于9:15分钟K线的高低点,自动计算多空方向的目标价位和止损价位。通过RSI指标判断当前市场的超买超卖情况,在价格突破9:15高低点且RSI满足条件时进行开仓操作。该策略能够自动预测多空方向的目标价位和止损价位,简化了交易者的操作流程。

## 策略原理

1. 确定9:15分钟K线的高低点,分别作为多空方向的关键价位。
2. 多头方向:目标价位为9:15高点+200个点,止损价位为9:15低点。
3. 空头方向:目标价位为9:15低点-200个点,止损价位为9:15高点。
4. 计算RSI指标,默认参数为14,超买线为60,超卖线为40。
5. 多头开仓条件:收盘价突破9:15高点,且RSI大于超买线。
6. 空头开仓条件:收盘价突破9:15低点,且RSI小于超卖线。
7. 当开仓条件满足时,执行相应的多头或空头开仓操作。
8. 在图表上绘制9:15高低点,多空目标价位和止损价位,以及开仓信号。

该策略利用9:15分钟K线的高低点作为关键价位,自动计算出多空方向的目标和止损,简化了交易者的操作。同时,引入了RSI指标作为过滤条件,在一定程度上可以避免频繁开仓和假突破。

## 优势分析

1. 自动计算多空目标和止损:该策略基于9:15分钟K线的高低点,自动计算出多空方向的目标价位和止损价位。交易者无需手动设置,简化了操作流程,提高了交易效率。

2. RSI指标过滤:策略引入了RSI指标作为开仓的过滤条件。在价格突破关键位置时,还需要RSI指标达到超买或超卖状态,才会触发开仓信号。这在一定程度上可以帮助交易者避免频繁交易和假突破陷阱。

3. 直观的图表展示:该策略在图表上绘制了9:15高低点,多空目标价位,止损价位以及开仓信号。交易者可以直观地看到关键价位和交易信号,便于做出交易决策。

4. 适用于短线交易:该策略以9:15分钟的高低点为基础,目标价位和止损价位的设置也相对较近。因此,该策略更适合短线交易操作,可以快速进出,博取短期内的价格波动。

## 风险分析

1. 盘中波动风险:该策略以9:15分钟K线的高低点为关键位置,但盘中价格可能出现大幅波动。如果价格在触发开仓后迅速反转,可能导致交易者的损失超出预期。

2. 止损位置风险:策略中的止损位置是固定的,即多头止损位置为9:15低点,空头止损位置为9:15高点。如果价格突破了9:15高低点后继续大幅运行,固定的止损位置可能导致较大的损失。

3. RSI指标参数风险:该策略使用了默认的RSI参数,即长度为14,超买线为60,超卖线为40。但在不同的市场环境和标的中,这些参数可能并不适用。固定的参数设置可能影响策略的有效性。

4. 盈亏比风险:策略中固定的目标价位和止损价位,决定了每次交易的盈亏比。如果盈亏比设置不当,可能导致长期内策略的收益性不佳。

解决方法:
1. 对于盘中波动风险,可以考虑引入更多的过滤条件,如加入交易量指标,或者缩小止损范围。
2. 对于止损位置风险,可以考虑使用追踪止损或者条件止损,根据市场情况动态调整止损位置。
3. 对于RSI指标参数风险,可以对不同市场和标的进行参数优化,找到更适合的参数组合。
4. 对于盈亏比风险,可以根据历史数据测试不同的目标价位和止损价位组合,找到更优的盈亏比设置。

## 优化方向

1. 动态止损:目前策略使用固定的止损位置,可以考虑引入动态止损机制,如追踪止损或条件止损。这样可以在价格出现超预期波动时,及时控制风险。

2. 引入更多过滤条件:策略目前主要依赖价格突破和RSI指标,可以考虑引入更多过滤条件,如交易量指标,波动率指标等。通过多个条件的共同确认,可以提高开仓信号的有效性。

3. 参数优化:对于RSI指标的参数设置,可以针对不同的市场和标的进行优化。通过对历史数据的测试,找到更适合当前交易标的的参数组合,提高策略的稳定性。

4. 盈亏比优化:策略的盈亏比对长期收益有重要影响。可以通过对历史数据的回测,测试不同的目标价位和止损价位组合,找到能够带来更高收益的盈亏比设置。

5. 加入趋势判断:该策略目前主要依赖于盘中高低点的突破,属于逆势交易。可以考虑加入趋势判断,在大趋势方向上进行交易,提高胜率和盈亏比。

## 总结

该策略基于9:15分钟K线的高低点,自动计算多空目标价位和止损价位,同时使用RSI指标作为过滤条件,简化了交易者的操作流程。策略的优势在于自动化程度高,直观易用,适合短线交易操作。但同时也存在一些风险,如盘中波动风险,止损位置风险,指标参数风险和盈亏比风险等。针对这些风险,可以从动态止损,引入更多过滤条件,参数优化,盈亏比优化和趋势判断等方面进行策略改进。通过不断的优化和改进,可以提高该策略的稳定性和盈利能力,更好地适应不同的市场环境。

|| 

## Overview

This strategy automatically calculates long and short target prices and stop loss levels based on the high and low of the 9:15 minute candle. It uses the RSI indicator to determine the current overbought or oversold state of the market and triggers a long or short entry when the price breaks the 9:15 high/low and the RSI condition is met. The strategy simplifies the trading process by automatically predicting target prices and stop loss levels for long and short directions.

## Strategy Principle

1. Determine the high and low of the 9:15 minute candle as key levels for long and short directions.
2. Long direction: target price is 9:15 high + 200 points, stop loss is 9:15 low.
3. Short direction: target price is 9:15 low - 200 points, stop loss is 9:15 high.
4. Calculate the RSI indicator with default parameters of 14, overbought line at 60, and oversold line at 40.
5. Long entry condition: close price breaks above 9:15 high and RSI is greater than the overbought line.
6. Short entry condition: close price breaks below 9:15 low and RSI is less than the oversold line.
7. Execute the corresponding long or short entry when the entry conditions are met.
8. Plot the 9:15 high/low, long/short target prices, stop loss levels, and entry signals on the chart.

The strategy utilizes the high and low of the 9:15 minute candle as key levels and automatically calculates the target prices and stop losses for long and short directions, simplifying the trader's operation. Additionally, it introduces the RSI indicator as a filter condition, which can help avoid frequent entries and false breakouts to a certain extent.

## Advantage Analysis

1. Automatic calculation of long/short targets and stop losses: The strategy automatically calculates the target prices and stop loss levels for long and short directions based on the 9:15 high/low. Traders do not need to set them manually, simplifying the operation process and improving trading efficiency.

2. RSI indicator filter: The strategy introduces the RSI indicator as a filter condition for entry. When the price breaks a key level, the RSI needs to reach the overbought or oversold state to trigger an entry signal. This can help traders avoid frequent trading and false breakout traps to a certain extent.

3. Intuitive chart display: The strategy plots the 9:15 high/low, long/short target prices, stop loss levels, and entry signals on the chart. Traders can intuitively see the key levels and trading signals, facilitating their decision-making.

4. Suitable for short-term trading: The strategy is based on the high and low of the 9:15 minute candle, and the target prices and stop losses are set relatively close. Therefore, it is more suitable for short-term trading operations, allowing for quick entries and exits to capture short-term price movements.

## Risk Analysis

1. Intraday volatility risk: The strategy uses the 9:15 high/low as key levels, but prices may experience significant fluctuations during the trading day. If the price quickly reverses after triggering an entry, it may cause the trader's loss to exceed expectations.

2. Stop loss level risk: The stop loss levels in the strategy are fixed, with the long stop loss at the 9:15 low and the short stop loss at the 9:15 high. If the price continues to move significantly after breaking the 9:15 high/low, the fixed stop loss levels may result in larger losses.

3. RSI indicator parameter risk: The strategy uses default RSI parameters, with a length of 14, overbought line at 60, and oversold line at 40. However, these parameters may not be suitable for different market environments and instruments. Fixed parameter settings may affect the effectiveness of the strategy.

4. Risk-reward ratio risk: The fixed target prices and stop loss levels in the strategy determine the risk-reward ratio of each trade. If the risk-reward ratio is not set appropriately, it may lead to poor long-term profitability of the strategy.

Solutions:
1. For intraday volatility risk, consider introducing more filter conditions, such as volume indicators or narrowing the stop loss range.
2. For stop loss level risk, consider using trailing stop losses or conditional stop losses to dynamically adjust the stop loss levels based on market conditions.
3. For RSI indicator parameter risk, optimize the parameters for different markets and instruments to find more suitable combinations.
4. For risk-reward ratio risk, test different target price and stop loss combinations based on historical data to find more optimal risk-reward ratio settings.

## Optimization Directions

1. Dynamic stop loss: The current strategy uses fixed stop loss levels. Consider introducing dynamic stop loss mechanisms, such as trailing stop losses or conditional stop losses. This allows for timely risk control when prices experience unexpected volatility.

2. Introducing more filter conditions: The strategy currently relies mainly on price breakouts and the RSI indicator. Consider adding more filter conditions, such as volume indicators or volatility indicators. By confirming entry signals through multiple conditions, the effectiveness of the signals can be improved.

3. Parameter optimization: Optimize the RSI indicator parameters for different markets and instruments. By testing historical data, find parameter combinations that are more suitable for the current trading instrument to improve the stability of the strategy.

4. Risk-reward ratio optimization: The risk-reward ratio has a significant impact on long-term profitability. By backtesting historical data, test different target price and stop loss combinations to find risk-reward ratio settings that can generate higher returns.

5. Incorporating trend analysis: The current strategy mainly relies on intraday high/low breakouts, which is a counter-trend approach. Consider incorporating trend analysis to trade in the direction of the larger trend, improving the win rate and risk-reward ratio.

## Conclusion

This strategy automatically calculates long and short target prices and stop loss levels based on the 9:15 high/low, while using the RSI indicator as a filter condition, simplifying the trader's operation process. The advantages of the strategy lie in its high degree of automation, intuitive usability, and suitability for short-term trading operations. However, it also involves certain risks, such as intraday volatility risk, stop loss level risk, indicator parameter risk, and risk-reward ratio risk. To address these risks, the strategy can be improved through dynamic stop losses, introducing more filter conditions, parameter optimization, risk-reward ratio optimization, and trend analysis. By continuously optimizing and improving the strategy, its stability and profitability can be enhanced to better adapt to different market environments.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_1|true|Show Signals|
|v_input_2|9|Session Start Hour|
|v_input_3|false|Session Start Minute|
|v_input_4|9|Session End Hour|
|v_input_5|15|Session End Minute|
|v_input_6|14|RSI Length|
|v_input_7|60|Overbought Level|
|v_input_8|40|Oversold Level|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-02-01 00:00:00
end: 2024-02-29 23:59:59
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("9:15 AM High/Low with Automatic Forecasting", overlay=true)

// Parameters
showSignals = input(true, title="Show Signals")

// Define session time
sessionStartHour = input(9, title="Session Start Hour")
sessionStartMinute = input(0, title="Session Start Minute")
sessionEndHour = input(9, title="Session End Hour")
sessionEndMinute = input(15, title="Session End Minute")

// Calculate session high and low
var float sessionHigh = na
var float sessionLow = na
if (hour == sessionStartHour and minute == sessionStartMinute)
    sessionHigh := high
    sessionLow := low

// Update session high and low if within session time
if (hour == sessionStartHour and minute >= sessionStartMinute and minute < sessionEndMinute)
    sessionHigh := high > sessionHigh or na(sessionHigh) ? high : sessionHigh
    sessionLow := low < sessionLow or na(sessionLow) ? low : sessionLow

// Plot horizontal lines for session high and low
plot(sessionHigh, color=color.green, title="9:00 AM High", style=plot.style_stepline, linewidth=1)
plot(sessionLow, color=color.red, title="9:00 AM Low", style=plot.style_stepline, linewidth=1)

// Calculate targets and stop loss
longTarget = sessionHigh + 200
longStopLoss = sessionLow
shortTarget = sessionLow - 200
shortStopLoss = sessionHigh

// Plot targets and stop loss
plot(longTarget, color=color.blue, title="Long Target", style=plot.style_cross, linewidth=1)
plot(longStopLoss, color=color.red, title="Long Stop Loss", style=plot.style_cross, linewidth=1)
plot(shortTarget, color=color.blue, title="Short Target", style=plot.style_cross, linewidth=1)
plot(shortStopLoss, color=color.red, title="Short Stop Loss", style=plot.style_cross, linewidth=1)

// RSI
rsiLength = input(14, title="RSI Length")
overboughtLevel = input(60, title="Overbought Level")
oversoldLevel = input(40, title="Oversold Level")
rsi = ta.rsi(close, rsiLength)

// Entry conditions
longCondition = close > sessionHigh and rsi > overboughtLevel
shortCondition = close < sessionLow and rsi < oversoldLevel

// Long entry
if (showSignals and longCondition)
    strategy.entry("Long", strategy.long)

// Short entry
if (showSignals and shortCondition)
    strategy.entry("Short", strategy.short)

```

> Detail

https://www.fmz.com/strategy/445477

> Last Modified

2024-03-19 18:37:37
