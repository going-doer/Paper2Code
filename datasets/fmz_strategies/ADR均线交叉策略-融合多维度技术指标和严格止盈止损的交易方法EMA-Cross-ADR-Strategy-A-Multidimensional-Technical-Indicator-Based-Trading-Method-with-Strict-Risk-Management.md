
> Name

ADR均线交叉策略-融合多维度技术指标和严格止盈止损的交易方法EMA-Cross-ADR-Strategy-A-Multidimensional-Technical-Indicator-Based-Trading-Method-with-Strict-Risk-Management

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/10e215963179b73dec5.png)
[trans]

## 概述

ADR均线交叉策略是一个基于TradingView平台的量化交易策略,融合了多个技术指标来判断趋势、过滤信号并设置止盈止损。该策略采用两条不同周期的指数移动平均线(EMA)来识别主要趋势,使用平均真实波幅(ATR)作为波动率过滤器,并根据风险回报比动态设置止盈止损。此外,该策略还引入了交易时间窗口、盈亏平衡、最大日亏损等风控措施,力求在把握趋势机会的同时,严格控制下行风险。

## 策略原理

1. 双均线交叉:策略采用两条不同周期的EMA线来判断趋势。当短期EMA上穿长期EMA时,认为趋势向上,产生做多信号;反之,当短期EMA下穿长期EMA时,认为趋势向下,产生做空信号。

2. ADR波动率过滤:为了避免在波动率较低的环境下产生交易信号,策略引入了ADR指标作为波动率过滤器。只有当ADR值高于预设的最小阈值时,才允许开仓。

3. 交易时间窗口:该策略允许用户设置每日交易的起止时间。只有在指定的时间窗口内,才会执行交易。这有助于避开流动性较差或波动性较大的时段。

4. 动态止盈止损:策略根据最近N根K线的平均最高价和最低价,并结合预设的风险回报比,动态计算止盈价和止损价。这确保了每笔交易的风险回报是可控的。

5. 盈亏平衡:当持仓达到一定的盈利幅度后(用户可设置风险回报比),策略会将止损位上移至开仓价格,即盈亏平衡位。这有助于保护已获得的利润。

6. 最大日亏损限制:为了控制单日最大亏损,策略设置了每日亏损限额。一旦当日亏损达到该限额,策略将停止交易,直到第二天开盘。

7. 收盘平仓:无论持仓是否触及止盈或止损线,策略都会在每个交易日的固定时间(如16:00)平掉所有仓位,避免隔夜风险。

## 优势分析

1. 趋势跟踪能力强:通过双均线交叉来判断趋势,可以有效捕捉到市场的主要趋势,从而提高策略的胜率和盈利潜力。

2. 波动率适应性好:引入ADR指标作为波动率过滤器,可以避免在波动率较低的环境下频繁交易,减少了无效信号和虚假突破带来的损失。

3. 风险控制严格:该策略从多个维度设置了风控措施,包括动态止盈止损、盈亏平衡、最大日亏损限制等,有效控制了策略的下行风险,提高了风险调整后收益。

4. 参数灵活可调:策略的各项参数,如均线周期、ADR的长度、风险回报比、交易时间窗口等,都可以根据用户的偏好和市场特点进行灵活设置,从而优化策略表现。

5. 自动化程度高:该策略基于TradingView平台,交易逻辑完全由程序自动执行,减少了人为情绪和主观判断的干扰,有利于战略的长期稳定运行。

## 风险分析

1. 参数优化风险:尽管该策略的参数可以灵活调整,但如果优化过度,可能会导致过度拟合,在样本外表现不佳。因此,在参数设置时,需要进行充分的回测和分析,以确保策略的稳健性。

2. 突发事件风险:该策略主要基于技术指标交易,对于一些突发的重大基本面事件,如政策变动、经济数据大幅波动等,可能反应不足,导致较大的回撤。

3. 趋势转折风险:在趋势转折的关键时期,双均线交叉信号可能会出现延迟,导致策略错过最佳的建仓时机,或者在趋势反转初期遭受损失。

4. 流动性风险:尽管策略设置了交易时间窗口,但如果交易的标的流动性较差,仍可能面临滑点、交易延迟等风险,影响策略表现。

5. 技术指标失效风险:该策略高度依赖于技术指标,如果市场环境发生重大变化,导致指标失去原有的指示意义,策略的有效性可能会下降。

## 优化方向

1. 引入更多维度的指标:在现有双均线和ADR的基础上,可以考虑引入更多有效的技术指标,如MACD、RSI等,以提高信号的可靠性和稳健性。

2. 动态优化参数:可以建立一套参数优化的机制,根据不同的市场状态(如趋势型、震荡型等),动态调整策略的关键参数,以适应市场的变化。

3. 加入基本面因素:适当考虑一些重要的基本面指标,如经济数据、政策风向等,可以帮助策略更好地把握市场趋势,并及时规避系统性风险。

4. 改进止盈止损机制:在现有的动态止盈止损基础上,可以进一步优化止盈止损的逻辑,如引入追踪止损、部分止盈等方法,以更好地保护利润和控制风险。

5. 多标的、多时间周期:将该策略扩展到多个交易标的和多个时间周期上,通过分散投资和时间周期的优化,提高策略的适应性和稳定性。

## 总结

ADR均线交叉策略是一个基于技术分析的量化交易策略,通过双均线交叉来判断趋势,并使用ADR指标进行波动率过滤。该策略还设置了严格的风控措施,包括动态止盈止损、盈亏平衡、最大日亏损限制等,以控制下行风险。策略的优势在于趋势跟踪能力强、波动率适应性好、风险控制严格、参数灵活可调、自动化程度高。但同时也存在一些风险,如参数优化风险、突发事件风险、趋势转折风险、流动性风险和技术指标失效风险。未来,该策略可以考虑从引入更多维度指标、动态优化参数、加入基本面因素、改进止盈止损机制、扩展到多标的多周期等方面进行优化和改进,以进一步提升策略的稳健性和盈利能力。总的来说,ADR均线交叉策略为量化交易者提供了一个可供参考的交易模型,但在实际应用中还需要根据自己的风险偏好和交易风格进行适当的调整和优化。

|| 

## Overview

The EMA Cross ADR Strategy is a quantitative trading strategy based on the TradingView platform. It combines multiple technical indicators to determine trends, filter signals, and set stop-loss and take-profit levels. The strategy employs two Exponential Moving Averages (EMAs) with different periods to identify the main trend, uses the Average Daily Range (ADR) as a volatility filter, and dynamically sets stop-loss and take-profit levels based on a risk-reward ratio. In addition, the strategy incorporates risk management measures such as a trading time window, break-even stops, and a maximum daily loss limit, aiming to capture trend opportunities while strictly controlling downside risk.

## Strategy Principles

1. Dual EMA Crossover: The strategy uses two EMAs with different periods to determine the trend. When the short-term EMA crosses above the long-term EMA, it is considered an uptrend, generating a long signal; conversely, when the short-term EMA crosses below the long-term EMA, it is considered a downtrend, generating a short signal.

2. ADR Volatility Filter: To avoid generating trading signals in low volatility environments, the strategy introduces the ADR indicator as a volatility filter. Positions are only allowed to be opened when the ADR value is above a pre-set minimum threshold.

3. Trading Time Window: The strategy allows users to set the start and end times for daily trading. Trades are only executed within the specified time window, which helps avoid illiquid or highly volatile periods.

4. Dynamic Stop-Loss and Take-Profit: The strategy dynamically calculates the stop-loss and take-profit prices based on the average highest and lowest prices of the most recent N candlesticks, combined with a pre-set risk-reward ratio. This ensures that the risk-reward of each trade is controllable.

5. Break-Even Stops: When a position reaches a certain profit level (user-defined risk-reward ratio), the strategy moves the stop-loss to the break-even point (entry price). This helps protect profits that have already been earned.

6. Maximum Daily Loss Limit: To control the maximum loss per day, the strategy sets a daily loss limit. Once the daily loss reaches this limit, the strategy stops trading until the next day's opening.

7. Close All Positions at End of Day: Regardless of whether positions have hit the take-profit or stop-loss levels, the strategy closes all positions at a fixed time each trading day (e.g., 16:00) to avoid overnight risk.

## Advantage Analysis

1. Strong Trend-Following Ability: By using dual EMA crossovers to determine trends, the strategy can effectively capture the main market trends, thereby improving the win rate and profit potential.

2. Good Volatility Adaptability: The introduction of the ADR indicator as a volatility filter can avoid frequent trading in low volatility environments, reducing losses caused by invalid signals and false breakouts.

3. Strict Risk Control: The strategy sets risk control measures from multiple dimensions, including dynamic stop-loss and take-profit, break-even stops, and maximum daily loss limits, effectively controlling downside risk and improving risk-adjusted returns.

4. Flexible Parameter Settings: The various parameters of the strategy, such as EMA periods, ADR length, risk-reward ratio, trading time window, etc., can be flexibly set according to user preferences and market characteristics to optimize strategy performance.

5. High Degree of Automation: The strategy is based on the TradingView platform, and the trading logic is executed entirely by the program, reducing the interference of human emotions and subjective judgments, which is conducive to the long-term stable operation of the strategy.

## Risk Analysis

1. Parameter Optimization Risk: Although the parameters of the strategy can be flexibly adjusted, over-optimization may lead to overfitting and poor out-of-sample performance. Therefore, when setting parameters, sufficient backtesting and analysis are required to ensure the robustness of the strategy.

2. Sudden Event Risk: The strategy mainly trades based on technical indicators and may not react sufficiently to some sudden major fundamental events, such as policy changes or significant economic data fluctuations, leading to large drawdowns.

3. Trend Reversal Risk: During key periods of trend reversals, dual EMA crossover signals may be delayed, causing the strategy to miss the best timing for establishing positions or suffer losses at the beginning of a trend reversal.

4. Liquidity Risk: Although the strategy sets a trading time window, if the liquidity of the traded instruments is poor, it may still face risks such as slippage and trading delays, affecting strategy performance.

5. Technical Indicator Failure Risk: The strategy relies heavily on technical indicators. If market conditions change significantly, causing the indicators to lose their original meaning, the effectiveness of the strategy may decline.

## Optimization Directions

1. Introduce More Dimensional Indicators: On the basis of the existing dual EMAs and ADR, consider introducing more effective technical indicators, such as MACD and RSI, to improve the reliability and robustness of signals.

2. Dynamic Parameter Optimization: Establish a mechanism for parameter optimization that dynamically adjusts key strategy parameters based on different market states (such as trending or oscillating) to adapt to market changes.

3. Incorporate Fundamental Factors: Give appropriate consideration to some important fundamental indicators, such as economic data and policy directions, which can help the strategy better grasp market trends and avoid systemic risks in a timely manner.

4. Improve Stop-Loss and Take-Profit Mechanisms: Further optimize the logic of stop-loss and take-profit on the basis of the existing dynamic stop-loss and take-profit, such as introducing trailing stops and partial take-profits, to better protect profits and control risks.

5. Multiple Instruments and Time Frames: Extend the strategy to multiple trading instruments and time frames, and improve the adaptability and stability of the strategy through diversified investment and time frame optimization.

## Summary

The EMA Cross ADR Strategy is a quantitative trading strategy based on technical analysis. It determines trends through dual EMA crossovers and uses the ADR indicator for volatility filtering. The strategy also sets strict risk control measures, including dynamic stop-loss and take-profit, break-even stops, and maximum daily loss limits to control downside risk. The advantages of the strategy lie in its strong trend-following ability, good volatility adaptability, strict risk control, flexible parameter settings, and high degree of automation. However, it also has some risks, such as parameter optimization risk, sudden event risk, trend reversal risk, liquidity risk, and technical indicator failure risk. In the future, the strategy can consider optimizing and improving from aspects such as introducing more dimensional indicators, dynamic parameter optimization, incorporating fundamental factors, improving stop-loss and take-profit mechanisms, and extending to multiple instruments and time frames to further enhance the robustness and profitability of the strategy. Overall, the EMA Cross ADR Strategy provides quantitative traders with a trading model for reference, but in practical application, it needs to be appropriately adjusted and optimized according to one's own risk preferences and trading style.

[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_1|10|Short EMA Length|
|v_input_2|50|Long EMA Length|
|v_input_3|14|ADR Length|
|v_input_4|2|Risk/Reward Ratio|
|v_input_5|10|Lookback Candles for Stop Loss|
|v_input_6|900|Start Time|
|v_input_7|1600|End Time|
|v_input_8|10|Minimum ADR Value for Entry|
|v_input_float_1|true|Break-Even Profit|
|v_input_float_2|true|Break-Even Risk-Reward Ratio|
|v_input_9|-2000|Daily Loss Limit|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-02-26 00:00:00
end: 2024-03-27 00:00:00
period: 2h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © Sameh_Hussein

//@version=5
strategy('EMA Cross ADR Strategy with Stats', overlay=true)

// Adjustable Parameters
shortEmaLength = input(10, title='Short EMA Length')
longEmaLength = input(50, title='Long EMA Length')
adrLength = input(14, title='ADR Length')
riskRewardRatio = input(2.0, title='Risk/Reward Ratio')
lookbackCandles = input(10, title='Lookback Candles for Stop Loss')
startTime = input(0900, title='Start Time')
endTime = input(1600, title='End Time')
minAdrValue = input(10, title='Minimum ADR Value for Entry')
breakEvenProfit = input.float(1.0, title='Break-Even Profit', minval=0.0)
breakEvenRR = input.float(1.0, title='Break-Even Risk-Reward Ratio', minval=0.0)
dailyLossLimit = input(-2000.0, title='Daily Loss Limit')

// Exponential Moving Averages
shortEma = ta.ema(close, shortEmaLength)
longEma = ta.ema(close, longEmaLength)

// Average Daily Range
adr = ta.sma(ta.tr, adrLength)

// Time Filter Function
timeFilter() => true

// Entry Conditions with ADR filter
longCondition = ta.crossover(shortEma, longEma) and timeFilter() and adr > minAdrValue
shortCondition = ta.crossunder(shortEma, longEma) and timeFilter() and adr > minAdrValue

// Calculate the average low and average high of the previous 'lookbackCandles' candles
averageLow = ta.sma(low, lookbackCandles)
averageHigh = ta.sma(high, lookbackCandles)

// Risk and Reward Calculation
stopLossLong = averageLow
takeProfitLong = close + (close - averageLow) * riskRewardRatio
stopLossShort = averageHigh
takeProfitShort = close - (averageHigh - close) * riskRewardRatio

// Entry Control Variables
var longEntryAllowed = true
var shortEntryAllowed = true

// Update entry price on trade execution
var float entryPriceLong = na
var float entryPriceShort = na

if (strategy.position_size > 0)
    if (strategy.position_size[1] <= 0)
        entryPriceLong := strategy.opentrades.entry_price(strategy.opentrades - 1)
    else
        entryPriceLong := entryPriceLong
else
    entryPriceLong := na

if (strategy.position_size < 0)
    if (strategy.position_size[1] >= 0)
        entryPriceShort := strategy.opentrades.entry_price(strategy.opentrades - 1)
    else
        entryPriceShort := entryPriceShort
else
    entryPriceShort := na

// Adjust stop loss to break-even plus the defined profit when the specified risk-reward ratio is reached
breakEvenTriggerLong = entryPriceLong + (entryPriceLong - stopLossLong) * breakEvenRR
breakEvenTriggerShort = entryPriceShort - (stopLossShort - entryPriceShort) * breakEvenRR

if (longEntryAllowed and close >= breakEvenTriggerLong)
    stopLossLong := entryPriceLong + breakEvenProfit

if (shortEntryAllowed and close <= breakEvenTriggerShort)
    stopLossShort := entryPriceShort - breakEvenProfit

// Close all trades at 1600
if (hour == 15 and minute == 59)
    strategy.close_all(comment='Close at 1600')

// Define the daily loss variable and last trade day
var float[] dailyLossArray = array.new_float(1, 0.0)
var int[] lastTradeDayArray = array.new_int(1, na)

// Function to update the daily loss
updateDailyLoss() =>
    _dailyLoss = array.get(dailyLossArray, 0)
    _lastTradeDay = array.get(lastTradeDayArray, 0)
    if na(_lastTradeDay) or dayofmonth != _lastTradeDay
        _dailyLoss := 0.0
        array.set(lastTradeDayArray, 0, dayofmonth)
    if not na(strategy.closedtrades.entry_bar_index(strategy.closedtrades - 1))
        _dailyLoss += strategy.closedtrades.profit(strategy.closedtrades - 1)
    array.set(dailyLossArray, 0, _dailyLoss)

// Call the function to update the daily loss
updateDailyLoss()

// Execute Strategy
if longCondition and longEntryAllowed
    strategy.entry('Long', strategy.long)
    strategy.exit('Take Profit/Stop Loss', 'Long', stop=stopLossLong, limit=takeProfitLong)
    longEntryAllowed := false

if shortCondition and shortEntryAllowed
    strategy.entry('Short', strategy.short)
    strategy.exit('Take Profit/Stop Loss', 'Short', stop=stopLossShort, limit=takeProfitShort)
    shortEntryAllowed := false

// Reset entry control variables on position close
if strategy.position_size == 0
    longEntryAllowed := true
    shortEntryAllowed := true

// // Statistics
// winRate = strategy.wintrades / strategy.closedtrades * 100
// totalTrades = strategy.closedtrades
// averageProfit = strategy.grossprofit / strategy.wintrades
// averageLoss = strategy.grossloss / strategy.losstrades

// // Plotting
// plot(shortEma, color=color.new(color.red, 0), title='Short EMA')
// plot(longEma, color=color.new(color.blue, 0), title='Long EMA')

// // Display Table
// table statsTable = table.new(position=position.top_right, columns=2, rows=4, bgcolor=color.gray, border_width=1)
// table.cell(statsTable, column=0, row=0, text='Win Rate (%)', bgcolor=color.blue)
// table.cell(statsTable, column=1, row=0, text=str.tostring(winRate), bgcolor=color.blue)
// table.cell(statsTable, column=0, row=1, text='Total Trades', bgcolor=color.blue)
// table.cell(statsTable, column=1, row=1, text=str.tostring(totalTrades), bgcolor=color.blue)
// table.cell(statsTable, column=0, row=2, text='Average Profit', bgcolor=color.blue)
// table.cell(statsTable, column=1, row=2, text=str.tostring(averageProfit), bgcolor=color.blue)
// table.cell(statsTable, column=0, row=3, text='Average Loss', bgcolor=color.blue)
// table.cell(statsTable, column=1, row=3, text=str.tostring(averageLoss), bgcolor=color.blue)

```

> Detail

https://www.fmz.com/strategy/446441

> Last Modified

2024-03-28 16:46:29
