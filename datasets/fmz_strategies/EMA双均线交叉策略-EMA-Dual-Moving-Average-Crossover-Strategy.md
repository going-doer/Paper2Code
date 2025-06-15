
> Name

EMA双均线交叉策略-EMA-Dual-Moving-Average-Crossover-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/10ec51b33944b7bdd21.png)
[trans]
#### 概述
该策略使用两条指数移动平均线(EMA)来捕捉价格趋势的变化。当短期EMA从下方向上穿越长期EMA时,产生买入信号;当短期EMA从上方向下穿越长期EMA时,产生卖出信号。该策略同时设置了每日止损和止盈限制,以控制单日亏损和盈利。

#### 策略原理
1. 计算短期EMA(默认周期为9)和长期EMA(默认周期为21)。
2. 当短期EMA向上穿越长期EMA时,开仓做多;当短期EMA向下穿越长期EMA时,开仓做空。
3. 记录每个交易日开始时的账户权益,并计算当前账户权益与之的差值,即当日盈亏。
4. 如果当日亏损超过最大允许亏损(账户初始资金的0.25%),平掉所有仓位。
5. 如果当日盈利超过最大允许盈利(账户初始资金的2%),平掉所有仓位。

#### 策略优势
1. 简单易懂:该策略逻辑清晰,仅使用两条移动平均线即可产生交易信号,易于理解和实现。
2. 趋势跟踪:通过EMA快慢线的交叉,能够较好地捕捉价格趋势的变化,适合在趋势市场中使用。
3. 风险控制:设置了每日止损和止盈限制,可以有效控制单日亏损和盈利,防止账户出现过大波动。

#### 策略风险
1. 参数优化:该策略的表现很大程度上取决于EMA周期的选择,不同的参数设置可能导致截然不同的结果。因此,需要在不同市场环境下进行参数优化和回测。
2. 震荡市:在震荡市场中,价格频繁在EMA上下波动,可能会产生较多的虚假信号,导致频繁交易和资金损耗。
3. 趋势转折:当市场趋势发生转折时,该策略可能会延迟入场或出场,错过最佳交易时机。

#### 策略优化方向
1. 引入其他技术指标,如RSI、MACD等,以辅助判断趋势强度和方向,提高信号准确性。
2. 优化止损和止盈规则,如采用移动止损或动态止盈,以更好地保护利润和控制风险。
3. 根据市场波动性动态调整EMA周期,以适应不同的市场状态。
4. 结合基本面分析,如经济数据、重大事件等,对交易信号进行过滤和确认。

#### 总结
EMA双均线交叉策略是一个简单易懂、适合趋势市场的交易策略。通过快慢均线的交叉,可以较好地捕捉价格趋势的变化。同时,每日止损和止盈的设置可以有效控制风险。但该策略在震荡市或趋势转折时可能表现欠佳,需要结合其他技术指标和分析方法进行优化和改进。

|| 

#### Overview
This strategy uses two exponential moving averages (EMAs) to capture changes in price trends. When the short-term EMA crosses above the long-term EMA from below, a buy signal is generated; when the short-term EMA crosses below the long-term EMA from above, a sell signal is generated. The strategy also sets daily stop-loss and take-profit limits to control single-day losses and profits.

#### Strategy Principles
1. Calculate the short-term EMA (default period of 9) and long-term EMA (default period of 21).
2. When the short-term EMA crosses above the long-term EMA, open a long position; when the short-term EMA crosses below the long-term EMA, open a short position.
3. Record the account equity at the start of each trading day and calculate the difference between the current account equity and the starting equity, i.e., the daily profit and loss.
4. If the daily loss exceeds the maximum allowed loss (0.25% of the initial account funds), close all positions.
5. If the daily profit exceeds the maximum allowed profit (2% of the initial account funds), close all positions.

#### Strategy Advantages
1. Simple and easy to understand: The strategy logic is clear and uses only two moving averages to generate trading signals, making it easy to understand and implement.
2. Trend following: By using the crossover of fast and slow EMAs, the strategy can capture changes in price trends relatively well, making it suitable for use in trending markets.
3. Risk control: The daily stop-loss and take-profit limits can effectively control single-day losses and profits, preventing excessive fluctuations in the account.

#### Strategy Risks
1. Parameter optimization: The performance of the strategy largely depends on the choice of EMA periods, and different parameter settings may lead to drastically different results. Therefore, parameter optimization and backtesting need to be performed in different market environments.
2. Choppy markets: In choppy markets, prices frequently fluctuate above and below the EMAs, potentially generating many false signals and leading to frequent trades and capital erosion.
3. Trend reversals: When market trends reverse, the strategy may delay entry or exit, missing the best trading opportunities.

#### Strategy Optimization Directions
1. Introduce other technical indicators such as RSI and MACD to help judge trend strength and direction and improve signal accuracy.
2. Optimize stop-loss and take-profit rules, such as using trailing stops or dynamic take-profit levels, to better protect profits and control risks.
3. Dynamically adjust EMA periods based on market volatility to adapt to different market states.
4. Combine fundamental analysis, such as economic data and major events, to filter and confirm trading signals.

#### Summary
The EMA dual moving average crossover strategy is a simple, easy-to-understand trading strategy suitable for trending markets. By using the crossover of fast and slow moving averages, it can capture changes in price trends relatively well. At the same time, the daily stop-loss and take-profit settings can effectively control risks. However, the strategy may underperform in choppy markets or during trend reversals and needs to be optimized and improved by combining other technical indicators and analysis methods.

[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-06-01 00:00:00
end: 2024-06-06 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © DD173838

//@version=5
strategy("Moving Average Strategy with Daily Limits", overlay=true)

// Moving Average settings
shortMaLength = input.int(9, title="Short MA Length")
longMaLength = input.int(21, title="Long MA Length")

// Calculate MAs
shortMa = ta.ema(close, shortMaLength)
longMa = ta.ema(close, longMaLength)

// Plot MAs
plot(shortMa, title="9 EMA", color=color.blue)
plot(longMa, title="21 EMA", color=color.red)

// Strategy conditions
crossUp = ta.crossover(shortMa, longMa)
crossDown = ta.crossunder(shortMa, longMa)

// Debug plots to check cross conditions
plotshape(series=crossUp, title="Cross Up", location=location.belowbar, color=color.green, style=shape.labelup, text="UP")
plotshape(series=crossDown, title="Cross Down", location=location.abovebar, color=color.red, style=shape.labeldown, text="DOWN")

// Entry at cross signals
if (crossUp)
    strategy.entry("Long", strategy.long)

if (crossDown)
    strategy.entry("Short", strategy.short)

// Daily drawdown and profit limits
var float startOfDayEquity = na
if (na(startOfDayEquity) or ta.change(time('D')) != 0)
    startOfDayEquity := strategy.equity

maxDailyLoss = 50000 * 0.0025
maxDailyProfit = 50000 * 0.02
currentDailyPL = strategy.equity - startOfDayEquity

if (currentDailyPL <= -maxDailyLoss)
    strategy.close_all(comment="Max Daily Loss Reached")

if (currentDailyPL >= maxDailyProfit)
    strategy.close_all(comment="Max Daily Profit Reached")

```

> Detail

https://www.fmz.com/strategy/453662

> Last Modified

2024-06-07 15:58:15
