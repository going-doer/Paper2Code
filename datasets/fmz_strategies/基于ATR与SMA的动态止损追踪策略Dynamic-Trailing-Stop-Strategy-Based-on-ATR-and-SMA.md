
> Name

基于ATR与SMA的动态止损追踪策略Dynamic-Trailing-Stop-Strategy-Based-on-ATR-and-SMA

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/f1c5134f002b90bb04.png)
[trans]
## 概述

该策略结合了ATR(Average True Range)指标和SMA(Simple Moving Average)指标,实现了一个动态止损追踪的交易系统。当价格在SMA之上时开多单,同时设置基于ATR的动态止损,止损价格会随着价格的上涨而不断提高。当价格跌破动态止损价格时平仓。该策略的主要思想是在趋势行情中,利用动态止损锁定利润,降低回撤。

## 策略原理

1. 计算50日SMA,当收盘价大于50日SMA时开多单。
2. 计算ATR指标,ATR周期为10,乘以一个关键值(默认为3)得到止损幅度nLoss。
3. 计算动态止损价格xATRTrailingStop,初始值为0。
   - 当收盘价和前一个收盘价都大于前一个止损价时,新的止损价为前一个止损价和(收盘价-nLoss)中的较大值。
   - 当收盘价和前一个收盘价都小于前一个止损价时,新的止损价为前一个止损价和(收盘价+nLoss)中的较小值。
   - 其他情况下,新的止损价为(收盘价-nLoss)或(收盘价+nLoss)。
4. 当收盘价跌破动态止损价格时平仓。
5. 止损点用不同颜色标出,多头止损为绿色,空头止损为红色,其他情况为蓝色。

## 优势分析

1. 动态止损机制可以在趋势行情中保护利润,降低回撤风险。与固定止损相比,动态止损更加灵活,能够适应不同的市场状况。
2. 止损幅度基于ATR指标计算,ATR能够很好地反映市场波动率的大小,因此止损距离会根据近期行情的波动率自动调整,在波动加大时放大止损空间,减小波动时缩小止损空间。
3. 使用SMA作为趋势判断依据,能够捕捉到相对明确的趋势行情。在SMA之上开多单,可以在趋势初期介入,博取更大的利润空间。
4. 允许用户设置ATR周期和关键值参数,可以灵活调整策略参数,适应不同品种和周期的特点。

## 风险分析

1. 在趋势不明朗或震荡行情中,该策略可能会出现频繁开平仓的情况,导致交易成本增加,利润减少。
2. 该策略只有做多逻辑,在下跌趋势中无法获利,面临单边市的风险。可以考虑增加做空逻辑,实现双向交易。
3. 止损点基于ATR计算,在行情波动剧烈时,止损空间可能过大,导致风险放大。可以考虑设置一个最大止损幅度,控制单笔交易的最大亏损。
4. 参数选择不当可能导致策略失效。比如ATR周期选择过小,可能会导致止损过于敏感,频繁触发;过大则可能无法及时止损,放大亏损。

## 优化方向

1. 加入做空逻辑,在下跌趋势中也能够获利,提高策略的适应性。可以在价格跌破SMA时开空单,同样采用动态止损逻辑。
2. 引入多空仓位管理,根据趋势强度调整仓位大小。在趋势强烈时加大仓位,提高收益;在趋势较弱时减小仓位,控制风险。
3. 优化止损逻辑,设置一个最大止损幅度,防止在极端行情下出现过大的亏损。同时可以考虑设置一个止盈点,在达到预期收益后主动平仓,而不是一直持有直到止损。
4. 对参数进行优化,通过遍历不同的参数组合,寻找最佳的参数设置。可以使用遗传算法等智能优化方法,提高优化效率。
5. 考虑加入更多的过滤条件,如交易量、波动率等指标,以更好地判断趋势和风险,提高信号的可靠性。

## 总结

该策略基于ATR和SMA指标实现了一个动态止损追踪交易系统,能够在趋势行情中自动调整止损位置,起到保护利润、控制风险的作用。策略逻辑清晰,优势明显,但也存在一些局限性和风险点。通过合理的优化和改进,如加入做空逻辑、优化仓位管理、设置最大止损等,可以进一步提升策略的稳健性和盈利能力。在实际应用中,需要根据不同的交易品种和周期,灵活调整策略参数,并严格控制风险。总的来说,该策略为量化交易提供了一个可行的思路,值得进一步探索和优化。

||

## Overview

This strategy combines the ATR (Average True Range) indicator and the SMA (Simple Moving Average) indicator to implement a dynamic trailing stop trading system. When the price is above the SMA, it opens a long position and sets a dynamic stop loss based on ATR. The stop loss price will continue to rise as the price rises. When the price falls below the dynamic stop loss price, the position is closed. The main idea of this strategy is to lock in profits and reduce drawdowns in trend markets using dynamic stop loss.

## Strategy Principles

1. Calculate the 50-day SMA. When the closing price is greater than the 50-day SMA, open a long position.
2. Calculate the ATR indicator with a period of 10, multiplied by a key value (default is 3) to get the stop loss range nLoss.
3. Calculate the dynamic stop loss price xATRTrailingStop, with an initial value of 0.
   - When the closing price and the previous closing price are both greater than the previous stop loss price, the new stop loss price is the larger of the previous stop loss price and (closing price - nLoss).
   - When the closing price and the previous closing price are both less than the previous stop loss price, the new stop loss price is the smaller of the previous stop loss price and (closing price + nLoss).
   - In other cases, the new stop loss price is (closing price - nLoss) or (closing price + nLoss).
4. When the closing price falls below the dynamic stop loss price, close the position.
5. The stop loss points are marked with different colors: green for long stop loss, red for short stop loss, and blue for other cases.

## Advantage Analysis

1. The dynamic stop loss mechanism can protect profits and reduce drawdown risk in trend markets. Compared with fixed stop loss, dynamic stop loss is more flexible and can adapt to different market conditions.
2. The stop loss range is calculated based on the ATR indicator. ATR can well reflect the size of market volatility, so the stop loss distance will automatically adjust according to the volatility of recent market conditions. It increases the stop loss space when volatility increases and reduces the stop loss space when volatility decreases.
3. Using SMA as the basis for trend judgment can capture relatively clear trend markets. Opening long positions above the SMA can intervene in the early stages of the trend and aim for greater profit space.
4. Allows users to set ATR period and key value parameters, which can flexibly adjust strategy parameters to adapt to the characteristics of different varieties and cycles.

## Risk Analysis

1. In unclear or oscillating markets, this strategy may experience frequent opening and closing of positions, leading to increased transaction costs and reduced profits.
2. This strategy only has long logic and cannot profit in a downward trend, facing the risk of a one-sided market. Consider adding short logic to achieve two-way trading.
3. The stop loss point is based on ATR calculation. When the market fluctuates violently, the stop loss space may be too large, leading to increased risk. Consider setting a maximum stop loss range to control the maximum loss of a single transaction.
4. Improper parameter selection may lead to strategy failure. For example, if the ATR period is too small, it may lead to overly sensitive stop loss and frequent triggers; if it is too large, it may not stop loss in time and amplify losses.

## Optimization Direction

1. Add short logic to profit in downward trends and improve the adaptability of the strategy. You can open a short position when the price falls below the SMA, and also use dynamic stop loss logic.
2. Introduce long and short position management to adjust the position size according to the trend strength. Increase positions when the trend is strong to increase returns; decrease positions when the trend is weak to control risk.
3. Optimize the stop loss logic and set a maximum stop loss range to prevent excessive losses in extreme situations. At the same time, consider setting a take profit point to actively close the position when the expected return is reached, rather than holding it until the stop loss is triggered.
4. Optimize parameters by traversing different parameter combinations to find the best parameter settings. Intelligent optimization methods such as genetic algorithms can be used to improve optimization efficiency.
5. Consider adding more filtering conditions, such as trading volume and volatility indicators, to better judge trends and risks and improve the reliability of signals.

## Summary

This strategy implements a dynamic trailing stop trading system based on the ATR and SMA indicators, which can automatically adjust the stop loss position in trend markets to protect profits and control risks. The strategy logic is clear and has obvious advantages, but it also has some limitations and risk points. Through reasonable optimization and improvement, such as adding short logic, optimizing position management, setting maximum stop loss, etc., the robustness and profitability of the strategy can be further improved. In practical applications, it is necessary to flexibly adjust strategy parameters according to different trading varieties and cycles, and strictly control risks. In general, this strategy provides a feasible idea for quantitative trading and is worthy of further exploration and optimization.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_1|3|Key Value. 'This changes the sensitivity'|
|v_input_2|10|ATR Period|


> Source (PineScript)

``` pinescript
/*backtest
start: 2023-03-05 00:00:00
end: 2024-03-10 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=4
strategy("Trailingstop", overlay=true)

if close > sma(close, 50)
    strategy.entry("long", strategy.long)

// Trailing stop loss for long positions
Trailperc = 0.20
price_stop_long = 0.0

if (strategy.position_size > 0)
    stopValue = close * (1 - Trailperc)
    price_stop_long := max(stopValue, price_stop_long[1])
else
    price_stop_long := 0

if (strategy.position_size > 0)
    strategy.exit(id="stoploss_long", stop=price_stop_long)

// Trailing stop loss for short positions
Trailperc_short = 0.20
price_stop_short = 0.0

if (strategy.position_size < 0)
    stopValue_short = close * (1 + Trailperc_short)
    price_stop_short := min(stopValue_short, price_stop_short[1])
else
    price_stop_short := 0

if (strategy.position_size < 0)
    strategy.exit(id="stoploss_short", stop=price_stop_short)

// ATR Trailing Stop for visualization
keyvalue = input(3, title="Key Value. 'This changes the sensitivity'", step=0.5)
atrperiod = input(10, title="ATR Period")
xATR = atr(atrperiod)
nLoss = keyvalue * xATR

xATRTrailingStop = 0.0
xATRTrailingStop := iff(close > nz(xATRTrailingStop[1], 0) and close[1] > nz(xATRTrailingStop[1], 0), max(nz(xATRTrailingStop[1]), close - nLoss),
   iff(close < nz(xATRTrailingStop[1], 0) and close[1] < nz(xATRTrailingStop[1], 0), min(nz(xATRTrailingStop[1]), close + nLoss),
   iff(close > nz(xATRTrailingStop[1], 0), close - nLoss, close + nLoss)))

pos = 0  
pos :=   iff(close[1] < nz(xATRTrailingStop[1], 0) and close > nz(xATRTrailingStop[1], 0), 1,
   iff(close[1] > nz(xATRTrailingStop[1], 0) and close < nz(xATRTrailingStop[1], 0), -1, nz(pos[1], 0)))

xcolor = pos == -1 ? color.red: pos == 1 ? color.green : color.blue

plot(xATRTrailingStop, color = xcolor, title = "Trailing Stop")
```

> Detail

https://www.fmz.com/strategy/444361

> Last Modified

2024-03-11 11:55:21
