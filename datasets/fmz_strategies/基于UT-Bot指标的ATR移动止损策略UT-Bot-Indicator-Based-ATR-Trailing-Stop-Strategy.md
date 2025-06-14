
> Name

基于UT-Bot指标的ATR移动止损策略UT-Bot-Indicator-Based-ATR-Trailing-Stop-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/11ddb3b9afabc59941a.png)
[trans]
## 概述

该策略基于QuantNomad开发的UT Bot指标,结合了移动止损的思路。原始代码由@Yo_adriiiiaan编写,@HPotter进行了修改。该策略将与LuxAlgo的Smart Money Concepts配合使用。目前该策略处于测试阶段。

## 策略原理

该策略的主要原理如下:

1. 当收盘价高于50期简单移动平均线时,进行做多交易。
2. 对于多头仓位,设置移动止损价格。移动止损价格为当前收盘价的80%(1-20%)。移动止损价格会随着价格上涨而上移,但不会下移,从而起到保护利润的作用。
3. 对于空头仓位,同样设置移动止损价格。移动止损价格为当前收盘价的120%(1+20%)。移动止损价格会随着价格下跌而下移,但不会上移。
4. 使用ATR(Average True Range,平均真实波幅)作为移动止损的参考依据。ATR移动止损价格的计算方法为:向上移动时,取前一个ATR移动止损价格和(当前收盘价-ATR*Key Value)两者的较大值;向下移动时,取前一个ATR移动止损价格和(当前收盘价+ATR*Key Value)两者的较小值。其中Key Value为用户设置的参数,用于调节移动止损的敏感度。
5. 根据ATR移动止损价格的突破情况,判断当前持仓方向。当价格向上突破ATR移动止损价格时,持多头仓位;当价格向下突破ATR移动止损价格时,持空头仓位;其他情况下保持当前持仓状态不变。

## 优势分析

1. 移动止损的设置可以很好地保护利润,使得策略在趋势行情中能够获得更多收益。
2. 分别对多头和空头仓位设置止损,能够适应不同的行情。
3. 使用ATR作为止损的参考依据,可以动态调整止损位置,更加灵活有效。
4. 提供了Key Value参数供用户优化,可以根据不同品种和周期进行调节,提高适应性。

## 风险分析

1. 在震荡行情中,频繁的止损可能导致过多的交易,增加手续费成本,降低收益。
2. 固定百分比的移动止损方式相对简单,在一些行情中可能无法很好地应对价格波动。
3. 策略中只考虑了移动止损,而没有设置移动止盈,可能错失一些盈利机会。
4. 参数的选择对策略表现有较大影响,不当的参数可能带来更大的回撤风险。

## 优化方向

1. 可以考虑结合其他指标或条件,例如交易量、波动率等,对进场条件进行优化,提高信号的可靠性。
2. 对于移动止损的计算方法,可以探索更加复杂和有效的方式,例如使用抛物线止损、动态百分比止损等。
3. 可以加入移动止盈的机制,例如根据ATR或百分比设置动态止盈位,以更好地锁定利润。
4. 针对不同的品种和周期,可以进行参数优化,寻找最适合的参数组合。也可以根据市场状态的变化,动态调整参数。

## 总结

该策略在UT Bot指标的基础上,加入了移动止损的逻辑,能够在趋势行情中起到保护利润的作用。同时,策略对多头和空头仓位分别设置止损,适应性较强。使用ATR作为移动止损的参考依据,可以动态调整止损位置,提高灵活性。但是,该策略在震荡行情中可能面临频繁止损带来的高交易成本风险,且缺乏移动止盈的设置,有错失盈利的可能。此外,参数的选择对策略表现有较大影响。

未来,可以从优化进场条件、探索更复杂的移动止损方式、加入移动止盈机制、针对不同品种和周期进行参数优化等方面对策略进行完善,以期获得更稳健的收益。总的来说,该策略思路简单明了,易于理解和实现,但还有进一步优化的空间,值得继续探索和改进。

|| 

## Overview

This strategy is based on the UT Bot indicator developed by QuantNomad and incorporates the idea of a trailing stop loss. The original code was written by @Yo_adriiiiaan and modified by @HPotter. The strategy will be used in conjunction with LuxAlgo's Smart Money Concepts. Currently, the strategy is in the testing phase.

## Strategy Principle

The main principles of this strategy are as follows:

1. When the closing price is higher than the 50-period simple moving average, a long trade is entered.
2. For long positions, a trailing stop loss price is set. The trailing stop loss price is 80% (1-20%) of the current closing price. The trailing stop loss price moves up with rising prices but does not move down, thus protecting profits.
3. For short positions, a trailing stop loss price is also set. The trailing stop loss price is 120% (1+20%) of the current closing price. The trailing stop loss price moves down with falling prices but does not move up.
4. ATR (Average True Range) is used as a reference for the trailing stop. The calculation method for the ATR trailing stop price is: when moving up, take the larger of the previous ATR trailing stop price and (current closing price - ATR * Key Value); when moving down, take the smaller of the previous ATR trailing stop price and (current closing price + ATR * Key Value). The Key Value is a user-set parameter used to adjust the sensitivity of the trailing stop.
5. Based on the breakthrough of the ATR trailing stop price, the current position direction is determined. When the price breaks above the ATR trailing stop price, a long position is held; when the price breaks below the ATR trailing stop price, a short position is held; in other cases, the current position status remains unchanged.

## Advantage Analysis

1. The setting of the trailing stop can effectively protect profits, allowing the strategy to obtain more gains in trend markets.
2. Setting stop losses separately for long and short positions can adapt to different market conditions.
3. Using ATR as a reference for the stop loss allows for dynamic adjustment of the stop loss position, making it more flexible and effective.
4. The Key Value parameter is provided for user optimization, which can be adjusted according to different varieties and cycles to improve adaptability.

## Risk Analysis

1. In choppy markets, frequent stop-outs may lead to excessive transactions, increasing commission costs and reducing returns.
2. The fixed percentage trailing stop method is relatively simple and may not be able to cope well with price fluctuations in some market conditions.
3. The strategy only considers trailing stops and does not set trailing profit targets, which may miss some profit opportunities.
4. The choice of parameters has a significant impact on strategy performance, and inappropriate parameters may bring greater drawdown risks.

## Optimization Direction

1. Other indicators or conditions, such as trading volume and volatility, can be considered to optimize entry conditions and improve signal reliability.
2. For the calculation method of the trailing stop, more complex and effective methods can be explored, such as using parabolic stop loss or dynamic percentage stop loss.
3. A trailing profit target mechanism can be added, for example, setting dynamic profit targets based on ATR or percentages to better lock in profits.
4. Parameter optimization can be performed for different varieties and cycles to find the most suitable parameter combinations. Parameters can also be dynamically adjusted according to changes in market conditions.

## Summary

Based on the UT Bot indicator, this strategy incorporates trailing stop logic, which can protect profits in trend markets. At the same time, the strategy sets stop losses separately for long and short positions, making it highly adaptable. Using ATR as a reference for the trailing stop allows for dynamic adjustment of the stop loss position, improving flexibility. However, this strategy may face the risk of high transaction costs due to frequent stop-outs in choppy markets, and it lacks a trailing profit target setting, which may miss profit opportunities. In addition, the choice of parameters has a significant impact on strategy performance.

In the future, the strategy can be improved by optimizing entry conditions, exploring more complex trailing stop methods, adding a trailing profit target mechanism, and optimizing parameters for different varieties and cycles, in order to obtain more stable returns. Overall, the strategy idea is simple and straightforward, easy to understand and implement, but there is room for further optimization and it is worth continuing to explore and improve.
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

https://www.fmz.com/strategy/444344

> Last Modified

2024-03-11 11:17:33
