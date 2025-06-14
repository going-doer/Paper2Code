
> Name

基于超级趋势和MACD的风险控制策略Samsuga-SuperTrend-MACD-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/16ec00fc8fedb3abb86.png)
[trans]

## 概述

该策略结合了超级趋势指标和MACD指标,通过捕捉小趋势来获取利润。策略使用超级趋势指标来判断当前市场趋势,同时使用MACD指标作为进场和出场的辅助条件。策略逻辑清晰,易于理解和实现。

## 策略原理

1. 使用ta.supertrend函数计算超级趋势指标,参数为ATR周期和乘数因子。
2. 根据超级趋势指标的方向变化来判断多空趋势,当direction从大于0变为小于等于0时,视为上涨趋势;反之,则视为下跌趋势。
3. 使用request.security函数获取30分钟周期的MACD指标值,包括MACD线、信号线和柱状图。
4. 在上涨趋势中,如果MACD柱状图大于0,则开多仓,同时平掉之前的空仓。
5. 在下跌趋势中,如果MACD柱状图小于0,则开空仓,同时平掉之前的多仓。

## 优势分析

1. 结合了趋势跟踪和动量指标,能够较好地适应不同的市场状况。
2. 使用了更长周期的MACD指标作为辅助条件,可以有效过滤掉一些假信号。
3. 策略逻辑简单明了,易于理解和实现,适合初学者学习。
4. 策略参数可调,可以根据不同的市场和品种进行优化。

## 风险分析

1. 策略在震荡市场中可能会出现较多的交易信号,导致频繁交易和高滑点成本。
2. 超级趋势指标对参数较为敏感,不同的参数设置可能会得到不同的结果。
3. MACD指标可能会出现与价格背离的情况,导致错误的交易信号。
4. 策略缺乏止损措施,在行情持续性不强或突发事件时可能会承担较大风险。

## 优化方向

1. 可以考虑加入更多的过滤条件,如价格突破重要支撑或阻力位、交易量变化等,以提高信号的可靠性。
2. 对于震荡市场,可以考虑使用更短周期的MACD指标或其他适合震荡市的指标来判断趋势。
3. 可以加入止损措施,如固定点数止损、移动止损等,以控制单笔交易的最大风险。
4. 可以对不同市场和品种进行参数优化,找到最适合的参数组合。

## 总结

该策略通过结合超级趋势指标和MACD指标,在捕捉小趋势的同时也考虑了趋势的持续性,是一个较为全面和均衡的策略。策略优势在于逻辑清晰,易于理解和实现,同时通过使用更长周期的MACD指标作为辅助条件,可以有效过滤掉一些假信号。但策略也存在一些风险,如在震荡市场中可能出现频繁交易,对参数设置较为敏感,以及缺乏止损措施等。针对这些风险,可以从加入更多过滤条件、优化参数、加入止损等方面进行优化和改进。总的来说,该策略可以作为一个基础性的策略框架,通过不断优化和改进,有望成为一个稳定盈利的策略。

|| 

## Overview

This strategy combines the SuperTrend indicator and the MACD indicator to capture small trends for profit. It uses the SuperTrend indicator to determine the current market trend and the MACD indicator as an auxiliary condition for entry and exit. The strategy logic is clear and easy to understand and implement.

## Strategy Principle

1. Use the ta.supertrend function to calculate the SuperTrend indicator with parameters for ATR period and multiplier factor.
2. Determine the long/short trend based on changes in the direction of the SuperTrend indicator. When direction changes from greater than 0 to less than or equal to 0, it is considered an uptrend; otherwise, it is considered a downtrend.
3. Use the request.security function to obtain the MACD indicator values on the 30-minute timeframe, including the MACD line, signal line, and histogram.
4. In an uptrend, if the MACD histogram is greater than 0, open a long position and close any previous short position.
5. In a downtrend, if the MACD histogram is less than 0, open a short position and close any previous long position.

## Advantage Analysis

1. Combines trend following and momentum indicators, allowing it to adapt well to different market conditions.
2. Uses a longer timeframe MACD indicator as an auxiliary condition, which can effectively filter out some false signals.
3. The strategy logic is simple and straightforward, easy to understand and implement, suitable for beginners to learn.
4. Strategy parameters are adjustable and can be optimized for different markets and instruments.

## Risk Analysis

1. The strategy may generate frequent trading signals in choppy markets, leading to high trading frequency and slippage costs.
2. The SuperTrend indicator is sensitive to parameter settings, and different parameter values may produce different results.
3. The MACD indicator may experience divergence from price, resulting in erroneous trading signals.
4. The strategy lacks stop-loss measures, which may expose it to greater risk during weak trend continuity or unexpected events.

## Optimization Direction

1. Consider adding more filtering conditions, such as price breaking through important support/resistance levels, changes in trading volume, etc., to improve signal reliability.
2. For choppy markets, consider using a shorter timeframe MACD indicator or other indicators suitable for range-bound markets to determine the trend.
3. Incorporate stop-loss measures, such as fixed point stop-loss, trailing stop-loss, etc., to control the maximum risk per trade.
4. Optimize parameters for different markets and instruments to find the most suitable parameter combinations.

## Conclusion

By combining the SuperTrend indicator and MACD indicator, this strategy captures small trends while also considering trend continuity, making it a relatively comprehensive and balanced strategy. The strategy's strengths lie in its clear logic, ease of understanding and implementation, and the use of a longer timeframe MACD indicator as an auxiliary condition to effectively filter out false signals. However, the strategy also has some risks, such as the potential for frequent trading in choppy markets, sensitivity to parameter settings, and lack of stop-loss measures. To address these risks, improvements can be made by adding more filtering conditions, optimizing parameters, incorporating stop-losses, etc. Overall, this strategy can serve as a basic strategy framework that, with continuous optimization and improvement, has the potential to become a consistently profitable strategy.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_int_1|7|ATR Length|
|v_input_float_1|true|Factor|


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
strategy("Samsuga supertrend", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=100)


atrPeriod = input.int(7,    "ATR Length", minval = 1)
factor =    input.float(1.0, "Factor",     minval = 0.01, step = 0.01)

[supertrend, direction] = ta.supertrend(factor, atrPeriod)

supertrend := barstate.isfirst ? na : supertrend
upTrend =    plot(direction <= 0 ? supertrend : na, "Up Trend",   color = color.green, style = plot.style_linebr)
downTrend =  plot(direction <= 0 ? na : supertrend, "Down Trend", color = color.red,   style = plot.style_linebr)
bodyMiddle = plot(barstate.isfirst ? na : (open + close) / 2, "Body Middle",display = display.none)
longcondition = direction[1] > direction 
shortCondition = direction[1] < direction 

macdp1 = 3
macdp2=10
macdp3=6

[macdLine, signalLine, histLine] =request.security(symbol = syminfo.tickerid, timeframe = "30",expression = ta.macd(close,macdp1,macdp2,macdp3),lookahead=barmerge.lookahead_on)
// plot(macdLine,   title = "MACD",   color = #2962FF)
// plot(signalLine, title = "Signal", color = #FF6D00)
// 8, 21, 5
// 8,13,9
// 12,26,9
//  1--> 3, 17, 5
// 3, 10, 16
// log.info(str.tostring(syminfo.tickerid)+str.tostring(histLine[0]))
//  /////////----------------METHOD 1-----------------////////////////
// if(longcondition)
//     if(strategy.opentrades>0)
//         strategy.close("Long","Prev Exit", immediately = true)
//     if( histLine[0] > 0.1)
//         strategy.entry(id= "Long", direction=strategy.long,  comment = "update long")

    
// else if(shortCondition and strategy.openprofit<=0.1) 
//     strategy.close("Long",comment = "Close",immediately = true)
//  /////////----------------METHOD 2-----------------////////////////
// if(longcondition)
//     if(histLine[0] > 0)
//         strategy.entry(id= "Long", direction=strategy.long,  comment = "update long" )
//         strategy.exit("Long", loss = close*0.2)


    
// else if(shortCondition ) 
//     strategy.close("Long",comment = "Close",immediately = true)
//  /////////----------------METHOD 3-----------------////////////////
// log.info(str.tostring(syminfo.tickerid)+str.tostring(histLine[0]))
if(longcondition)
    if(histLine[0] > 0)    
        strategy.close("Short",comment = "E-S", alert_message = "E-S",disable_alert = true)
        strategy.entry(id= "Long", direction=strategy.long,  comment = "L",alert_message = "L")
else if(shortCondition) 
    if(histLine[0] < 0)    
        strategy.close("Long",comment = "E-L",alert_message = "E-L",disable_alert = true)
        strategy.entry(id= "Short", direction=strategy.short,  comment = "S",alert_message = "S")
```

> Detail

https://www.fmz.com/strategy/444350

> Last Modified

2024-03-11 11:24:20
