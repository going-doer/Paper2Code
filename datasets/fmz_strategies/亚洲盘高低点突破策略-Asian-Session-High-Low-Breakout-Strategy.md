
> Name

亚洲盘高低点突破策略-Asian-Session-High-Low-Breakout-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1f4e31354879f1107e9.png)
[trans]

#### 概述
该策略的主要思路是利用亚洲盘的高低点作为突破点,在欧美盘开盘后的几个小时内,如果价格突破亚洲盘高点则做多,突破亚洲盘低点则做空。同时设置止损和止盈,控制风险。该策略每天只开一个交易,最大同时开仓数量为100000。

#### 策略原理
1. 确定亚洲盘的交易时间,用户可以自定义起始时间和结束时间。
2. 在亚洲盘期间,记录当天的最高价和最低价。
3. 在欧美盘开盘后的某个时间(用户自定义的偏移小时数),如果价格突破亚洲盘高点则做多,突破亚洲盘低点则做空。
4. 设置止损和止盈,止损和止盈的点数可以自定义。
5. 每天只开一个新的交易,同时最大同时开仓数量为100000。
6. 如果当天已经开过仓,则不再开新的交易。

#### 优势分析
1. 利用亚洲盘相对平静的特点,以当天亚洲盘的高低点作为突破点,可以较好地捕捉欧美盘的趋势机会。
2. 设置了止损和止盈,可以有效控制风险,让盈利单够跑,亏损单快止损。
3. 限制了每天只开一单以及最大同时开仓数量,可以避免过度交易和资金的过度使用。
4. 用户可以根据自己的需求,灵活设置亚洲盘时间和偏移小时数等参数。

#### 风险分析
1. 亚洲盘高低点不一定是当天真正的高低点,有可能欧美盘突破后又快速回撤,造成亏损。
2. 固定点数的止损和止盈可能无法应对行情的大幅波动,有时可能止损过早,有时可能止盈过早。
3. 在趋势不明显或者市场波动较大的情况下,该策略可能会出现频繁开仓止损的情况。

#### 优化方向
1. 可以考虑根据ATR等波动率指标来动态调整止损和止盈的点数,以适应不同的行情。
2. 可以加入一些趋势判断的指标,如MA,只有在大趋势向上时做多,向下时做空,以提高成功率。
3. 可以考虑分时段设置不同的参数,如在欧美盘开盘初期使用较小的止损止盈,而在趋势明显时则加大止损止盈。

#### 总结
该策略利用亚洲盘的高低点作为突破点来进行交易,适合在欧美盘趋势较为明显的品种上使用。但是固定点数止损止盈以及标准的突破入场方式也存在一些局限性。通过引入一些动态化和趋势性的指标,可以对该策略进行优化,以期获得更好的效果。

|| 

#### Overview
The main idea of this strategy is to use the high and low points of the Asian session as breakout points. Within a few hours after the European and American markets open, if the price breaks above the Asian session high, it goes long; if it breaks below the Asian session low, it goes short. Stop loss and take profit are set to control risk. The strategy only opens one trade per day, with a maximum of 100,000 simultaneous positions.

#### Strategy Principle
1. Determine the trading time of the Asian session. Users can customize the start and end times.
2. During the Asian session, record the highest and lowest prices of the day.
3. At a certain time (user-defined offset hours) after the European and American markets open, if the price breaks above the Asian session high, go long; if it breaks below the Asian session low, go short.
4. Set stop loss and take profit. The number of points for stop loss and take profit can be customized.
5. Only open one new trade per day, with a maximum of 100,000 simultaneous positions.
6. If a position has already been opened for the day, no new trades will be opened.

#### Advantage Analysis
1. By using the relatively calm characteristics of the Asian session and using the high and low points of the Asian session as breakout points, it can better capture the trend opportunities of the European and American sessions.
2. Setting stop loss and take profit can effectively control risk, letting profitable trades run and quickly stopping losses on unprofitable trades.
3. Limiting to only one trade per day and a maximum number of simultaneous positions can avoid overtrading and excessive use of funds.
4. Users can flexibly set parameters such as Asian session time and offset hours according to their own needs.

#### Risk Analysis
1. The high and low points of the Asian session may not be the true high and low points of the day. It is possible that after the European and American markets break through, they quickly retrace, causing losses.
2. Fixed-point stop loss and take profit may not be able to cope with large fluctuations in the market. Sometimes the stop loss may be too early, and sometimes the take profit may be too early.
3. In situations where the trend is not obvious or the market volatility is high, the strategy may experience frequent opening and stopping losses.

#### Optimization Direction
1. Consider dynamically adjusting the number of points for stop loss and take profit based on volatility indicators such as ATR to adapt to different market conditions.
2. Add some trend judgment indicators, such as MA, and only go long when the big trend is up and go short when it is down to improve the success rate.
3. Consider setting different parameters for different time periods, such as using smaller stop loss and take profit at the beginning of the European and American trading sessions, and increasing stop loss and take profit when the trend is obvious.

#### Summary
This strategy uses the high and low points of the Asian session as breakout points for trading and is suitable for use on varieties with obvious trends in the European and American markets. However, fixed-point stop loss and take profit and standard breakout entry methods also have some limitations. By introducing some dynamic and trend-based indicators, the strategy can be optimized to obtain better results.

[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_1|19|Asia session start hour|
|v_input_2|true|Asia session end hour|
|v_input_3|3|Offset hours after Asia session end|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-02-27 00:00:00
end: 2024-03-28 00:00:00
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=4
strategy("Asia Session", overlay=true)

var hourSessionStart = input(19, "Asia session start hour", minval=0, maxval=23)
var hourSessionStop  = input(1, "Asia session end hour", minval=0, maxval=23)
var offsetHours = input(3, "Offset hours after Asia session end")

var float hi              = na
var float lo              = na
var float plotHi          = na
var float plotLo          = na

var bool  inSession       = na
var bool  enteringSession = na
var bool  exitingSession  = na

inSession       := (hour >= hourSessionStart or hour < hourSessionStop)
enteringSession := inSession and not inSession[1]
exitingSession  := not inSession and inSession[1]

if enteringSession
    plotLo := na
    plotHi := na

if inSession
    lo := min(low,  nz(lo, 1.0e23))
    hi := max(high, nz(hi))

if exitingSession
    plotLo := lo
    plotHi := hi
    lo     := na
    hi     := na

bgcolor(inSession ? color.blue : na)

plot(plotLo, "Asia session Low",  color.red,   style=plot.style_linebr)
plot(plotHi, "Asia session High", color.green, style=plot.style_linebr)

// Implementazione delle condizioni di entrata
var float asiaSessionLow = na
var float asiaSessionHigh = na
var int maxTrades = 100000 // Impostiamo il massimo numero di operazioni contemporanee
var int tradesOpened = 0 // Variabile per tenere traccia del numero di operazioni aperte
var bool tradeOpened = false
var bool operationClosed = false // Nuova variabile per tenere traccia dello stato di chiusura dell'operazione

// Calcolo del range asiatico
if (inSession)
    asiaSessionLow := lo
    asiaSessionHigh := hi

// Apertura di un solo trade al giorno
if (enteringSession)
    tradeOpened := false

// Condizioni di entrata
var float stopLoss = 300 * syminfo.mintick
var float takeProfit = 300 * syminfo.mintick

if (not tradeOpened and not operationClosed and close < asiaSessionLow and tradesOpened < maxTrades and hour >= hourSessionStop + offsetHours)
    strategy.entry("Buy", strategy.long)
    tradeOpened := true
    tradesOpened := tradesOpened + 1 // Incrementiamo il contatore delle operazioni aperte

if (not tradeOpened and not operationClosed and close > asiaSessionHigh and tradesOpened < maxTrades and hour >= hourSessionStop + offsetHours)
    strategy.entry("Sell", strategy.short)
    tradeOpened := true
    tradesOpened := tradesOpened + 1 // Incrementiamo il contatore delle operazioni aperte

// Impostazione dello stop loss e del take profit
strategy.exit("Stop Loss / Take Profit", "Buy", stop=close - stopLoss, limit=close + takeProfit)
strategy.exit("Stop Loss / Take Profit", "Sell", stop=close + stopLoss, limit=close - takeProfit)

```

> Detail

https://www.fmz.com/strategy/446567

> Last Modified

2024-03-29 17:12:49
