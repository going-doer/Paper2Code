
> Name

SR-突破策略-SR-Breakout-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/11747ab4f0b0f935ba8.png)

[trans]
#### 概述
SR Breakout Strategy是一个基于LonesomeTheBlue的breakout finder指标开发的支撑阻力突破策略。该策略的主要思路是通过判断收盘价是否突破支撑位或阻力位来产生做多或做空信号。默认设置基于8小时K线,但在4小时K线上有更优的参数设置。该策略使用了pivothigh和pivotlow函数来确定支撑位和阻力位,并使用最高价和最低价来判断突破。同时,该策略还设置了止损和止盈。

#### 策略原理
1. 使用pivothigh和pivotlow函数分别计算过去一定周期内的高点和低点,并存储到数组中。
2. 判断当前收盘价是否高于阻力位,如果是,则判断为看涨突破,产生做多信号。
3. 判断当前收盘价是否低于支撑位,如果是,则判断为看跌突破,产生做空信号。
4. 在产生交易信号后,根据设置的止损止盈比例计算止损价和止盈价,并设置相应的止损单和止盈单。
5. 根据突破方向绘制相应的突破区间。

#### 策略优势
1. 支撑阻力突破是一个经典的交易策略,有一定的实战基础。
2. 通过使用pivothigh和pivotlow函数计算支撑位和阻力位,可以比较准确地捕捉到突破行情。
3. 该策略代码结构清晰,通过存储高点和低点到数组中,可以方便地进行回测和优化。
4. 设置了止损和止盈,可以比较好地控制风险。

#### 策略风险
1. 支撑阻力突破策略在震荡行情中表现不佳,容易出现频繁的false breakout。
2. 固定的止损止盈比例可能无法适应不同的行情,导致风险收益失衡。
3. 该策略只考虑了价格因素,而没有考虑成交量等其他重要指标,可能会错过一些重要的信号。

#### 策略优化方向
1. 可以考虑引入更多的技术指标,如成交量、MACD等,以提高信号的准确性和可靠性。
2. 对于止损和止盈,可以考虑使用移动止损或者动态止损止盈比例,以更好地适应不同的行情。
3. 可以考虑引入过滤条件,如趋势过滤、波动率过滤等,以减少在震荡行情中的false breakout。
4. 可以考虑对支撑位和阻力位进行优化,如使用自适应周期、引入斐波那契等级等。

#### 总结
SR Breakout Strategy是一个基于经典支撑阻力突破思路的交易策略,通过使用pivothigh和pivotlow函数计算支撑位和阻力位,并通过判断收盘价是否突破这些位置来产生交易信号。该策略的优势在于思路清晰,易于实现和优化;同时也存在一些风险,如在震荡行情中表现不佳,以及固定的止损止盈比例可能带来的风险。未来可以考虑从技术指标、止损止盈、过滤条件、支撑阻力优化等方面对该策略进行优化和改进,以提高其稳定性和盈利能力。

|| 

#### Overview
The SR Breakout Strategy is a support and resistance breakout strategy developed based on LonesomeTheBlue's breakout finder indicator. The main idea of this strategy is to generate long or short signals by judging whether the closing price breaks through the support or resistance level. The default settings are based on the 8-hour candlestick chart, but there are more optimal parameter settings on the 4-hour candlestick chart. This strategy uses the pivothigh and pivotlow functions to determine support and resistance levels, and uses the highest and lowest prices to determine breakouts. At the same time, this strategy also sets stop loss and take profit.

#### Strategy Principle
1. Use the pivothigh and pivotlow functions to calculate the highs and lows over a certain period of time and store them in arrays.
2. Determine whether the current closing price is higher than the resistance level. If so, it is judged as a bullish breakout and a long signal is generated.
3. Determine whether the current closing price is lower than the support level. If so, it is judged as a bearish breakout and a short signal is generated.
4. After generating a trading signal, calculate the stop loss and take profit prices based on the set stop loss and take profit ratios, and set the corresponding stop loss and take profit orders.
5. Draw the corresponding breakout range according to the breakout direction.

#### Strategy Advantages
1. Support and resistance breakout is a classic trading strategy with a certain practical basis.
2. By using the pivothigh and pivotlow functions to calculate support and resistance levels, breakout opportunities can be captured relatively accurately.
3. The code structure of this strategy is clear, and by storing highs and lows in arrays, it is convenient for backtesting and optimization.
4. Stop loss and take profit are set, which can control risks relatively well.

#### Strategy Risks
1. The support and resistance breakout strategy performs poorly in choppy markets and is prone to frequent false breakouts.
2. Fixed stop loss and take profit ratios may not be able to adapt to different market conditions, resulting in an imbalance of risk and return.
3. This strategy only considers price factors and does not consider other important indicators such as trading volume, which may miss some important signals.

#### Strategy Optimization Direction
1. Consider introducing more technical indicators, such as trading volume, MACD, etc., to improve the accuracy and reliability of signals.
2. For stop loss and take profit, consider using trailing stop or dynamic stop loss and take profit ratios to better adapt to different market conditions.
3. Consider introducing filtering conditions, such as trend filtering, volatility filtering, etc., to reduce false breakouts in choppy markets.
4. Consider optimizing support and resistance levels, such as using adaptive periods, introducing Fibonacci levels, etc.

#### Summary
The SR Breakout Strategy is a trading strategy based on the classic idea of support and resistance breakout. By using the pivothigh and pivotlow functions to calculate support and resistance levels, and by judging whether the closing price breaks through these levels to generate trading signals. The advantage of this strategy is that the idea is clear and easy to implement and optimize; at the same time, there are also some risks, such as poor performance in choppy markets, and the risks that may be brought about by fixed stop loss and take profit ratios. In the future, we can consider optimizing and improving this strategy from aspects such as technical indicators, stop loss and take profit, filtering conditions, support and resistance optimization, etc., to improve its stability and profitability.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-05-07 00:00:00
end: 2024-05-14 00:00:00
period: 10m
basePeriod: 1m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © LonesomeTheBlue © chanu_lev10k

//@version=5
strategy('SR Breakout Strategy', overlay=true, max_bars_back=500, max_lines_count=400)
prd = input.int(defval=5, title='Period', minval=2)
bo_len = input.int(defval=71, title='Max Breakout Length', minval=30, maxval=300)
cwidthu = input.float(defval=3., title='Threshold Rate %', minval=1., maxval=10) / 100
mintest = input.int(defval=2, title='Minimum Number of Tests', minval=1)
bocolorup = input.color(defval=color.blue, title='Breakout Colors', inline='bocol')
bocolordown = input.color(defval=color.red, title='', inline='bocol')
// lstyle = input.string(defval=line.style_solid, title='Line Style')
issl = input.bool(title='SL', inline='linesl1', group='Stop Loss / Take Profit:', defval=false)
slpercent = input.float(title=', %', inline='linesl1', group='Stop Loss / Take Profit:', defval=18.0, minval=0.0, step=0.1)
istp = input.bool(title='TP', inline='linetp1', group='Stop Loss / Take Profit:', defval=false)
tppercent = input.float(title=', %', inline='linetp1', group='Stop Loss / Take Profit:', defval=18.0, minval=0.0, step=0.1)

//width
lll = math.max(math.min(bar_index, 300), 1)
float h_ = ta.highest(lll)
float l_ = ta.lowest(lll)
float chwidth = (h_ - l_) * cwidthu

// check if PH/PL
ph = ta.pivothigh(prd, prd)
pl = ta.pivotlow(prd, prd)

//keep Pivot Points and their locations in the arrays
var phval = array.new_float(0)
var phloc = array.new_int(0)
var plval = array.new_float(0)
var plloc = array.new_int(0)

// keep PH/PL levels and locations
if bool(ph)
    array.unshift(phval, ph)
    array.unshift(phloc, bar_index - prd)
    if array.size(phval) > 1  // cleanup old ones
        for x = array.size(phloc) - 1 to 1 by 1
            if bar_index - array.get(phloc, x) > bo_len
                array.pop(phloc)
                array.pop(phval)

if bool(pl)
    array.unshift(plval, pl)
    array.unshift(plloc, bar_index - prd)
    if array.size(plval) > 1  // cleanup old ones
        for x = array.size(plloc) - 1 to 1 by 1
            if bar_index - array.get(plloc, x) > bo_len
                array.pop(plloc)
                array.pop(plval)

// check bullish cup
float bomax = na
int bostart = bar_index
num = 0
hgst = ta.highest(prd)[1]
if array.size(phval) >= mintest and close > open and close > hgst
    bomax := array.get(phval, 0)
    xx = 0
    for x = 0 to array.size(phval) - 1 by 1
        if array.get(phval, x) >= close
            break
        xx := x
        bomax := math.max(bomax, array.get(phval, x))
        bomax
    if xx >= mintest and open <= bomax
        for x = 0 to xx by 1
            if array.get(phval, x) <= bomax and array.get(phval, x) >= bomax - chwidth
                num += 1
                bostart := array.get(phloc, x)
                bostart
        if num < mintest or hgst >= bomax
            bomax := na
            bomax

// if not na(bomax) and num >= mintest
//     line.new(x1=bar_index, y1=bomax, x2=bostart, y2=bomax, color=bocolorup)
//     line.new(x1=bar_index, y1=bomax - chwidth, x2=bostart, y2=bomax - chwidth, color=bocolorup)
//     line.new(x1=bostart, y1=bomax - chwidth, x2=bostart, y2=bomax, color=bocolorup)
//     line.new(x1=bar_index, y1=bomax - chwidth, x2=bar_index, y2=bomax, color=bocolorup)

plotshape(not na(bomax) and num >= mintest, location=location.belowbar, style=shape.triangleup, color=bocolorup, size=size.small)
//alertcondition(not na(bomax) and num >= mintest, title='Breakout', message='Breakout')

// check bearish cup
float bomin = na
bostart := bar_index
num1 = 0
lwst = ta.lowest(prd)[1]
if array.size(plval) >= mintest and close < open and close < lwst
    bomin := array.get(plval, 0)
    xx = 0
    for x = 0 to array.size(plval) - 1 by 1
        if array.get(plval, x) <= close
            break
        xx := x
        bomin := math.min(bomin, array.get(plval, x))
        bomin
    if xx >= mintest and open >= bomin
        for x = 0 to xx by 1
            if array.get(plval, x) >= bomin and array.get(plval, x) <= bomin + chwidth
                num1 += 1
                bostart := array.get(plloc, x)
                bostart
        if num1 < mintest or lwst <= bomin
            bomin := na
            bomin

// if not na(bomin) and num1 >= mintest
//     line.new(x1=bar_index, y1=bomin, x2=bostart, y2=bomin, color=bocolordown)
//     line.new(x1=bar_index, y1=bomin + chwidth, x2=bostart, y2=bomin + chwidth, color=bocolordown)
//     line.new(x1=bostart, y1=bomin + chwidth, x2=bostart, y2=bomin, color=bocolordown)
//     line.new(x1=bar_index, y1=bomin + chwidth, x2=bar_index, y2=bomin, color=bocolordown)

plotshape(not na(bomin) and num1 >= mintest, location=location.abovebar, style=shape.triangledown, color=bocolordown, size=size.small)

//alertcondition(not na(bomin) and num1 >= mintest, title='Breakdown', message='Breakdown')
//alertcondition(not na(bomax) and num >= mintest or not na(bomin) and num1 >= mintest, title='Breakout or Breakdown', message='Breakout or Breakdown')

// Long Short conditions
longCondition = not na(bomax) and num >= mintest
if longCondition
    strategy.entry('Long', strategy.long)
shortCondition = not na(bomin) and num1 >= mintest
if shortCondition
    strategy.entry('Short', strategy.short)

// Entry price / Take Profit / Stop Loss
//entryprice = strategy.position_avg_price
entryprice = ta.valuewhen(condition=longCondition or shortCondition, source=close, occurrence=0)
pm = longCondition ? 1 : shortCondition ? -1 : 1 / math.sign(strategy.position_size)
takeprofit = entryprice * (1 + pm * tppercent * 0.01)
stoploss = entryprice * (1 - pm * slpercent * 0.01)
strategy.exit(id='Exit Long', from_entry='Long', stop=issl ? stoploss : na, limit=istp ? takeprofit : na, alert_message='Exit Long')
strategy.exit(id='Exit Short', from_entry='Short', stop=issl ? stoploss : na, limit=istp ? takeprofit : na, alert_message='Exit Short')
```

> Detail

https://www.fmz.com/strategy/451527

> Last Modified

2024-05-15 16:30:14
