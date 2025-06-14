
> Name

基于动量顺势策略Momentum-Trend-Following-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/101c4bd245a7f199d15.png)
[trans]
### 概述

本策略结合使用Aroon指标和绝对力度指标(ASH),旨在识别市场趋势和潜在交易机会。Aroon有助于识别趋势的强度和方向,而ASH提供动量实力的洞察。通过组合这些指标,策略试图捕捉以太坊市场中的可盈利交易机会。

### 策略原理

该策略使用两个Aroon指标参数组:

- 多头仓位:Aroon周期为56(上轨)和20(下轨)
- 空头仓位:Aroon周期为17(上轨)和55(下轨)

ASH的长度为9根K线,使用收盘价作为数据源。

策略包含特定的进入和退出交易条件:

1. 多头仓位进入:当Aroon指标上穿下轨时,表示潜在上涨趋势,因此开多仓。  
2. 多头仓位退出:当Aroon指标下穿下轨时,平多仓。
3. 空头仓位进入:当Aroon指标下穿上轨时,表示潜在下跌趋势,因此开空仓。 
4. 空头仓位退出:当Aroon指标上穿上轨时,平空仓。

### 优势分析

本策略最大的优势在于结合运用两个指标的优势。Aroon指标可以有效判断趋势方向和强度,ASH指标提供额外的动量洞察,有助于判断进入和退出时机。

此外,使用两个不同参数组的Aroon指标进行多空判断,可以灵活适应市场态势的变化。

### 风险分析

该策略主要的风险在于指标本身的局限性。Aroon指标对于震荡盘整市较弱,容易产生错误信号。ASH指标对于短期过度反转也较为敏感。

此外,参数设置如果不当,也会影响策略表现。需要对Aroon指标的长短周期以及ASH指标的长度进行优化和测试,找到最佳参数组合。

### 优化方向

可以考虑添加过滤器,如价格突破、交易量增加等,避免在震荡行情中产生错误信号。

可以测试不同的指标参数组合和权重,找到最优参数。也可以尝试结合其他指标,如RSI、KD等,形成更强大的指标组合,提高策略表现。

### 总结

本策略整合运用Aroon和ASH两个指标的优势,通过双指标确认,在判断趋势和捕捉转折点时效果较好。但参数设置和指标本身局限仍需优化。总体来说,思路新颖,值得进一步改进和验证。

||

### Overview

This strategy combines the Aroon indicator and the Absolute Strength Histogram (ASH) to identify market trends and potential trading opportunities. Aroon helps determine the strength and direction of trends, while ASH provides insights into the momentum strength. By combining these indicators, the strategy aims to capture profitable trades in Ethereum markets.

### Strategy Logic

The strategy uses two sets of parameters for the Aroon indicator:

- Long Positions: Aroon periods are 56 (upper) and 20 (lower) 
- Short Positions: Aroon periods are 17 (upper) and 55 (lower)

The ASH is calculated with a length of 9 bars using the closing price as the data source.

The strategy incorporates specific entry and exit rules:

1. Long Entry: A long position is initiated when the Aroon indicator crosses over the lower threshold, signaling a potential uptrend.
2. Long Exit: A long position is closed when the Aroon crosses back under the lower threshold.  
3. Short Entry: A short position is initiated when the Aroon crosses under the upper threshold, signaling a potential downtrend.
4. Short Exit: A short position is closed when the Aroon crosses back over the upper threshold.

### Advantage Analysis 

The main advantage of this strategy is the synergy from combining the two indicators. Aroon effectively gauges trend direction and strength. ASH provides additional momentum insights to aid with timing entry and exit signals.

Using two Aroon parameters allows flexibility in adapting to changing market conditions.

### Risk Analysis

The main limitations stem from the indicators themselves. Aroon struggles during range-bound markets and can generate false signals. ASH is also prone to overreactions in the short term.

Inappropriate parameter settings could also impact performance. The long/short periods of Aroon and length of ASH would need optimization to find the ideal combinations.

### Improvement Directions

Additional filters could be added, such as price breakouts or rising volumes, to avoid false signals during choppy conditions.

Different parameter combinations and weights could be tested to find optimal settings. Other indicators like RSI or KD could also complement the strategy.

### Conclusion

The strategy effectively combines the strengths of Aroon and ASH for dual-confirmation of trends and turning points. But parameters and indicator limitations still need refinement. The creative concept shows promise for further improvements and testing.

[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_int_1|56|length_upper_long|
|v_input_int_2|20|length_lower_long|
|v_input_int_3|17|length_upper_short|
|v_input_int_4|55|length_lower_short|
|v_input_1|9|Length|
|v_input_2_close|0|Source: close|high|low|open|hl2|hlc3|hlcc4|ohlc4|
|v_input_3|timestamp(01 Jan 2018 00:00)|Start Date|
|v_input_string_1|0|Trade Direction: Long|Short|Both|


> Source (PineScript)

``` pinescript
/*backtest
start: 2023-03-05 00:00:00
end: 2024-03-10 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// © IkkeOmar

//@version=5
strategy("Aroon and ASH strategy - ETHERIUM [IkkeOmar]", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, pyramiding=1, commission_value=0, slippage=2)


// AROON SETTINGS ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Inputs for longs 

length_upper_long = input.int(56, minval=15)
length_lower_long = input.int(20, minval=5)

// Inputs for shorts
//Aroon Short Side Inputs
length_upper_short = input.int(17, minval=10)
length_lower_short = input.int(55)

// ABSOLUTE STRENGTH HISTOGRAM SETTINGS ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
length = input(title='Length', defval=9)
src = input(title='Source', defval=close)




// CALCULATIONS: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Aroon
upper_long = 100 * (ta.highestbars(high, length_upper_long + 1) + length_upper_long) / length_upper_long
lower_long = 100 * (ta.lowestbars(low, length_lower_long + 1) + length_lower_long) / length_lower_long

upper_short = 100 * (ta.highestbars(high, length_upper_short + 1) + length_upper_short) / length_upper_short
lower_short = 100 * (ta.lowestbars(low, length_lower_short + 1) + length_lower_short) / length_lower_short

// Ahrens Moving Average
ahma = 0.0
ahma := nz(ahma[1]) + (src - (nz(ahma[1]) + nz(ahma[length])) / 2) / length



// CONDITIONS: ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


// Options that configure the backtest start date
startDate = input(title='Start Date', defval=timestamp('01 Jan 2018 00:00'))


// Option to select trade directions
tradeDirection = input.string(title='Trade Direction', options=['Long', 'Short', 'Both'], defval='Long')

// Translate input into trading conditions
longOK = tradeDirection == 'Long' or tradeDirection == 'Both'
shortOK = tradeDirection == 'Short' or tradeDirection == 'Both'


// Check if the close time of the current bar falls inside the date range
inDateRange = true

longCondition = ta.crossover(upper_long, lower_long) and inDateRange and lower_long >= 5 and longOK
longCloseCondition = ta.crossunder(upper_long, lower_long) and inDateRange

shortCondition = ta.crossunder(upper_short, lower_short) and inDateRange and shortOK
shortCloseCondition = ta.crossover(upper_short, lower_short) and inDateRange

// Start off with the initial states for the longs and shorts
var in_short_trade = false
var in_long_trade = false

var long_signal = false
var short_signal = false

if longCondition
    long_signal := true
if longCloseCondition
    long_signal := false
    
if shortCondition
    short_signal := true
if shortCloseCondition
    short_signal := false

// While no trades active and short condition is met, OPEN short
if true and in_short_trade == false and in_long_trade == false and shortCondition
    strategy.entry("short", strategy.short, when = shortCondition)
    in_short_trade := true
    in_long_trade := false

// While no trades and long condition is met, OPEN LONG
if true and in_short_trade == false and in_long_trade == false and longCondition
    strategy.entry("long", strategy.long, when = longCondition)
    in_long_trade := true
    in_short_trade := false

    
// WHILE short trade and long condition is met, CLOSE SHORT and OPEN LONG
if true and in_short_trade == true and in_long_trade == false and longCondition
    // strategy.close("short", when = longCondition)
    strategy.entry("long", strategy.long, when = longCondition)
    in_short_trade := false
    in_long_trade := true
    
    
// WHILE long trade and short condition is met, CLOSE LONG and OPEN SHORT
if true and in_short_trade == false and in_long_trade == true and shortCondition
    // strategy.close("long", when = shortCondition)
    strategy.entry("short", strategy.short, when = shortCondition)
    in_short_trade := true
    in_long_trade := false

// WHILE long trade and exit long condition is met, CLOSE LONG
// if short signal is active, OPEN SHORT
if true and in_short_trade == false and in_long_trade == true and longCloseCondition
    if short_signal
        strategy.entry("short", strategy.short, when = short_signal)
        in_long_trade := false
        in_short_trade := true
    else
        strategy.close("long", when = longCloseCondition)
        in_long_trade := false
        in_short_trade := false

// if in short trade only and exit short condition is met, close the short
// if long signal still active, OPEN LONG
if true and in_short_trade == true and in_long_trade == false and shortCloseCondition
    if long_signal
        strategy.entry("long", strategy.long, when = long_signal)
        in_short_trade := false
        in_long_trade := true
    else
        strategy.close("short", when = shortCloseCondition)
        in_short_trade := false
        in_long_trade := false





```

> Detail

https://www.fmz.com/strategy/444336

> Last Modified

2024-03-11 10:54:08
