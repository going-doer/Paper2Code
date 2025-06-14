
> Name

Donchian突破交易策略-Donchian-Breakout-Trading-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/168d56ab283ab5cbf6b.png)

[trans]
#### 概述
Donchian突破交易策略是一个基于Donchian通道指标的交易系统。该策略的主要思路是通过Donchian通道上轨和下轨的突破来捕捉市场趋势,并采用固定风险收益比(RR)进行止盈止损。当价格突破Donchian通道上轨并创出相对于Donchian通道周期的新高时开多,突破下轨并创出新低时开空。同时,止损位置设置在Donchian通道中轨,止盈位置根据设定的风险收益比(RR)计算得出。

#### 策略原理
1. 计算Donchian通道:根据设定的Donchian通道周期(默认20),计算该周期内的最高价和最低价,分别作为Donchian通道的上轨和下轨,并计算上下轨的中点作为Donchian通道中轨。
2. 判断是否创出新高/新低:通过循环比较当前Donchian通道上轨和下轨与前几周期的上轨和下轨,判断当前是否创出了相对Donchian通道周期的新高或新低。若创出新高,则Donchian上轨显示为蓝色;若创出新低,则Donchian下轨显示为蓝色。
3. 突破开仓:当价格收盘突破蓝色Donchian上轨时开多头仓位,突破蓝色Donchian下轨时开空头仓位。即只有在创出新高/新低后的突破才有效。
4. 止盈止损:开仓时记录开仓价格和当前Donchian通道中轨价格,并计算两者价差。止损位置设置在Donchian通道中轨,止盈位置根据设定的风险收益比(默认5倍)和该价差计算得出。
5. 平仓:当价格触及止盈或止损价格时平仓。

#### 策略优势
1. 适合趋势市:Donchian突破策略通过突破上轨/下轨开仓,顺应市场趋势方向进行交易,在趋势行情中表现出色。
2. 新高/新低过滤:策略通过判断是否创出Donchian通道周期新高/新低,过滤掉部分噪音信号和假突破,提高开仓信号质量。
3. 固定风险收益比:每笔交易的止盈和止损位置基于固定的风险收益比,风险可控,有利于资金管理。
4. 参数简单:策略参数设置较为简单,主要为Donchian通道周期和风险收益比,优化和把控较为容易。

#### 策略风险
1. 幅度损失:策略止损位置为Donchian通道中轨,在趋势不明朗或震荡行情中,可能出现单次交易大幅损失的情况。
2. 频繁交易:若Donchian通道周期设置较小,可能导致频繁开平仓,增加交易成本。
3. 趋势转折:在趋势转折期,策略可能出现连续多次止损的情况。
4. 参数敏感:策略表现对参数设置较为敏感,需要根据不同市场特点和行情周期进行参数优化。

#### 策略优化方向
1. 动态止损:根据价格走势、波动率等实时调整止损位置,如采用ATR作为止损参考,降低单次交易风险。
2. 趋势过滤:加入趋势判断指标如移动平均线,只在明确趋势方向时开仓,提高信号质量。
3. 结合其他指标:如结合RSI、MACD等动量指标,综合评估开仓时机。
4. 仓位管理:根据市场趋势强度、波动率等动态调整仓位大小,控制整体风险。
5. 参数自适应:采用机器学习等方法,自适应优化参数设置。

#### 总结
Donchian突破交易策略是一个基于经典Donchian通道指标的趋势跟踪交易系统。通过Donchian通道上下轨的突破及新高/新低的判断来开仓,止盈止损基于固定风险收益比。该策略逻辑简单,适合趋势性市场。但在震荡市中表现一般,且对参数设置敏感。可通过引入动态止损、趋势过滤、仓位管理等方式进一步优化,提高策略稳健性。

|| 

#### Overview
The Donchian Breakout Trading Strategy is a trading system based on the Donchian Channel indicator. The main idea of this strategy is to capture market trends by breaking through the upper and lower bands of the Donchian Channel, and to use a fixed Risk Reward Ratio (RR) for take profit and stop loss. When the price breaks above the upper band of the Donchian Channel and creates a new high relative to the Donchian Channel period, it goes long; when it breaks below the lower band and creates a new low, it goes short. At the same time, the stop loss is set at the middle band of the Donchian Channel, and the take profit is calculated based on the set Risk Reward Ratio.

#### Strategy Principle
1. Calculate Donchian Channel: Based on the set Donchian Channel period (default 20), calculate the highest and lowest prices within that period as the upper and lower bands of the Donchian Channel respectively, and calculate the midpoint of the upper and lower bands as the middle band of the Donchian Channel.
2. Determine if a new high/low is created: By looping and comparing the current Donchian Channel upper and lower bands with the upper and lower bands of the previous few periods, determine if a new high or low relative to the Donchian Channel period is created. If a new high is created, the Donchian upper band is displayed in blue; if a new low is created, the Donchian lower band is displayed in blue.
3. Breakout entry: When the closing price breaks above the blue Donchian upper band, it enters a long position; when it breaks below the blue Donchian lower band, it enters a short position. That is, only breakouts that occur after a new high/low is created are valid.
4. Take profit and stop loss: When opening a position, record the entry price and the current Donchian Channel middle band price, and calculate the price difference between the two. The stop loss is set at the Donchian Channel middle band, and the take profit is calculated based on the set Risk Reward Ratio (default 5 times) and the price difference.
5. Close position: When the price reaches the take profit or stop loss price, the position is closed.

#### Strategy Advantages
1. Suitable for trending markets: The Donchian Breakout Strategy enters positions by breaking through the upper/lower bands, following the direction of the market trend, and performs well in trending markets.
2. New high/low filtering: The strategy filters out some noise signals and false breakouts by determining whether a new high/low is created within the Donchian Channel period, improving the quality of entry signals.
3. Fixed Risk Reward Ratio: The take profit and stop loss positions for each trade are based on a fixed Risk Reward Ratio, making the risk controllable and conducive to money management.
4. Simple parameters: The strategy parameters are relatively simple to set, mainly the Donchian Channel period and Risk Reward Ratio, making optimization and control easier.

#### Strategy Risks
1. Magnitude loss: The stop loss position of the strategy is the Donchian Channel middle band. In unclear trends or fluctuating markets, there may be situations where a single transaction suffers a large loss.
2. Frequent trading: If the Donchian Channel period is set too small, it may lead to frequent opening and closing of positions, increasing transaction costs.
3. Trend reversal: During trend reversals, the strategy may experience multiple consecutive stop losses.
4. Parameter sensitivity: The strategy performance is sensitive to parameter settings and needs to be optimized based on different market characteristics and trading cycles.

#### Strategy Optimization Directions
1. Dynamic stop loss: Adjust the stop loss position in real-time based on price movements, volatility, etc., such as using ATR as a stop loss reference to reduce single transaction risk.
2. Trend filtering: Add trend judgment indicators such as moving averages and only open positions when the trend direction is clear to improve signal quality.
3. Combine with other indicators: Combine with momentum indicators such as RSI and MACD to comprehensively evaluate the timing of opening positions.
4. Position management: Dynamically adjust position sizes based on market trend strength, volatility, etc., to control overall risk.
5. Parameter adaptation: Use machine learning and other methods to adaptively optimize parameter settings.

#### Summary
The Donchian Breakout Trading Strategy is a trend-following trading system based on the classic Donchian Channel indicator. It opens positions through breakouts of the upper and lower bands of the Donchian Channel and judgments of new highs/lows, with take profit and stop loss based on a fixed Risk Reward Ratio. The strategy has a simple logic and is suitable for trending markets. However, it performs poorly in fluctuating markets and is sensitive to parameter settings. It can be further optimized through the introduction of dynamic stop losses, trend filtering, position management, etc., to improve the robustness of the strategy.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_int_1|20|don_length|
|v_input_float_1|5|RR Profit Target|


> Source (PineScript)

``` pinescript
/*backtest
start: 2023-04-23 00:00:00
end: 2024-04-28 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//---------------------------------------------//
// This source code is subject to the terms of 
// the Mozilla Public License 2.0 at 
// https://mozilla.org/MPL/2.0/
// © Dillon_Grech
//---------------------------------------------//

//---------------------------------------------//
// Simple donchian channel break out strategy
// which only enters trades when price closes
// above donchian upper and creates new high 
// (long) or price closes below donchian lower
// and creates new low, relative to the donchian
// length. This is indicated by the donchian
// upper and lower color (blue). Stop loss is
// located at donchian basis and take profit
// is set at Risk Reward (RR) profit target.
//---------------------------------------------//
//@version=5
strategy("Donchian New High/Low Strategy [Dillon Grech]", overlay=true)

//---------------------------------------------//

//---------------------------------------------//
//INDICATOR 1 - Donchian New High Low Price Close
don_length = input.int(20, minval = 1)
don_lower  = ta.lowest(don_length)
don_upper  = ta.highest(don_length)
don_basis  = math.avg(don_upper, don_lower)

//loop
don_lower_upper  = true
don_higher_lower = true
for i = 0 to don_length - 1
    //Check for higher high over don_length
    if don_upper > don_upper[i]
        don_lower_upper := false
    //Check for lower low over don_length
    if don_lower < don_lower[i]
        don_higher_lower := false

//Plot
c_ora = color.orange
c_blu = color.blue
c_gra = color.gray
color_basis = c_ora
color_upper = don_lower_upper  ? c_blu : c_gra
color_lower = don_higher_lower ? c_blu : c_gra
plot(don_basis,     "Don Basis", color_basis, 2)
u = plot(don_upper, "Don Upper", color_upper, 2)
l = plot(don_lower, "Don Lower", color_lower, 2)

//Conditions
Ind_1_L = ta.crossover(close, don_upper[1]) and 
   don_lower_upper[1]
Ind_1_S = ta.crossunder(close,don_lower[1]) and 
   don_higher_lower[1]
//---------------------------------------------//

//---------------------------------------------//
//ENTRY CONDITIONS
entry_long  = strategy.position_size<=0 and
   Ind_1_L
entry_short = strategy.position_size>=0 and
   Ind_1_S

if(entry_long)
    strategy.entry("Long Entry", strategy.long)
if(entry_short)
    strategy.entry("Short Entry", strategy.short)
//---------------------------------------------/

//---------------------------------------------//
//TAKE PROFIT AND STOP LOSS CONDITIONS
profit_RR = input.float(5.0,"RR Profit Target")

//Store Price on new entry signal
entry_price = strategy.opentrades.entry_price(
   strategy.opentrades-1)

//Store Donchain Channel Basis
entry_don_basis = float(0.0)
if entry_long or entry_short
    entry_don_basis := don_basis
else
    entry_don_basis := entry_don_basis[1]

//Get stop loss distance
stop_distance = math.abs(entry_price -
   entry_don_basis)
stop_L   = entry_price - stop_distance
profit_L = entry_price + stop_distance*profit_RR
stop_S   = entry_price + stop_distance
profit_S = entry_price - stop_distance*profit_RR

//Plot TP and SL
plot(entry_long or entry_short ? na :
   strategy.position_size > 0 ? profit_L : na,
   color=color.lime, style=plot.style_linebr,
   linewidth=2)
plot(entry_long or entry_short ? na :
   strategy.position_size > 0 ? stop_L : na,
   color=color.red,  style=plot.style_linebr,
   linewidth=2)
plot(entry_long or entry_short ? na : 
   strategy.position_size < 0 ? profit_S : na,
   color=color.lime, style=plot.style_linebr,
   linewidth=2)
plot(entry_long or entry_short ? na :
   strategy.position_size < 0 ? stop_S : na,
   color=color.red,  style=plot.style_linebr,
   linewidth=2)

//Exit long trades
strategy.exit(id = 'Exit Long', 
   from_entry ='Long Entry', 
   stop = stop_L, limit = profit_L)
strategy.exit(id = 'Exit Short', 
   from_entry ='Short Entry', 
   stop = stop_S, limit = profit_S)
//---------------------------------------------//
```

> Detail

https://www.fmz.com/strategy/449818

> Last Modified

2024-04-29 14:56:35
