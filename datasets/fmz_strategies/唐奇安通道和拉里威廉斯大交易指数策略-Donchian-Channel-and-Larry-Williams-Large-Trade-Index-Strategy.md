
> Name

唐奇安通道和拉里威廉斯大交易指数策略-Donchian-Channel-and-Larry-Williams-Large-Trade-Index-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/ca642dd9163504f03e.png)

[trans]
#### 概述
该策略结合唐奇安通道、拉里威廉斯大交易指数(LWTI)和成交量移动平均线三个指标来进行交易。当价格突破唐奇安通道上轨,LWTI为绿色,成交量大于移动平均线时做多;当价格突破唐奇安通道下轨,LWTI为红色,成交量大于移动平均线时做空。策略在开仓后,当价格触及止损或止盈位置,或者价格回到唐奇安通道中轨时平仓。为了防止在同一趋势方向内重复开仓,该策略使用一个交易计数器,只有当价格突破唐奇安通道中轨后,才允许再次开仓。

#### 策略原理
1. 唐奇安通道:当价格突破通道上轨时产生做多信号,突破下轨时产生做空信号。
2. 拉里威廉斯大交易指数:LWTI颜色为绿色时只允许做多,红色时只允许做空。
3. 成交量:当前成交量大于成交量移动平均线时,才允许开仓。
4. 交易计数器:防止在同一趋势方向内重复开仓,只有当价格突破唐奇安通道中轨后,才允许再次开仓。
5. 止盈止损:开仓时根据ATR计算止盈止损距离,止盈距离为止损距离乘以风险收益比。

#### 策略优势
1. 结合多个指标共同确认交易信号,可以有效过滤伪信号,提高信号质量。
2. 动态止盈止损 - 根据波动率动态调整止盈止损距离,可以更好地适应市场变化。
3. 通过交易计数器防止在同一趋势内重复开仓,控制交易频率。
4. 风险收益比止盈 - 根据预设的风险收益比设置止盈位置,让盈利空间大于风险。

#### 策略风险
1. 参数风险 - 不同参数设置对策略表现影响很大,需要根据不同市场特点和周期进行优化。
2. 震荡市风险 - 在震荡市场环境下,频繁的波动可能导致策略频繁开平仓,表现不佳。
3. 趋势风险 - 如果趋势持续性不强,可能出现频繁开平仓,导致亏损加大。
4. 黑天鹅风险 - 极端行情下指标可能失效,策略表现不佳。

#### 策略优化方向 
1. 针对不同品种和周期进行参数优化,找到最佳参数组合。
2. 增加趋势过滤条件,比如使用均线或者动量指标,只在趋势明确时开仓,减少震荡环境下的交易次数。
3. 针对震荡市环境,可以考虑使用范围突破策略。
4. 对止盈止损逻辑进行优化,引入移动止盈或者尾随止损等方法。
5. 针对极端行情,可以考虑引入固定资金管理和最大回撤限制等措施。

#### 总结
唐奇安通道和拉里威廉斯大交易指数策略是一个经典的趋势跟踪交易策略。通过唐奇安通道捕捉趋势方向,使用LWTI、成交量等指标过滤信号,动态止盈止损,风险控制严格,整体是一个具有稳健收益的策略框架。但需要注意的是,该策略对参数比较敏感,在震荡市环境下表现不佳,建议在趋势型市场中使用。在实际应用中,还需要根据交易标的和市场特点,对参数和逻辑进行进一步优化,并配合严格的资金管理,才能取得良好稳定的回报。

|| 

#### Overview
This strategy combines three indicators - Donchian Channel, Larry Williams Large Trade Index (LWTI), and Volume Moving Average to generate trading signals. It enters a long position when the price breaks above the upper band of the Donchian Channel, LWTI is green, and volume is greater than the moving average. It enters a short position when the price breaks below the lower band of the Donchian Channel, LWTI is red, and volume is greater than the moving average. The strategy exits positions when the price reaches the stop loss or take profit levels, or when the price returns to the middle band of the Donchian Channel. To prevent repeated entries in the same trend direction, the strategy employs a trade counter that only allows new entries after the price crosses the middle band of the Donchian Channel.

#### Strategy Principle
1. Donchian Channel: A long signal is generated when the price breaks above the upper band, and a short signal is generated when the price breaks below the lower band.
2. Larry Williams Large Trade Index: Long positions are only allowed when the LWTI color is green, and short positions are only allowed when it is red.
3. Volume: Entries are only allowed when the current volume is greater than the volume moving average.
4. Trade Counter: To prevent repeated entries in the same trend direction, new entries are only allowed after the price crosses the middle band of the Donchian Channel.
5. Stop Loss and Take Profit: Upon entry, stop loss and take profit distances are calculated based on ATR, with the take profit distance being the stop loss distance multiplied by the risk-reward ratio.

#### Strategy Advantages
1. The combination of multiple indicators to confirm trading signals can effectively filter out false signals and improve signal quality.
2. Dynamic stop loss and take profit - Adjusting stop loss and take profit distances based on volatility allows the strategy to better adapt to market changes.
3. The trade counter prevents repeated entries in the same trend, controlling trading frequency.
4. Risk-reward ratio based take profit - Setting the take profit level based on a predetermined risk-reward ratio allows profit potential to exceed risk.

#### Strategy Risks
1. Parameter risk - Strategy performance is greatly affected by different parameter settings, requiring optimization based on different market characteristics and timeframes.
2. Choppy market risk - In choppy market conditions, frequent fluctuations may cause the strategy to enter and exit positions frequently, resulting in poor performance.
3. Trend risk - If the trend lacks persistence, frequent entries and exits may occur, leading to increased losses.
4. Black swan risk - In extreme market conditions, indicators may fail, resulting in poor strategy performance.

#### Strategy Optimization Directions
1. Optimize parameters for different instruments and timeframes to find the best parameter combinations.
2. Add trend filtering conditions, such as using moving averages or momentum indicators, to only enter positions when the trend is clear, reducing the number of trades in choppy environments.
3. Consider using range breakout strategies for choppy market conditions.
4. Optimize stop loss and take profit logic by introducing trailing stops or other methods.
5. For extreme market conditions, consider introducing fixed money management and maximum drawdown limits.

#### Summary
The Donchian Channel and Larry Williams Large Trade Index strategy is a classic trend-following trading strategy. It captures trend direction using the Donchian Channel, filters signals using LWTI, volume, and other indicators, and employs dynamic stop loss and take profit with strict risk control. Overall, it is a strategy framework with the potential for steady returns. However, it is important to note that the strategy is sensitive to parameters and performs poorly in choppy market conditions. It is recommended for use in trending markets. In practical application, further optimization of parameters and logic based on trading instruments and market characteristics, along with strict money management, is necessary to achieve good and stable returns.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_1|true|(?Donchian Settings)Use Donchian Channel?|
|v_input_int_1|96|don_length|
|v_input_2|true|(?LWTI Settings)Use LWTI?|
|v_input_int_2|25|Period|
|v_input_bool_1|false|Smooth LWPI?|
|v_input_string_1|0|Smoothing Type: SMA|WMA|RMA|EMA|
|v_input_int_3|20|Smoothing Period|
|v_input_3|true|(?Volume Settings)Use Volume?|
|v_input_int_4|30|Volume MA Period|
|v_input_bool_2|false|(?Plot Settings)Plot Trade Position Counter?|
|v_input_4|true|(?Risk Settings)Use Stop Loss?|
|v_input_5|true|Use Take Profit?|
|v_input_float_1|2|Risk Reward Profit Target|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-04-04 00:00:00
end: 2024-04-11 00:00:00
period: 1m
basePeriod: 1m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This source code is subject to the terms of the Mozilla Public License 2.0 
//at https://mozilla.org/MPL/2.0/
// © DillonGrech
//
//This is a Donchian Channel Trading Strategy which was found through the 
//YouTube channel TradingLab. 
//
//This Strategy uses the Donchian Channel, Larry Williams Large Trade Index,
//and Volume with Moving Average indicators for long and short trades.
//
//Strategy will enter based off indicator entry conditions and will not
//re-enter a trade until the price crosses through the Donchian Channel
//basis line (to prevent re-entering trades in same trend). Strategy will
//exit at stop loss or take profit, or when price crosses donchian basis.
//
//The strategy has been coded by Dillon Grech as part of his YouTube channel
//and a detailed video can be found on his channel at:
//https://www.youtube.com/c/DillonGrech
//https://www.youtube.com/watch?v=5IFe2Vjf61Y
//Source code and template files can be found on his GitHub at:
//https://github.com/Dillon-Grech
//==============================================================================
//@version=5
strategy("Donchian Channel Strategy [DillonGrech]", overlay=true, margin_long=100, margin_short=100)

//==============================================================================
//DONCHIAN CHANNEL
//==============================================================================
//Allow user to select whether they would like to use indicator
Don_Input = input(true, title='Use Donchian Channel?', group = "Donchian Settings")

//Indicator
don_length = input.int(96, minval = 1, group = "Donchian Settings")
don_lower  = ta.lowest(don_length)
don_upper  = ta.highest(don_length)
don_basis  = math.avg(don_upper, don_lower)
plot(don_basis,     "Don Basis", color = #FF6D00)
u = plot(don_upper, "Don Upper", color = #2962FF)
l = plot(don_lower, "Don Lower", color = #2962FF)
fill(u, l, color = color.rgb(33, 150, 243, 95), title = "Background")

//Conditions - Enter trades when there is a cross of price and previous donchian channel value
Ind_1_L = Don_Input == false ? false : ta.crossover(close,don_upper[1])
Ind_1_S = Don_Input == false ? false : ta.crossunder(close,don_lower[1])

//==============================================================================
//LARRY WILLIAMS LARGE TRADE INDEX (LWTI) - LOXX
//==============================================================================
//Allow user to select whether they would like to use indicator
LWTI_Input = input(true, title='Use LWTI?', group = "LWTI Settings")

//Indicator
greencolor = #2DD204
redcolor = #D2042D 

variant(type, src, len) =>
    sig = 0.0
    if type == "SMA"
        sig := ta.sma(src, len) 
    else if type == "EMA"
        sig := ta.ema(src, len) 
    else if type == "WMA"
        sig := ta.wma(src, len)   
    else if type == "RMA"
        sig := ta.rma(src, len)  
    sig
    
LWTI_per = input.int(25, "Period", group = "LWTI Settings")
LWTI_smthit = input.bool(false, "Smooth LWPI?", group = "LWTI Settings")
LWTI_type = input.string("SMA", "Smoothing Type", options = ["EMA", "WMA", "RMA", "SMA"], group = "LWTI Settings")
LWTI_smthper = input.int(20, "Smoothing Period", group = "LWTI Settings")

LWTI_ma = ta.sma(close - nz(close[LWTI_per]), LWTI_per)
LWTI_atr = ta.atr(LWTI_per)
LWTI_out = LWTI_ma/LWTI_atr * 50 + 50
LWTI_out := LWTI_smthit ? variant(LWTI_type, LWTI_out, LWTI_smthper) : LWTI_out

LWTI_colorout = LWTI_out > 50 ? greencolor : redcolor

//Conditions - Enter on color of indicator
Ind_2_L = LWTI_Input == false ? true : LWTI_colorout == greencolor
Ind_2_S = LWTI_Input == false ? true : LWTI_colorout == redcolor

//==============================================================================
//VOLUME INDICATOR
//==============================================================================
//Allow user to select whether they would like to use indicator
Vol_Input = input(true, title='Use Volume?', group = "Volume Settings")

//Indicator
Vol_Ma_Period = input.int(30,"Volume MA Period", group = "Volume Settings")
Vol_Ma = ta.sma(volume,Vol_Ma_Period)

//Conditions - Enter when volume is greater than moving average
Ind_3_L = Vol_Input == false ? true : volume > Vol_Ma
Ind_3_S = Vol_Input == false ? true : volume > Vol_Ma

//==============================================================================
//DONCHIAN CHANNEL TRADE COUNTER
//==============================================================================
//Stores whether a trade has been taken, and resets when there is a cross of price and donchain basis
Trade_Counter = float(0)
Don_Basis_Cross = ta.cross(don_basis[1], close)
if strategy.position_size!=0
    Trade_Counter := 1
else if Don_Basis_Cross
    Trade_Counter := 0
else 
    Trade_Counter := Trade_Counter[1]

Plot_Trade_Counter = input.bool(false, "Plot Trade Position Counter?", group = "Plot Settings")
plotchar(Plot_Trade_Counter and Trade_Counter == 0 ? true : false, color = na, text = '0')
plotchar(Plot_Trade_Counter and Trade_Counter == 1 ? true : false, color = na, text = '1')

//==============================================================================
//ENTRY CONDITIONS
//==============================================================================
entry_long  = strategy.position_size<=0 and Ind_1_L and Ind_2_L and Ind_3_L and Trade_Counter[1] == 0
entry_short = strategy.position_size>=0 and Ind_1_S and Ind_2_S and Ind_3_S and Trade_Counter[1] == 0

if(entry_long)
    strategy.entry("Long Entry", strategy.long)
if(entry_short)
    strategy.entry("Short Entry", strategy.short)

//==============================================================================
// TAKE PROFIT AND STOP LOSS CONDITIONS
//==============================================================================
Stop_Input   = input(true, title='Use Stop Loss?', group = "Risk Settings")
Profit_Input = input(true, title='Use Take Profit?', group = "Risk Settings")
Profit_RR = input.float(2.0,"Risk Reward Profit Target", group = "Risk Settings")

//Store Price on new entry signal
Entry_Price = strategy.opentrades.entry_price(strategy.opentrades - 1)

//Store Donchain Channel Basis value on new entry signal
Entry_Don_Basis = float(0.0)
if strategy.position_size == 0 or entry_long or entry_short
    Entry_Don_Basis := don_basis
else
    Entry_Don_Basis := Entry_Don_Basis[1]

//Get stop loss distance
Stop_Distance = math.abs(Entry_Price - Entry_Don_Basis)*1.02

//For Long Trades, find the stop loss level
Stop_L = float(0.0)
if Stop_Input == true
    Stop_L := Entry_Price - Stop_Distance
else
    na

//For Long Trades, find the profit level
Profit_L = float(0.0)
if Profit_Input == true
    Profit_L := Entry_Price + Stop_Distance*Profit_RR
else
    na

//For Short Trades, find the stop loss level
Stop_S = float(0.0)
if Stop_Input == true
    Stop_S   := Entry_Price + Stop_Distance
else
    na

//For Short Trades, find the profit level
Profit_S = float(0.0)
if Profit_Input == true
    Profit_S := Entry_Price - Stop_Distance*Profit_RR
else
    na

//Plot profit and stop loss levels for long and short trades
plot(strategy.position_size > 0 ? Profit_L : na, color=color.lime, style=plot.style_linebr, linewidth=2)
plot(strategy.position_size > 0 ? Stop_L : na,   color=color.red,  style=plot.style_linebr, linewidth=2)
plot(strategy.position_size < 0 ? Profit_S : na, color=color.lime, style=plot.style_linebr, linewidth=2)
plot(strategy.position_size < 0 ? Stop_S : na,   color=color.red,  style=plot.style_linebr, linewidth=2)

//==============================================================================
//EXIT ORDERS
//==============================================================================
//Exit long trades
if Stop_Input
    strategy.exit(id = 'Exit Long', from_entry ='Long Entry', comment='Long Stop',  stop = Stop_L)

if Profit_Input
    strategy.exit(id = 'Exit Long', from_entry ='Long Entry', comment='Long Profit', limit = Profit_L)

//Exit short trades
if Stop_Input
    strategy.exit(id = 'Exit Short', from_entry ='Short Entry', comment='Short Stop', stop = Stop_S)

if Profit_Input
    strategy.exit(id = 'Exit Short', from_entry ='Short Entry', comment='Short Profit', limit = Profit_S)

//==============================================================================
//CLOSE ORDERS
//==============================================================================
exit_long  = close < don_basis
exit_short = close > don_basis

if(exit_long)
    strategy.close("Long Entry", comment='Long Close', qty_percent=100)
if(exit_short)
    strategy.close("Short Entry", comment='Short Close', qty_percent=100)
```

> Detail

https://www.fmz.com/strategy/448079

> Last Modified

2024-04-12 17:27:25
