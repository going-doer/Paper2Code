
> Name

双时间尺度动量策略-Dual-Timeframe-Momentum-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1ecdb2cabbd335138e9.png)

[trans]
#### 概述
该策略是一个双时间尺度动量策略。它通过在高级别时间周期上使用简单移动平均线(SMA)来判断趋势方向,在低级别时间周期上使用枢轴点(PivotLow和PivotHigh)来识别反转点。当高级别时间周期呈现上升趋势且低级别时间周期出现看涨枢轴点时开多,当高级别时间周期呈现下降趋势且低级别时间周期出现看跌枢轴点时开空。

#### 策略原理
该策略的主要原理是高级别时间周期的趋势方向会影响低级别时间周期的走势。当高级别时间周期呈现上升趋势时,低级别时间周期的回调更可能是买入机会;当高级别时间周期呈现下降趋势时,低级别时间周期的反弹更可能是做空机会。该策略利用简单移动平均线(SMA)来判断高级别时间周期的趋势方向,利用枢轴点(PivotLow和PivotHigh)来识别低级别时间周期的反转点。

#### 策略优势
1. 双时间尺度分析,利用了高级别时间周期对低级别时间周期的影响,增加了交易的成功概率。
2. 使用SMA判断趋势方向比较可靠,使用枢轴点捕捉反转点比较精准。
3. 参数可调,适应性强。用户可以根据自己的需求调整高低时间尺度、SMA的周期、枢轴点的参数等。
4. 逻辑清晰,容易理解和实现。

#### 策略风险
1. 趋势突变风险。如果高级别时间周期的趋势突然发生变化,低级别时间周期可能还没有反应过来,导致策略失效。
2. 参数设置风险。不恰当的参数设置可能导致策略表现不佳。比如SMA周期选择过短可能导致频繁交易,选择过长可能导致趋势判断滞后。
3. 极端行情风险。在极端行情下(如暴涨暴跌),该策略可能失效。因为这种行情下,低级别时间周期可能不遵循高级别时间周期的趋势。

#### 策略优化方向  
1. 增加趋势变化的判断。可以增加一些逻辑来判断高级别时间周期趋势是否发生变化,以便更快地调整低级别时间周期的交易。
2. 优化参数选择。可以使用一些参数优化的方法(如遗传算法、网格搜索等)来寻找最优参数组合。
3. 增加风险控制。可以增加一些风险控制的措施(如止损、仓位管理等)来降低极端行情下的损失。
4. 多因子融合。可以考虑将其他指标或因子(如波动率、成交量等)融入到该策略中,以提高策略的稳健性。

#### 总结
该双时间尺度动量策略利用了高低级别时间周期之间的联系,通过在高级别时间周期判断趋势方向,在低级别时间周期捕捉反转点,以此实现趋势跟随和反转交易。该策略逻辑清晰,优势明显,但同时也存在一些风险。未来可以从趋势变化判断、参数优化、风险控制、多因子融合等方面对该策略进行优化,以提高其适应性和稳健性。

|| 

#### Overview
This strategy is a dual timeframe momentum strategy. It determines the trend direction on the higher timeframe using a Simple Moving Average (SMA) and identifies reversal points on the lower timeframe using pivot points (PivotLow and PivotHigh). It enters long when the higher timeframe shows an uptrend and a bullish pivot point appears on the lower timeframe, and enters short when the higher timeframe shows a downtrend and a bearish pivot point appears on the lower timeframe.

#### Strategy Principles
The main principle of this strategy is that the trend direction of the higher timeframe will influence the movement of the lower timeframe. When the higher timeframe shows an uptrend, pullbacks on the lower timeframe are more likely to be buying opportunities; when the higher timeframe shows a downtrend, rebounds on the lower timeframe are more likely to be shorting opportunities. This strategy uses the Simple Moving Average (SMA) to determine the trend direction of the higher timeframe and pivot points (PivotLow and PivotHigh) to identify reversal points on the lower timeframe.

#### Strategy Advantages
1. Dual timeframe analysis, leveraging the impact of the higher timeframe on the lower timeframe, increases the probability of successful trades.
2. Using SMA to determine the trend direction is relatively reliable, and using pivot points to capture reversal points is relatively accurate.
3. Parameters are adjustable, making the strategy highly adaptable. Users can adjust the higher and lower timeframes, the period of the SMA, and the parameters of the pivot points according to their needs.
4. The logic is clear and easy to understand and implement.

#### Strategy Risks
1. Risk of trend change. If the trend of the higher timeframe suddenly changes, the lower timeframe may not have reacted yet, causing the strategy to fail.
2. Risk of parameter settings. Inappropriate parameter settings may lead to poor strategy performance. For example, choosing an SMA period that is too short may lead to frequent trading, while choosing one that is too long may lead to lagging trend judgments.
3. Risk of extreme market conditions. In extreme market conditions (such as sharp rises or falls), this strategy may fail because the lower timeframe may not follow the trend of the higher timeframe.

#### Strategy Optimization Directions
1. Add trend change detection. Logic can be added to determine whether the trend of the higher timeframe has changed in order to adjust trading on the lower timeframe more quickly.
2. Optimize parameter selection. Parameter optimization methods (such as genetic algorithms, grid search, etc.) can be used to find the optimal parameter combination.
3. Add risk control. Risk control measures (such as stop-loss, position management, etc.) can be added to reduce losses under extreme market conditions.
4. Multi-factor fusion. Other indicators or factors (such as volatility, volume, etc.) can be considered to be incorporated into the strategy to improve its robustness.

#### Summary
This dual timeframe momentum strategy leverages the connection between higher and lower timeframes, determining the trend direction on the higher timeframe and capturing reversal points on the lower timeframe to achieve trend following and reversal trading. The strategy has clear logic and obvious advantages, but also has some risks. In the future, the strategy can be optimized from aspects such as trend change detection, parameter optimization, risk control, and multi-factor fusion to improve its adaptability and robustness.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_int_1|20|Moving Average Period|
|v_input_source_1_close|0|Source: close|high|low|open|hl2|hlc3|hlcc4|ohlc4|
|v_input_timeframe_1|240|Resolution|
|v_input_int_2|5|Pivot Let Bars|
|v_input_int_3|2|Pivot Right Bars|


> Source (PineScript)

``` pinescript
/*backtest
start: 2023-04-19 00:00:00
end: 2024-04-24 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © Riester

//@version=5
strategy("Dual Timeframe Momentum", overlay=true, precision=6, pyramiding=0, initial_capital=1000, default_qty_type=strategy.percent_of_equity, default_qty_value=25.0, commission_value=0.05)

n = input.int(20, "Moving Average Period", minval=1)
src = input.source(close, "Source")
high_tf = input.timeframe("240", "Resolution")
pivot_l = input.int(5, "Pivot Let Bars")
pivot_r = input.int(2, "Pivot Right Bars")

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------
// Calculations
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

// 1. Define low and high timeframe prices
low_src = src
high_src = request.security(syminfo.tickerid, high_tf, src)

// 2. Use simple moving average to determine trend of higher timeframe (up or down)
high_tf_ma = ta.sma(high_src, n)
plot(high_tf_ma,  color=color.yellow)
high_tf_trend = high_tf_ma > high_tf_ma[1] ? 1 : -1

// 3. Use pivots to identify reversals on the low timeframe
low_tf_pl = ta.pivotlow(high_src, pivot_l, pivot_r)
plot(low_tf_pl, style=plot.style_line, linewidth=3, color= color.green, offset=-pivot_r)

low_tf_ph = ta.pivothigh(high_src, pivot_l, pivot_r)
plot(low_tf_ph, style=plot.style_line, linewidth=3, color= color.red, offset=-pivot_r)

bool long = low_tf_pl and high_tf_trend == 1
bool short = low_tf_ph and high_tf_trend == -1

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------
// Plots
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

// this message is an alert that can be sent to a webhook, which allows for simple automation if you have a server that listens to alerts and trades programmatically.
enter_long_alert = '{"side": "Long", "order": "Enter", "price": ' + str.tostring(open) + ', "timestamp": ' + str.tostring(timenow) + '}'
exit_long_alert = '{"side": "Long", "order": "Exit", "price": ' + str.tostring(open) + ', "timestamp": ' + str.tostring(timenow) + '}'

if long
    strategy.entry(id="Long", direction=strategy.long, limit=open, alert_message=enter_long_alert)

if short
    strategy.close(id="Long", comment="Close Long", alert_message=exit_long_alert)

```

> Detail

https://www.fmz.com/strategy/449451

> Last Modified

2024-04-25 17:33:02
