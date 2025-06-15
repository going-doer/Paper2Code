
> Name

趋势跟踪型双均线交叉策略结合SMA趋势过滤系统-Trend-Following-Dual-EMA-Crossover-Strategy-with-SMA-Trend-Filtering-System

> Author

ianzeng123

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/2d8743e88586f0e22b480.png)
![IMG](https://www.fmz.com/upload/asset/2d98af12c002226a558e3.png)



[trans]
#### 概述
该策略是一个结合了移动平均线(MA)交叉和趋势跟踪的量化交易系统。它使用15周期简单移动平均线(SMA)作为趋势过滤器,同时利用9周期和21周期指数移动平均线(EMA)的交叉来产生交易信号。该策略采用了严格的入场条件和固定的1:4风险收益比来管理风险。

#### 策略原理
策略的核心逻辑基于以下几个关键要素:
1. 趋势确认:使用15周期SMA作为主要趋势判断指标。价格在15SMA之上视为上升趋势,反之为下降趋势。
2. 交易信号:通过9EMA和21EMA的交叉来触发交易信号。当9EMA上穿21EMA且满足其他条件时产生做多信号;当9EMA下穿21EMA且满足其他条件时产生做空信号。
3. 确认条件:做多要求出现两根连续的阳线,且两个EMA都位于15SMA之上;做空要求出现阴线,且两个EMA都位于15SMA之下。
4. 风险管理:系统自动根据入场点位计算止损和获利目标,采用1:4的风险收益比设置。

#### 策略优势
1. 趋势跟踪能力强:通过15SMA的趋势过滤机制,能够有效避免在横盘或逆势行情中交易。
2. 多重确认机制:结合了均线交叉、蜡烛图形态和趋势确认等多重条件,降低假信号风险。
3. 风险管理完善:固定的风险收益比和自动止损止盈设置,有利于长期稳定运行。
4. 视觉反馈清晰:系统提供了清晰的视觉指示,包括交易信号标记和止损止盈水平显示。

#### 策略风险
1. 滞后性风险:移动平均线本质上是滞后指标,可能在市场快速转向时反应不及时。
2. 假突破风险:在横盘市场中可能产生虚假的交叉信号。
3. 固定风险比例的局限:1:4的固定风险收益比可能不适合所有市场环境。
4. 连续损失风险:在震荡市场中可能出现连续的止损。

#### 策略优化方向
1. 动态周期优化:可以根据市场波动率自动调整移动平均线周期。
2. 引入波动率过滤:添加ATR或其他波动率指标来优化入场时机。
3. 动态风险管理:根据市场条件动态调整风险收益比。
4. 增加市场环境判断:引入趋势强度指标来优化交易条件。

#### 总结
这是一个设计合理、逻辑严谨的趋势跟踪策略。通过结合多重技术指标和严格的风险管理,该策略具有良好的实用性。虽然存在一些固有的风险,但通过建议的优化方向可以进一步提升策略的稳定性和盈利能力。策略特别适合在趋势明显的市场中应用,建议在中长期时间周期上使用。

|| 

#### Overview
This strategy is a quantitative trading system that combines moving average (MA) crossovers with trend following. It utilizes a 15-period Simple Moving Average (SMA) as a trend filter, while using the crossover of 9-period and 21-period Exponential Moving Averages (EMA) to generate trading signals. The strategy employs strict entry conditions and a fixed 1:4 risk-reward ratio for risk management.

#### Strategy Principles
The core logic of the strategy is based on the following key elements:
1. Trend Confirmation: Uses 15-period SMA as the primary trend indicator. Price above 15 SMA indicates uptrend, while below indicates downtrend.
2. Trading Signals: Triggers trades through 9 EMA and 21 EMA crossovers. Long signals are generated when 9 EMA crosses above 21 EMA with other conditions met; short signals when 9 EMA crosses below 21 EMA with conditions met.
3. Confirmation Conditions: Long entries require two consecutive bullish candles with both EMAs above 15 SMA; short entries require a bearish candle with both EMAs below 15 SMA.
4. Risk Management: System automatically calculates stop-loss and profit targets using a 1:4 risk-reward ratio.

#### Strategy Advantages
1. Strong Trend Following Capability: The 15 SMA trend filtering mechanism effectively avoids trading in ranging or counter-trend markets.
2. Multiple Confirmation Mechanism: Combines moving average crossovers, candlestick patterns, and trend confirmation to reduce false signals.
3. Comprehensive Risk Management: Fixed risk-reward ratio and automatic stop-loss/profit targets support long-term stability.
4. Clear Visual Feedback: System provides clear visual indicators including trade signal markers and stop-loss/profit target levels.

#### Strategy Risks
1. Lag Risk: Moving averages are inherently lagging indicators, potentially slow to react to rapid market reversals.
2. False Breakout Risk: May generate false crossover signals in ranging markets.
3. Fixed Risk Ratio Limitations: The 1:4 fixed risk-reward ratio may not suit all market conditions.
4. Consecutive Loss Risk: May experience consecutive stops in choppy markets.

#### Strategy Optimization Directions
1. Dynamic Period Optimization: Automatically adjust moving average periods based on market volatility.
2. Volatility Filtering: Add ATR or other volatility indicators to optimize entry timing.
3. Dynamic Risk Management: Adjust risk-reward ratio based on market conditions.
4. Enhanced Market Context: Incorporate trend strength indicators to optimize trading conditions.

#### Summary
This is a well-designed, logically rigorous trend following strategy. Through the combination of multiple technical indicators and strict risk management, the strategy demonstrates good practicality. While inherent risks exist, the suggested optimization directions can further enhance strategy stability and profitability. The strategy is particularly suitable for trending markets and recommended for medium to long-term timeframes.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-02-21 00:00:00
end: 2024-12-19 00:00:00
period: 4d
basePeriod: 4d
exchanges: [{"eid":"Binance","currency":"ETH_USDT"}]
*/

//@version=5
strategy("EMA Crossover Strategy with 15 SMA Trend", overlay=true, margin_long=100, margin_short=100)

// Calculate Indicators
sma15 = ta.sma(close, 15)
ema9 = ta.ema(close, 9)
ema21 = ta.ema(close, 21)

// Trend Detection
uptrend = close > sma15
downtrend = close < sma15

// Crossover Conditions
goldenCross = ta.crossover(ema9, ema21)
deathCross = ta.crossunder(ema9, ema21)

// Candle Conditions
twoBullish = (close > open) and (close[1] > open[1])
bearishCandle = (close < open)

// Entry Conditions
longCondition = goldenCross and uptrend and twoBullish and (ema9 > sma15) and (ema21 > sma15)
shortCondition = deathCross and downtrend and bearishCandle and (ema9 < sma15) and (ema21 < sma15)

// Risk Management
var float longStop = na
var float longTarget = na
var float shortStop = na
var float shortTarget = na

if longCondition
    longStop := low
    longTarget := close + 4*(close - longStop)
    strategy.entry("Long", strategy.long)
    strategy.exit("Long Exit", "Long", stop=longStop, limit=longTarget)

if shortCondition
    shortStop := high
    shortTarget := close - 4*(shortStop - close)
    strategy.entry("Short", strategy.short)
    strategy.exit("Short Exit", "Short", stop=shortStop, limit=shortTarget)

// Visual Elements
plot(sma15, "15 SMA", color=color.orange)
plot(ema9, "9 EMA", color=color.blue)
plot(ema21, "21 EMA", color=color.red)

// Plot trading levels
plot(longCondition ? longStop : na, "Long Stop", color=color.red, style=plot.style_linebr)
plot(longCondition ? longTarget : na, "Long Target", color=color.green, style=plot.style_linebr)
plot(shortCondition ? shortStop : na, "Short Stop", color=color.red, style=plot.style_linebr)
plot(shortCondition ? shortTarget : na, "Short Target", color=color.green, style=plot.style_linebr)

// Signal Markers
plotshape(longCondition, "Buy", shape.triangleup, location.belowbar, color=color.green, size=size.small)
plotshape(shortCondition, "Sell", shape.triangledown, location.abovebar, color=color.red, size=size.small)
```

> Detail

https://www.fmz.com/strategy/483123

> Last Modified

2025-02-21 14:35:29
