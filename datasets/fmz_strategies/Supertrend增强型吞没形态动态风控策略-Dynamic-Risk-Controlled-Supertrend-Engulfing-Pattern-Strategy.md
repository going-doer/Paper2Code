
> Name

Supertrend增强型吞没形态动态风控策略-Dynamic-Risk-Controlled-Supertrend-Engulfing-Pattern-Strategy

> Author

ianzeng123

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/2d8e5f91456df5f57df36.png)
![IMG](https://www.fmz.com/upload/asset/2d996e64e1bc9e388b1fa.png)




[trans]
#### 概述
这是一个结合了Supertrend指标和吞没形态的高级交易策略。策略通过识别市场中的吞没形态蜡烛图模式,并结合Supertrend指标的趋势方向确认,实现精准的交易信号筛选。该策略还整合了动态的止损和获利设置,有效控制风险的同时保证盈利空间。

#### 策略原理
策略主要基于以下核心原理:
1. 使用ATR(平均真实波幅)计算Supertrend指标,用于确定总体市场趋势。
2. 通过设定无聊蜡烛阈值(Boring Candle Threshold)和吞没蜡烛阈值(Engulfing Candle Threshold)来筛选有效的吞没形态。
3. 在Supertrend趋势方向与吞没形态方向一致时才开仓。
4. 采用动态的止损和获利点位设置,基于开仓价格按比例计算。
5. 使用策略仓位管理,确保同一时间只持有一个交易方向。

#### 策略优势
1. 信号质量控制严格,通过双重确认(趋势+形态)提高准确率。
2. 引入无聊蜡烛和吞没蜡烛阈值概念,有效过滤虚假信号。
3. 基于ATR的动态Supertrend计算,使策略具有良好的市场适应性。
4. 完善的止损和获利管理机制,既控制风险又锁定利润。
5. 可视化做得很完善,交易信号、止损点位和获利目标都清晰可见。

#### 策略风险
1. 在震荡市场中可能产生频繁的假突破信号。
2. 固定的止损和获利设置可能不适合所有市场环境。
3. 趋势反转时可能出现较大回撤。
4. 对参数设置敏感,不当的参数可能导致策略表现不佳。
5. 在流动性较差的市场中,可能面临滑点风险。

#### 策略优化方向
1. 可以引入成交量指标作为信号确认。
2. 考虑加入动态的ATR倍数调节机制。
3. 基于市场波动率动态调整止损和获利比例。
4. 增加时间过滤器,避免在不适合的时间段交易。
5. 考虑加入趋势强度过滤器,提高交易质量。

#### 总结
这是一个设计严谨、逻辑清晰的策略,通过结合技术指标和形态分析,实现了较好的信号质量控制。策略的风险管理机制完善,可视化效果出色,适合进行实盘测试和优化。建议交易者在实际应用中注意参数调优和市场环境的选择。 || 

#### Overview
This is an advanced trading strategy that combines the Supertrend indicator with engulfing candlestick patterns. The strategy identifies engulfing patterns in the market and confirms them with the Supertrend indicator's trend direction to achieve precise trade signal filtering. It also incorporates dynamic stop-loss and take-profit settings to effectively control risk while ensuring profit potential.

#### Strategy Principles
The strategy is based on the following core principles:
1. Uses ATR (Average True Range) to calculate the Supertrend indicator for determining overall market trend.
2. Filters effective engulfing patterns through Boring Candle Threshold and Engulfing Candle Threshold settings.
3. Only enters trades when Supertrend trend direction aligns with engulfing pattern direction.
4. Employs dynamic stop-loss and take-profit levels calculated proportionally from entry price.
5. Implements position management ensuring only one trade direction at a time.

#### Strategy Advantages
1. Strict signal quality control through dual confirmation (trend + pattern).
2. Introduction of boring candle and engulfing thresholds effectively filters false signals.
3. ATR-based dynamic Supertrend calculation provides good market adaptability.
4. Comprehensive stop-loss and profit management mechanism.
5. Excellent visualization of trade signals, stop-loss levels, and profit targets.

#### Strategy Risks
1. May generate frequent false breakout signals in ranging markets.
2. Fixed stop-loss and take-profit settings might not suit all market conditions.
3. Potential for significant drawdowns during trend reversals.
4. Sensitivity to parameter settings may lead to poor strategy performance.
5. Slippage risks in markets with lower liquidity.

#### Optimization Directions
1. Consider incorporating volume indicators for signal confirmation.
2. Implement dynamic ATR multiplier adjustment mechanism.
3. Develop dynamic stop-loss and take-profit ratios based on market volatility.
4. Add time filters to avoid trading during unsuitable periods.
5. Consider adding trend strength filters to improve trade quality.

#### Summary
This is a well-designed strategy with clear logic that achieves good signal quality control through combined technical indicators and pattern analysis. The strategy features comprehensive risk management mechanisms and excellent visualization, making it suitable for live testing and optimization. Traders should pay attention to parameter optimization and market environment selection during practical application.[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-02-21 00:00:00
end: 2024-06-01 00:00:00
period: 1d
basePeriod: 1d
exchanges: [{"eid":"Binance","currency":"ETH_USDT"}]
*/

//@version=5
strategy('Strategy Engulfing', overlay=true)

// Inputs
Periods = input(title='ATR Period', defval=5)
src = input(hl2, title='Source')
Multiplier = input.float(title='ATR Multiplier', step=0.1, defval=1.0)
highlighting = input(title='Highlighter On/Off?', defval=true)
boringThreshold = input.int(5, title='Boring Candle Threshold (%)', minval=1, maxval=100, step=1)
engulfingThreshold = input.int(50, title='Engulfing Candle Threshold (%)', minval=1, maxval=100, step=1)
OpenPosisi = input.int(2000, title='OpenPosisi (Pips)', minval=-25000)
stoploss = input.int(10000, title='Stop Loss (Pips)', minval=-25000)
takeprofit = input.int(20000, title='Take Profit (Pips)', minval=-25000)

// ATR Calculation
atr = ta.atr(Periods)

// Supertrend Calculation
up = src - Multiplier * atr
up := close[1] > nz(up[1]) ? math.max(up, nz(up[1])) : up
dn = src + Multiplier * atr
dn := close[1] < nz(dn[1]) ? math.min(dn, nz(dn[1])) : dn
trend = 1
trend := nz(trend[1], trend)
trend := trend == -1 and close > dn[1] ? 1 : trend == 1 and close < up[1] ? -1 : trend

// Plotting Supertrend
plot(trend == 1 ? up : na, color=color.new(color.green, 0), linewidth=1, style=plot.style_linebr, title='Supertrend Up')
plot(trend == -1 ? dn : na, color=color.new(color.red, 0), linewidth=1, style=plot.style_linebr, title='Supertrend Down')

// Engulfing Candlestick
isBoringCandle = math.abs(open[1] - close[1]) <= (high[1] - low[1]) * boringThreshold / 100
isEngulfingCandle = math.abs(open - close) * 100 / math.abs(high - low) <= engulfingThreshold

bullEngulfing = strategy.opentrades == 0 and trend == 1 and close[1] < open[1] and close > open[1] and not isBoringCandle and not isEngulfingCandle
bearEngulfing = strategy.opentrades == 0 and trend == -1 and close[1] > open[1] and close < open[1] and not isBoringCandle and not isEngulfingCandle

// Calculate Limit Price
limitbull = bullEngulfing ? close + OpenPosisi * syminfo.mintick : na
limitbear = bearEngulfing ? close - OpenPosisi * syminfo.mintick : na

// Calculate Stop Loss
bullishStopLoss = bullEngulfing ? limitbull - stoploss * syminfo.mintick : na
bearishStopLoss = bearEngulfing ? limitbear + stoploss * syminfo.mintick : na

// Calculate Take Profit
bullishTakeProfit = bullEngulfing ? limitbull + takeprofit * syminfo.mintick : na
bearishTakeProfit = bearEngulfing ? limitbear - takeprofit * syminfo.mintick : na


// Alerts for Engulfing Candles (Trigger Immediately)
if bullEngulfing
    alert('Bullish Engulfing Candle Formed!')

if bearEngulfing
    alert('Bearish Engulfing Candle Formed!')

// Plot shapes
plotshape(bullEngulfing, style=shape.triangleup, location=location.abovebar, color=color.new(color.green, 0))
plotshape(bearEngulfing, style=shape.triangledown, location=location.belowbar, color=color.new(color.red, 0))


plot(limitbull, title='Bullish Limit Price', color=color.new(color.purple, 0), style=plot.style_linebr, linewidth=1)
plot(limitbear, title='Bearish Limit Price', color=color.new(color.purple, 0), style=plot.style_linebr, linewidth=1)
plot(bullishStopLoss, title='Bullish Stop Loss', color=color.new(color.red, 0), style=plot.style_linebr, linewidth=1)
plot(bearishStopLoss, title='Bearish Stop Loss', color=color.new(color.red, 0), style=plot.style_linebr, linewidth=1)
plot(bullishTakeProfit, title='Bullish Take Profit', color=color.new(color.blue, 0), style=plot.style_linebr, linewidth=1)
plot(bearishTakeProfit, title='Bearish Take Profit', color=color.new(color.blue, 0), style=plot.style_linebr, linewidth=1)

// Label Stop Loss and Take Profit
label.new(bullEngulfing ? bar_index : na, bullishStopLoss, text='SL: ' + str.tostring(bullishStopLoss), color=color.red, textcolor=color.white, style=label.style_label_up, size=size.tiny)
label.new(bearEngulfing ? bar_index : na, bearishStopLoss, text='SL: ' + str.tostring(bearishStopLoss), color=color.red, textcolor=color.white, style=label.style_label_down, size=size.tiny)
label.new(bullEngulfing ? bar_index : na, bullishTakeProfit, text='TP: ' + str.tostring(bullishTakeProfit), color=color.green, textcolor=color.white, style=label.style_label_down, size=size.tiny)
label.new(bearEngulfing ? bar_index : na, bearishTakeProfit, text='TP: ' + str.tostring(bearishTakeProfit), color=color.green, textcolor=color.white, style=label.style_label_up, size=size.tiny)


// Strategy execution
if bullEngulfing
    strategy.entry('BUY', strategy.long, stop=limitbull)
    strategy.exit('TP/SL', from_entry='BUY', limit=bullishTakeProfit, stop=bullishStopLoss)

if bearEngulfing
    strategy.entry('SELL', strategy.short, stop=limitbear)
    strategy.exit('TP/SL', from_entry='SELL', limit=bearishTakeProfit, stop=bearishStopLoss)


```

> Detail

https://www.fmz.com/strategy/482855

> Last Modified

2025-02-20 15:32:32
