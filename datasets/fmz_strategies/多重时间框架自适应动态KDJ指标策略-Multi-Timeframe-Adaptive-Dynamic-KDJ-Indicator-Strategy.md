
> Name

多重时间框架自适应动态KDJ指标策略-Multi-Timeframe-Adaptive-Dynamic-KDJ-Indicator-Strategy

> Author

ianzeng123

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/2d9109d948d3f10b585bd.png)
![IMG](https://www.fmz.com/upload/asset/2d8bf2206fec2ba89ece7.png)


[trans]
#### 概述

这是一个创新的多时间框架自适应KDJ震荡指标策略，旨在通过动态调整指标参数和跨多个时间框架分析市场趋势，提供更精确和灵活的交易信号。该策略结合了volatility-based长度计算、跨多个时间框架的权重分配以及自适应趋势判断，为交易者提供了一个复杂而强大的分析工具。

#### 策略原理

策略的核心原理包括以下关键技术：

1. 多时间框架分析：同时使用1分钟、5分钟和15分钟三个时间框架
2. 自适应swing长度计算：基于市场波动率动态调整指标参数
3. 动态权重分配：为不同时间框架分配不同的权重系数
4. 趋势判断机制：通过计算smoothAvgTotal的平均值确定市场趋势方向
5. 智能信号生成：结合主要信号和预期信号，提高信号准确性

#### 策略优势

1. 高度灵活性：可自定义时间框架和权重配置
2. 动态适应性：根据市场波动率调整指标参数
3. 多维度分析：综合多个时间框架信息
4. 低延迟信号：包含主要信号和预期信号
5. 内置趋势过滤：减少不利市场条件下的错误信号

#### 策略风险

1. 参数过度拟合风险
2. 多时间框架可能增加信号复杂性
3. 在极端市场条件下信号可靠性可能降低
4. 需要额外确认指标验证信号

#### 策略优化方向

1. 引入机器学习算法动态调整权重
2. 增加附加过滤条件
3. 优化止损机制
4. 开发跨品种适应性

#### 总结

该多时间框架自适应KDJ震荡指标策略通过创新的设计，为交易者提供了一个灵活、动态且多维度的市场分析工具，具有显著的技术优势和潜在的性能提升空间。

|| 

#### Overview

This is an innovative multi-timeframe adaptive KDJ oscillator strategy designed to provide more precise and flexible trading signals by dynamically adjusting indicator parameters and analyzing market trends across multiple timeframes. The strategy combines volatility-based length calculation, multi-timeframe weight allocation, and adaptive trend determination, offering traders a sophisticated and powerful analysis tool.

#### Strategy Principles

The core principles of the strategy include the following key technologies:

1. Multi-timeframe analysis: Simultaneously using 1-minute, 5-minute, and 15-minute timeframes
2. Adaptive swing length calculation: Dynamically adjusting indicator parameters based on market volatility
3. Dynamic weight allocation: Assigning different weight coefficients to different timeframes
4. Trend determination mechanism: Determining market trend direction by calculating the average of smoothAvgTotal
5. Intelligent signal generation: Combining primary signals and anticipated signals to improve signal accuracy

#### Strategy Advantages

1. High flexibility: Customizable timeframe and weight configuration
2. Dynamic adaptability: Adjusting indicator parameters based on market volatility
3. Multi-dimensional analysis: Integrating information from multiple timeframes
4. Low-latency signals: Including primary and anticipated signals
5. Built-in trend filtering: Reducing false signals in unfavorable market conditions

#### Strategy Risks

1. Parameter overfitting risk
2. Increased signal complexity with multiple timeframes
3. Potential signal reliability reduction in extreme market conditions
4. Requires additional confirmation indicators to validate signals

#### Strategy Optimization Directions

1. Introduce machine learning algorithms for dynamic weight adjustment
2. Add additional filtering conditions
3. Optimize stop-loss mechanisms
4. Develop cross-instrument adaptability

#### Summary

The multi-timeframe adaptive KDJ oscillator strategy provides traders with a flexible, dynamic, and multi-dimensional market analysis tool through innovative design, offering significant technical advantages and potential performance improvement space.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2025-01-01 00:00:00
end: 2025-01-25 00:00:00
period: 1d
basePeriod: 1d
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ is subject to the Mozilla Public License 2.0 (https://mozilla.org/MPL/2.0/)
// © Lausekopf

//@version=5
strategy("Adaptive KDJ (MTF)", overlay=false)

// Dropdown for the swing length calculation method
method = input.int(1, title="Calculation Method", options=[1, 2, 3], tooltip="1: Volatility Based\n2: Inverse Volatility\n3: Fixed Length")

// Fixed length for method 3
fixedLength = input.int(9, title="Fixed KDJ Length", minval=3, maxval=15)

// Timeframes
tf1 = input.timeframe("1", title="Timeframe 1")
tf2 = input.timeframe("5", title="Timeframe 2")
tf3 = input.timeframe("15", title="Timeframe 3")

// Timeframe weighting
weightOption = input.int(1, title="Timeframe Weighting", options=[1, 2, 3, 4, 5])
weightTF1 = weightOption == 1 ? 0.5 : weightOption == 2 ? 0.4 : weightOption == 3 ? 0.33 : weightOption == 4 ? 0.2 : 0.1
weightTF2 = 0.33
weightTF3 = 1.0 - (weightTF1 + weightTF2)

// EMA smoothing length
smoothingLength = input.int(5, title="EMA Smoothing Length", minval=1, maxval=50)

// Trend calculation period
trendLength = input.int(40, title="Trend Calculation Period", minval=5, maxval=50)

// KDJ function
f_kdj(len, srcHigh, srcLow, srcClose) =>
    roundedLen = int(math.round(len))
    high_max = ta.highest(srcHigh, roundedLen)
    low_min = ta.lowest(srcLow, roundedLen)
    rsv = 100 * (srcClose - low_min) / (high_max - low_min)
    k = ta.sma(rsv, 3)
    d = ta.sma(k, 3)
    j = 3 * k - 2 * d
    [k, d, j]

// Swing length function
f_swingLength(tf) =>
    atrLen = 14
    volatility = request.security(syminfo.tickerid, tf, ta.atr(atrLen) / close)
    var float length = na
    if method == 1
        length := volatility > 0.03 ? 3 : volatility > 0.002 ? 14 : 15
    if method == 2
        length := 18
    if method == 3
        length := fixedLength
    length

// Calculate swing lengths for each timeframe
swingLength1 = f_swingLength(tf1)
swingLength2 = f_swingLength(tf2)
swingLength3 = f_swingLength(tf3)

// Calculate KDJ values
[k1, d1, j1] = f_kdj(swingLength1, request.security(syminfo.tickerid, tf1, high), request.security(syminfo.tickerid, tf1, low), request.security(syminfo.tickerid, tf1, close))
[k2, d2, j2] = f_kdj(swingLength2, request.security(syminfo.tickerid, tf2, high), request.security(syminfo.tickerid, tf2, low), request.security(syminfo.tickerid, tf2, close))
[k3, d3, j3] = f_kdj(swingLength3, request.security(syminfo.tickerid, tf3, high), request.security(syminfo.tickerid, tf3, low), request.security(syminfo.tickerid, tf3, close))

// Weighted averages
avgK = (k1 * weightTF1 + k2 * weightTF2 + k3 * weightTF3)
avgD = (d1 * weightTF1 + d2 * weightTF2 + d3 * weightTF3)
avgJ = (j1 * weightTF1 + j2 * weightTF2 + j3 * weightTF3)
smoothAvgK = ta.ema(avgK, smoothingLength)
smoothAvgD = ta.ema(avgD, smoothingLength)
smoothAvgJ = ta.ema(avgJ, smoothingLength)
smoothAvgTotal = ta.ema((avgK + avgD + avgJ) / 3, smoothingLength)

// Trend determination
trendAvg = ta.sma(smoothAvgTotal, trendLength)
isUptrend = trendAvg > 60
isDowntrend = trendAvg < 40

// Dynamic signal thresholds
buyLevel = isUptrend ? 40 : isDowntrend ? 15 : 25
sellLevel = isUptrend ? 85 : isDowntrend ? 60 : 75

// Buy/Sell signals
buySignal = smoothAvgJ < buyLevel and ta.crossover(smoothAvgK, smoothAvgD)
sellSignal = smoothAvgJ > sellLevel and ta.crossunder(smoothAvgK, smoothAvgD)

// Anticipated signals
anticipateBuy = (smoothAvgK - smoothAvgK[1]) > 0 and (smoothAvgD - smoothAvgD[1]) < 0 and math.abs(smoothAvgK - smoothAvgD) < 5
anticipateSell = (smoothAvgK - smoothAvgK[1]) < 0 and (smoothAvgD - smoothAvgD[1]) > 0 and math.abs(smoothAvgK - smoothAvgD) < 5

// Entry conditions
longEntryCondition = (buySignal or anticipateBuy) and smoothAvgTotal < 22
shortEntryCondition = (sellSignal or anticipateSell) and smoothAvgTotal > 78

// Entry orders
strategy.entry("Long", strategy.long, when=longEntryCondition)
strategy.entry("Short", strategy.short, when=shortEntryCondition)

// Trailing Stop-Loss
atrMultiplierTSL = 2.5
atrValueTSL = ta.atr(12) * atrMultiplierTSL
strategy.exit("TSL Long", from_entry="Long", trail_points=atrValueTSL / syminfo.mintick, stop=open * 0.9972)
strategy.exit("TSL Short", from_entry="Short", trail_points=atrValueTSL / syminfo.mintick, stop=open * 1.0028)

// Plot signals
plotshape(series=buySignal, location=location.bottom, style=shape.triangleup, color=color.green, size=size.small)
plotshape(series=sellSignal, location=location.top, style=shape.triangledown, color=color.red, size=size.small)
plotshape(series=anticipateBuy, location=location.bottom, style=shape.triangleup, color=color.blue, size=size.tiny, offset=-1)
plotshape(series=anticipateSell, location=location.top, style=shape.triangledown, color=color.orange, size=size.tiny, offset=-1)

// Plot KDJ lines
plot(smoothAvgK, color=color.blue, linewidth=1)
plot(smoothAvgD, color=color.orange, linewidth=1)
plot(smoothAvgJ, color=color.purple, linewidth=1)
plot(smoothAvgTotal, color=color.white, linewidth=1)

// Alert for impending signals
alertcondition(anticipateBuy or anticipateSell, title='Impending KDJ Crossover', message='Possible KDJ crossover detected!')
// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © Lausekopf

```

> Detail

https://www.fmz.com/strategy/489320

> Last Modified

2025-04-03 15:13:59
