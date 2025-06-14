
> Name

动态RSI低高点背离趋势策略-Dynamic-RSI-Low-High-Point-Divergence-Trend-Strategy

> Author

ianzeng123

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/2d8d8b0a248a1a68fa372.png)
![IMG](https://www.fmz.com/upload/asset/2d8e4668d70229a0ffdc8.png)




[trans]
#### 概述

本文详细阐述了一种基于相对强弱指标(RSI)的低高点背离趋势交易策略。该策略通过识别价格与RSI指标之间的背离情况，捕捉潜在的趋势反转机会，为交易者提供精准的入场和出场信号。策略独特地结合了视觉化信号和技术指标分析，旨在提高交易决策的准确性和及时性。

#### 策略原理

策略的核心原理基于相对强弱指标(RSI)的低高点背离理论。具体实现包括以下关键步骤：

1. 计算RSI指标：使用14周期的RSI长度，评估市场当前的超买超卖状态。
2. 识别价格极值：通过回溯期(lookback)确定低点和高点。
3. 背离判断机制：
   - 看涨背离：价格创新低，而RSI指标未同步下跌
   - 看跌背离：价格创新高，而RSI指标未同步上涨
4. 信号生成：
   - 超卖区(低于30)下的看涨背离
   - 超买区(高于70)下的看跌背离

#### 策略优势

1. 高精确度信号识别：通过严格的背离条件过滤，减少假信号。
2. 可视化信号呈现：使用大型三角形标记和背景高亮，提升信号可读性。
3. 灵活性强：可调整RSI参数、回溯期和超买超卖阈值。
4. 多时间框架适应性：在1小时至4小时周期表现最佳。
5. 调试功能：内置调试表格，便于验证关键指标。

#### 策略风险

1. 误判风险：背离信号并非100%准确，存在一定概率的错误信号。
2. 市场剧烈波动：在趋势强劲市场中，背离策略可能表现欠佳。
3. 参数敏感性：RSI参数和回溯期的不当设置可能降低策略效果。
4. 交易成本：频繁交易可能产生较高的手续费和滑点成本。

#### 策略优化方向

1. 多指标确认：结合移动平均线、MACD等指标提高信号准确性。
2. 动态参数调整：根据市场波动性智能调整RSI参数。
3. 止损机制：引入基于ATR的动态止损策略。
4. 机器学习优化：使用机器学习算法动态优化入场出场点。
5. 风险管理：根据市场波动率调整仓位规模。

#### 总结

动态RSI低高点背离趋势策略通过精准的技术指标分析和可视化信号，为交易者提供了一种相对高效的趋势交易方法。通过持续优化和风险管理，该策略有望在不同市场环境中保持稳定的表现。

|| 

#### Overview

This article elaborates on a trend trading strategy based on the Relative Strength Index (RSI) low-high point divergence. The strategy captures potential trend reversal opportunities by identifying divergences between price and RSI indicators, providing traders with precise entry and exit signals. The strategy uniquely combines visual signals and technical indicator analysis to enhance the accuracy and timeliness of trading decisions.

#### Strategy Principles

The core principle is based on the RSI low-high point divergence theory. Key implementation steps include:

1. RSI Calculation: Using 14-period RSI length to assess market overbought and oversold states.
2. Price Extreme Identification: Determining low and high points through lookback period.
3. Divergence Judgment Mechanism:
   - Bullish Divergence: Price makes a new low, but RSI does not synchronously decline
   - Bearish Divergence: Price makes a new high, but RSI does not synchronously rise
4. Signal Generation:
   - Bullish divergence in oversold zone (below 30)
   - Bearish divergence in overbought zone (above 70)

#### Strategy Advantages

1. High-precision Signal Identification: Reduce false signals through strict divergence conditions.
2. Visualization of Signals: Large triangular markers and background highlighting enhance readability.
3. High Flexibility: Adjustable RSI parameters, lookback periods, and overbought/oversold thresholds.
4. Multi-timeframe Adaptability: Best performance on 1-hour to 4-hour periods.
5. Debugging Functionality: Built-in debug table for verifying key indicators.

#### Strategy Risks

1. Misjudgment Risk: Divergence signals are not 100% accurate.
2. Market Volatility: May underperform in strongly trending markets.
3. Parameter Sensitivity: Improper RSI and lookback period settings can reduce strategy effectiveness.
4. Transaction Costs: Frequent trading may incur high commission and slippage expenses.

#### Strategy Optimization Directions

1. Multi-indicator Confirmation: Incorporate moving averages, MACD to improve signal accuracy.
2. Dynamic Parameter Adjustment: Intelligently adjust RSI parameters based on market volatility.
3. Stop-loss Mechanism: Introduce ATR-based dynamic stop-loss strategy.
4. Machine Learning Optimization: Use machine learning algorithms to optimize entry and exit points.
5. Risk Management: Adjust position sizes based on market volatility.

#### Summary

The Dynamic RSI Low-High Point Divergence Trend Strategy provides traders with an efficient trend trading method through precise technical indicator analysis and visual signals. Continuous optimization and risk management can help maintain stable performance across different market environments.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-03-31 00:00:00
end: 2025-03-29 08:00:00
period: 1d
basePeriod: 1d
exchanges: [{"eid":"Futures_Binance","currency":"ETH_USDT"}]
*/

//@version=6
strategy("RSI Divergence Strategy - Visible Signals", overlay=true)

// 1. Basic Inputs (Keep it simple)
rsiLength = input.int(14, "RSI Length")
lookback = input.int(10, "Lookback Period", minval=5)
oversold = input.int(30, "Oversold Level")
overbought = input.int(70, "Overbought Level")

// 2. Calculate Indicators
rsi = ta.rsi(close, rsiLength)
priceLow = ta.lowest(low, lookback)
priceHigh = ta.highest(high, lookback)
rsiLow = ta.lowest(rsi, lookback)
rsiHigh = ta.highest(rsi, lookback)

// 3. Simple Divergence Detection
bullishDiv = low == priceLow and rsi > rsiLow and rsi < oversold
bearishDiv = high == priceHigh and rsi < rsiHigh and rsi > overbought

// 4. Visual Signals (Large and Clear)
plotshape(bullishDiv, "Buy", shape.triangleup, location.belowbar, 
     color=color.new(color.green, 0), size=size.large)
plotshape(bearishDiv, "Sell", shape.triangledown, location.abovebar, 
     color=color.new(color.red, 0), size=size.large)

// 5. Optional: Add Background for Better Visibility
bgcolor(bullishDiv ? color.new(color.green, 90) : bearishDiv ? color.new(color.red, 90) : na)

// 6. Basic Strategy Execution
if bullishDiv
    strategy.entry("Long", strategy.long)
    
if bearishDiv
    strategy.entry("Short", strategy.short)

// 7. Debugging Table (To verify values)
var table debugTable = table.new(position.top_right, 4, 1)
if barstate.islast
    table.cell(debugTable, 0, 0, "RSI: " + str.tostring(rsi))
    table.cell(debugTable, 1, 0, "Price Low: " + str.tostring(priceLow))
    table.cell(debugTable, 2, 0, "RSI Low: " + str.tostring(rsiLow))
    table.cell(debugTable, 3, 0, "Signal: " + (bullishDiv ? "BUY" : bearishDiv ? "SELL" : "NONE"))
    // Test Settings (paste these above the strategy call)
//rsiLength := 5
//lookback := 5
//oversold := 20
//overbought := 80
```

> Detail

https://www.fmz.com/strategy/488907

> Last Modified

2025-03-31 17:24:27
