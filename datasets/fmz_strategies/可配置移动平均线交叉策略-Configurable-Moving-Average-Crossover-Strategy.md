
> Name

可配置移动平均线交叉策略-Configurable-Moving-Average-Crossover-Strategy

> Author

ianzeng123

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/2d94ff2d1e4058cfa19a7.png)
![IMG](https://www.fmz.com/upload/asset/2d9207f2eee89e6ae03eb.png)





[trans]
#### 概述

本文介绍了一种灵活且强大的移动平均线交叉交易策略，该策略允许交易者根据不同市场条件自定义移动平均线参数和类型。策略的核心是利用不同周期和类型的移动平均线进行趋势跟踪和信号生成。

#### 策略原理

策略通过计算三个不同周期的移动平均线（快线、慢线和退出线）来生成交易信号。主要原理包括：

1. 移动平均线类型选择：支持简单移动平均线(SMA)、指数移动平均线(EMA)、加权移动平均线(WMA)和赫尔移动平均线(HMA)。
2. 入场条件：
   - 多头入场：收盘价高于快线，快线高于慢线，且收盘价高于退出线
   - 空头入场：收盘价低于快线，快线低于慢线，且收盘价低于退出线
3. 出场条件：
   - 多头出场：进入至少两根K线后，收盘价低于退出线
   - 空头出场：进入至少两根K线后，收盘价高于退出线

#### 策略优势

1. 高度可配置性：交易者可灵活调整移动平均线周期和类型
2. 多市场适应性：通过调整参数可适用于不同流动性的交易品种
3. 趋势跟踪能力强：利用多个移动平均线过滤假信号
4. 风险控制：默认设置为账户权益10%的仓位管理
5. 灵活的交易方向：可选择是否启用空头交易

#### 策略风险

1. 参数敏感性：不同市场可能需要不同的移动平均线参数
2. 趋势性市场表现更佳：在震荡市场可能产生更多无效信号
3. 交易成本：策略默认设置0.06%的交易佣金，实际交易中需考虑
4. 回测局限性：目前仅在部分品种（如BTCUSD和NIFTY）进行了初步验证

#### 策略优化方向

1. 动态参数调整：引入自适应移动平均线周期
2. 结合其他技术指标：增加RSI、MACD等指标进行信号过滤
3. 止损机制：添加基于波动率的止损策略
4. 多时间框架验证：在不同时间周期进行全面回测
5. 机器学习优化：使用算法自动寻找最优参数组合

#### 总结

可配置移动平均线交叉策略(MA-X)提供了一个灵活的趋势跟踪框架。通过合理配置和持续优化，该策略可以成为量化交易工具箱中的有力工具。交易者需要根据具体市场特征进行个性化调整，并进行充分的回测和验证。

|| 

#### Overview

This article introduces a flexible and powerful moving average crossover trading strategy that allows traders to customize moving average parameters and types based on different market conditions. The core of the strategy is to use moving averages of different periods and types for trend tracking and signal generation.

#### Strategy Principles

The strategy generates trading signals by calculating three moving averages of different periods (fast, slow, and exit lines). The main principles include:

1. Moving Average Type Selection: Supports Simple Moving Average (SMA), Exponential Moving Average (EMA), Weighted Moving Average (WMA), and Hull Moving Average (HMA).
2. Entry Conditions:
   - Long Entry: Close price above fast MA, fast MA above slow MA, and close price above exit MA
   - Short Entry: Close price below fast MA, fast MA below slow MA, and close price below exit MA
3. Exit Conditions:
   - Long Exit: After at least two bars, close price below exit MA
   - Short Exit: After at least two bars, close price above exit MA

#### Strategy Advantages

1. High Configurability: Traders can flexibly adjust moving average periods and types
2. Multi-Market Adaptability: Applicable to different liquidity instruments by adjusting parameters
3. Strong Trend Tracking Capability: Uses multiple moving averages to filter false signals
4. Risk Control: Default position management of 10% of account equity
5. Flexible Trading Direction: Option to enable or disable short trading

#### Strategy Risks

1. Parameter Sensitivity: Different markets may require different moving average parameters
2. Performance in Trending Markets: May generate more invalid signals in range-bound markets
3. Transaction Costs: Default setting of 0.06% commission needs consideration in actual trading
4. Backtesting Limitations: Currently validated only on a few instruments (e.g., BTCUSD and NIFTY)

#### Strategy Optimization Directions

1. Dynamic Parameter Adjustment: Introduce adaptive moving average periods
2. Combine with Other Technical Indicators: Add RSI, MACD for signal filtering
3. Stop Loss Mechanism: Add volatility-based stop loss strategy
4. Multi-Timeframe Verification: Comprehensive backtesting across different time periods
5. Machine Learning Optimization: Use algorithms to automatically find optimal parameter combinations

#### Summary

The Configurable Moving Average Crossover Strategy (MA-X) provides a flexible trend-tracking framework. Through reasonable configuration and continuous optimization, this strategy can become a powerful tool in the quantitative trading toolbox. Traders need to make personalized adjustments based on specific market characteristics and conduct thorough backtesting and verification.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-04-03 00:00:00
end: 2025-04-02 00:00:00
period: 2d
basePeriod: 2d
exchanges: [{"eid":"Futures_Binance","currency":"BNB_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © YetAnotherTA

//@version=6
strategy("Configurable MA Cross (MA-X) Strategy", "MA-X", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=10, commission_type = strategy.commission.percent, commission_value = 0.06)

// === Inputs ===
// Moving Average Periods
maPeriodA = input.int(13, title="Fast MA")
maPeriodB = input.int(55, title="Slow MA")
maPeriodC = input.int(34, title="Exit MA")

// MA Type Selection
maType = input.string("EMA", title="MA Type", options=["SMA", "EMA", "WMA", "HMA"])

// Toggle for Short Trades (Disabled by Default)
enableShorts = input.bool(false, title="Enable Short Trades", tooltip="Enable or disable short positions")

// === Function to Select MA Type ===
getMA(src, length) =>
    maType == "SMA" ? ta.sma(src, length) : maType == "EMA" ? ta.ema(src, length) : maType == "WMA" ? ta.wma(src, length) : ta.hma(src, length)

// === MA Calculation ===
maA = getMA(close, maPeriodA)
maB = getMA(close, maPeriodB)
maC = getMA(close, maPeriodC)

// === Global Variables for Crossover Signals ===
var bool crossAboveA = false
var bool crossBelowA = false

crossAboveA := ta.crossover(close, maA)
crossBelowA := ta.crossunder(close, maA)

// === Bar Counter for Exit Control ===
var int barSinceEntry = na

// Reset the counter on new entries
if (strategy.opentrades == 0)
    barSinceEntry := na

// Increment the counter on each bar
if (strategy.opentrades > 0)
    barSinceEntry := (na(barSinceEntry) ? 1 : barSinceEntry + 1)

// === Entry Conditions ===
goLong = close > maA and maA > maB and close > maC and crossAboveA
goShort = enableShorts and close < maA and maA < maB and close < maC and crossBelowA  // Shorts only when toggle is enabled

// === Exit Conditions (only after 1+ bar since entry) ===
exitLong = (strategy.position_size > 0) and (barSinceEntry >= 2) and (close < maC)
exitShort = enableShorts and (strategy.position_size < 0) and (barSinceEntry >= 2) and (close > maC)

// === Strategy Execution ===
// Long entry logic
if (goLong)
    strategy.close("Short")         // Close any short position
    strategy.entry("Long", strategy.long)
    alert("[MA-X] Go Long")
    barSinceEntry := 1               // Reset the bar counter

// Short entry logic (only if enabled)
if (enableShorts and goShort)
    strategy.close("Long")          // Close any long position
    strategy.entry("Short", strategy.short)
    alert("[MA-X] Go Short")
    barSinceEntry := 1               // Reset the bar counter

// Exit logic (only after at least 1 bar has passed)
if (exitLong)
    strategy.close("Long")
    alert("[MA-X] Exit Long")

if (enableShorts and exitShort)
    strategy.close("Short")
    alert("[MA-X] Exit Short")

// === Plotting ===
plot(maA, color=color.green, linewidth=2, title="Fast MA")
plot(maB, color=color.blue, linewidth=2, title="Slow MA")
plot(maC, color=color.red, linewidth=2, title="Exit MA")
```

> Detail

https://www.fmz.com/strategy/489300

> Last Modified

2025-04-03 11:31:33
