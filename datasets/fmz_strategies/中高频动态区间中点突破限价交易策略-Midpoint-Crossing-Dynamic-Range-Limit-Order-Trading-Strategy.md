
> Name

中高频动态区间中点突破限价交易策略-Midpoint-Crossing-Dynamic-Range-Limit-Order-Trading-Strategy

> Author

ianzeng123

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/2d8fcc61ea5666373b580.png)
![IMG](https://www.fmz.com/upload/asset/2d952b61c05b9814a7be6.png)




[trans]
#### 概述

本策略是一种基于市场动态区间中点的高精度交易方法，通过捕捉价格在特定时间范围内的波动特征，实现精准的入场和出场时机。策略核心是利用可配置的回溯周期，动态计算价格区间的高点、低点和中点，并在纽约证券交易所交易时段内执行限价交易。

#### 策略原理

策略原理基于以下关键机制：
1. 动态区间计算：通过设定可调整的回溯周期（默认30个K线），实时计算价格的最高点、最低点和中点。
2. 时间约束交易：严格限定在纽约证券交易所交易时段（上午9:30至下午3:00）内进行交易。
3. 中点突破信号：当收盘价格突破区间中点时，产生做多或做空信号。
4. 限价订单策略：在区间中点下单，并设置止盈和止损价格为区间高点和低点。

#### 策略优势

1. 高精度入场：通过动态计算区间中点，提供更加精准的入场时机。
2. 风险可控：严格的止盈和止损机制，有效控制单笔交易风险。
3. 时间选择性：仅在交易所活跃时段交易，避免低流动性periods。
4. 参数灵活：回溯周期可调，适应不同市场环境。
5. 避免隔夜风险：在交易日结束前自动平仓。

#### 策略风险

1. 区间计算局限性：在剧烈波动市场中，固定回溯周期可能无法准确反映实时市场状态。
2. 交易频率风险：频繁交易可能增加交易成本和滑点风险。
3. 参数敏感性：回溯周期和交易时段设置对策略表现影响显著。
4. 市场适应性：策略可能不适用于所有品种和市场环境。

#### 策略优化方向

1. 动态回溯周期：引入自适应算法，根据市场波动性动态调整回溯周期。
2. 多时间框架验证：结合不同时间框架的信号，提高信号准确性。
3. volatility过滤：增加波动率指标，过滤低质量交易信号。
4. 机器学习优化：使用机器学习算法动态调整入场和出场参数。
5. 风险管理增强：引入更复杂的仓位管理和动态止损机制。

#### 总结

本策略通过精确的区间中点突破和限价交易机制，为交易者提供了一种系统化、规则明确的交易方法。其核心优势在于高精度入场、风险可控和时间选择性。未来的优化方向将focus于提高策略的自适应性和稳定性。

#### 关键技术指标

- 回溯周期（Lookback Period）
- 区间高点（Range High）
- 区间低点（Range Low）
- 区间中点（Range Midpoint）
- 交易时段（NYSE Trading Hours）

#### 交易逻辑总结

通过动态计算价格区间并在中点附近进行限价交易，在严格的时间和风险管理框架下，捕捉短期价格趋势和反转机会。

#### 风险提示

本策略仅供参考，实际交易中需要根据个人风险承受能力和市场环境进行调整。

#### 推荐应用场景

适用于追求稳定、系统化交易策略的中短线投资者，特别是专注于期指和高流动性品种交易的交易者。

#### 结语

量化交易的核心在于不断优化和适应，本策略为交易者提供了一个值得深入研究和改进的交易框架。

|| 

#### Overview

This strategy is a high-precision trading method based on the market's dynamic range midpoint, capturing price fluctuation characteristics within a specific time range to achieve precise entry and exit timing. The core of the strategy is to use a configurable lookback period to dynamically calculate the price range's high, low, and midpoint, and execute limit orders during New York Stock Exchange trading hours.

#### Strategy Principles

The strategy principles are based on the following key mechanisms:
1. Dynamic Range Calculation: Real-time calculation of price highs, lows, and midpoints by setting an adjustable lookback period (default 30 candles).
2. Time-Constrained Trading: Strictly limited to New York Stock Exchange trading hours (9:30 AM to 3:00 PM).
3. Midpoint Breakthrough Signals: Generate long or short signals when closing prices cross the range midpoint.
4. Limit Order Strategy: Place orders at the range midpoint with take profit and stop loss set at range high and low points.

#### Strategy Advantages

1. High-Precision Entry: Provide more accurate entry timing through dynamic midpoint calculation.
2. Controllable Risk: Strict take profit and stop loss mechanisms effectively control single trade risk.
3. Time Selectivity: Trade only during active exchange hours, avoiding low liquidity periods.
4. Flexible Parameters: Adjustable lookback period adapts to different market environments.
5. Overnight Risk Avoidance: Automatic position closure before trading day end.

#### Strategy Risks

1. Range Calculation Limitations: Fixed lookback periods may not accurately reflect real-time market conditions during volatile markets.
2. Trading Frequency Risks: Frequent trading may increase transaction costs and slippage risks.
3. Parameter Sensitivity: Lookback period and trading hours significantly impact strategy performance.
4. Market Adaptability: Strategy may not apply to all instruments and market environments.

#### Strategy Optimization Directions

1. Dynamic Lookback Period: Introduce adaptive algorithms to dynamically adjust lookback periods based on market volatility.
2. Multi-Timeframe Verification: Combine signals from different timeframes to improve signal accuracy.
3. Volatility Filtering: Add volatility indicators to filter low-quality trading signals.
4. Machine Learning Optimization: Use machine learning algorithms to dynamically adjust entry and exit parameters.
5. Enhanced Risk Management: Introduce more complex position management and dynamic stop-loss mechanisms.

#### Summary

This strategy provides traders with a systematic, rule-clear trading method through precise range midpoint breakthrough and limit order trading mechanisms. Its core advantages lie in high-precision entry, risk control, and time selectivity. Future optimization will focus on improving the strategy's adaptability and stability.

#### Key Technical Indicators

- Lookback Period
- Range High
- Range Low
- Range Midpoint
- NYSE Trading Hours

#### Trading Logic Summary

Capture short-term price trends and reversal opportunities by dynamically calculating price ranges and executing limit trades near the midpoint within a strict time and risk management framework.

#### Risk Disclaimer

This strategy is for reference only. Actual trading requires adjustment based on individual risk tolerance and market environment.

#### Recommended Application Scenarios

Suitable for medium and short-term investors seeking stable, systematic trading strategies, especially those focusing on index futures and high-liquidity instruments.

#### Conclusion

The core of quantitative trading is continuous optimization and adaptation. This strategy provides traders with a trading framework worth in-depth research and improvement.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-03-31 00:00:00
end: 2025-03-29 08:00:00
period: 1h
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"ETH_USDT"}]
*/

//@version=5
strategy("Midpoint Crossing Strategy", overlay=true)

// Input for lookback period
lookback = input.int(30, title="Lookback Period", minval=1)

// Input for NYSE trading hours
startHour = 9
startMinute = 30
endHour = 15
endMinute = 0

// Variables to store high, low, and midpoint of the lookback period
var float rangeHigh = na
var float rangeLow = na
var float rangeMid = na

// Calculate high, low, and midpoint based on lookback period
if (bar_index >= lookback)
    rangeHigh := ta.highest(high, lookback)
    rangeLow := ta.lowest(low, lookback)
    rangeMid := (rangeHigh + rangeLow) / 2

// Plot high, low, and midpoint for reference
plot(rangeHigh, color=color.red, title="Range High")
plot(rangeLow, color=color.green, title="Range Low")
plot(rangeMid, color=color.blue, title="Range Mid")

// Time condition for NYSE hours
currentTime = timestamp("GMT-5", year, month, dayofmonth, hour, minute)
startTime = timestamp("GMT-5", year, month, dayofmonth, startHour, startMinute)
endTime = timestamp("GMT-5", year, month, dayofmonth, endHour, endMinute)

// Check if the current time is within NYSE hours
isNYSEHours = currentTime >= startTime and currentTime <= endTime

// Entry conditions (only during NYSE hours)
longCondition = ta.crossover(close, rangeMid) and isNYSEHours
shortCondition = ta.crossunder(close, rangeMid) and isNYSEHours

// Define stop loss and take profit levels based on the range
longStopLoss = rangeLow
longTakeProfit = rangeHigh
shortStopLoss = rangeHigh
shortTakeProfit = rangeLow

// Place limit order at mid-price
if (longCondition and not strategy.opentrades)
    strategy.order("Long Limit", strategy.long, limit=rangeMid)
    strategy.exit("Take Profit", "Long Limit", limit=longTakeProfit, stop=longStopLoss)

if (shortCondition and not strategy.opentrades)
    strategy.order("Short Limit", strategy.short, limit=rangeMid)
    strategy.exit("Take Profit", "Short Limit", limit=shortTakeProfit, stop=shortStopLoss)

// Close open positions at 4:00 PM to avoid overnight risk
if (currentTime >= endTime)
    strategy.close_all(comment="Close All at 4:00 PM")

// Add a check for open positions
if (strategy.opentrades > 0)
    // Ensure no recalculation while a position is open
    rangeHigh := na
    rangeLow := na
    rangeMid := na

```

> Detail

https://www.fmz.com/strategy/488897

> Last Modified

2025-03-31 16:43:45
