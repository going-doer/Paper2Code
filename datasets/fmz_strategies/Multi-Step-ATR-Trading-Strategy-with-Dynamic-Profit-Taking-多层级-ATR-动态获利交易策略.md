
> Name

Multi-Step-ATR-Trading-Strategy-with-Dynamic-Profit-Taking-多层级-ATR-动态获利交易策略

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1423d06f720beb8e5dd.png)

[trans]
#### 概述
这是一个整合了自适应平均真实波幅(ATR)计算和基于动量的趋势检测的多层级交易策略。该策略最显著的特点在于其独特的7步获利机制,它结合了4个基于ATR的退出水平和3个固定百分比水平。这种混合方法使交易者能够根据市场波动性进行动态调整,同时在多空市场中系统地获取利润。策略通过动态调整ATR计算、趋势强度检测和多重获利机制的组合,为交易者提供了一个全面的交易解决方案。

#### 策略原理
策略的核心是通过以下几个关键组件来运作:
1. 增强型真实波幅计算: 通过考虑最显著的价格移动来衡量市场波动性。
2. 动量因子整合: 基于最近价格变动调整ATR,使其更具适应性。
3. 自适应ATR计算: 根据动量因子调整传统ATR,使其在波动期间更为敏感。
4. 趋势强度量化: 通过复杂算法评估趋势的强度。
5. 七步获利机制: 包括基于ATR的四个退出水平和三个固定百分比水平。

#### 策略优势
1. 适应性强: 通过动态ATR计算适应不同市场条件。
2. 风险管理完善: 多层次获利机制提供了系统化的风险控制。
3. 灵活性高: 可以在多空市场中equally有效运作。
4. 参数可调: 提供多个可调参数以适应不同交易风格。
5. 系统化执行: 明确的入场和出场规则减少情绪化交易。

#### 策略风险
1. 参数敏感性: 不当的参数设置可能导致过度交易或错失机会。
2. 市场条件依赖: 在剧烈波动或横盘市场中可能表现不佳。
3. 复杂性风险: 多层次获利机制可能增加执行难度。
4. 滑点影响: 多个获利点可能受到滑点的显著影响。
5. 资金管理要求: 需要足够资金以执行多层次获利策略。

#### 策略优化方向
1. 动态参数调整: 根据市场状况自动调整参数。
2. 市场环境过滤: 添加市场环境识别机制。
3. 风险管理增强: 引入动态止损机制。
4. 执行优化: 简化获利机制以减少滑点影响。
5. 回测框架完善: 加入更多现实交易因素。

#### 总结
这个策略通过结合自适应ATR和多层次获利机制,为交易者提供了一个全面的交易系统。它的优势在于能够适应不同市场条件,同时通过系统化的方法管理风险。虽然存在一些潜在风险,但通过适当的优化和风险管理,该策略可以成为一个有效的交易工具。其创新的多层次获利机制特别适合那些寻求在保持风险控制的同时最大化利润的交易者。

|| 

#### Overview
This is a multi-layered trading strategy that integrates adaptive Average True Range (ATR) calculations with momentum-based trend detection. The strategy's most distinctive feature is its unique 7-step profit-taking mechanism, which combines four ATR-based exit levels and three fixed percentage levels. This hybrid approach enables traders to dynamically adjust to market volatility while systematically capturing profits in both long and short market positions. The strategy provides a comprehensive trading solution through the combination of dynamic ATR calculations, trend strength detection, and multiple profit-taking mechanisms.

#### Strategy Principles
The strategy operates through several key components:
1. Enhanced True Range Calculation: Measures market volatility by considering the most significant price movements.
2. Momentum Factor Integration: Adjusts ATR based on recent price movements for better adaptability.
3. Adaptive ATR Calculation: Modifies traditional ATR based on momentum factor for increased sensitivity during volatile periods.
4. Trend Strength Quantification: Evaluates trend strength through sophisticated algorithms.
5. Seven-Step Profit Mechanism: Includes four ATR-based exit levels and three fixed percentage levels.

#### Strategy Advantages
1. High Adaptability: Adapts to different market conditions through dynamic ATR calculations.
2. Comprehensive Risk Management: Multi-layered profit-taking mechanism provides systematic risk control.
3. High Flexibility: Works equally effectively in both long and short markets.
4. Adjustable Parameters: Offers multiple customizable parameters to suit different trading styles.
5. Systematic Execution: Clear entry and exit rules reduce emotional trading.

#### Strategy Risks
1. Parameter Sensitivity: Improper parameter settings may lead to overtrading or missed opportunities.
2. Market Condition Dependency: May underperform in highly volatile or ranging markets.
3. Complexity Risk: Multi-layered profit-taking mechanism may increase execution difficulty.
4. Slippage Impact: Multiple profit points may be significantly affected by slippage.
5. Capital Requirements: Requires sufficient capital to execute multi-layered profit strategy.

#### Strategy Optimization Directions
1. Dynamic Parameter Adjustment: Automatically adjust parameters based on market conditions.
2. Market Environment Filtering: Add market environment identification mechanism.
3. Risk Management Enhancement: Introduce dynamic stop-loss mechanism.
4. Execution Optimization: Simplify profit-taking mechanism to reduce slippage impact.
5. Backtesting Framework Improvement: Include more realistic trading factors.

#### Summary
This strategy provides traders with a comprehensive trading system by combining adaptive ATR and multi-layered profit-taking mechanisms. Its strength lies in its ability to adapt to different market conditions while managing risk through a systematic approach. While there are some potential risks, the strategy can become an effective trading tool through proper optimization and risk management. Its innovative multi-layered profit-taking mechanism is particularly suitable for traders seeking to maximize profits while maintaining risk control.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-11-04 00:00:00
end: 2024-12-04 00:00:00
period: 1h
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © PresentTrading

// The SuperATR 7-Step Profit Strategy is a multi-layered trading strategy that combines adaptive ATR and momentum-based trend detection 
// with a sophisticated 7-step take-profit mechanism. This approach utilizes four ATR-based exit levels and three fixed percentage levels, 
// enabling flexible and dynamic profit-taking in both long and short market positions.

//@version=5
strategy("SuperATR 7-Step Profit - Strategy [presentTrading] ", overlay=true, precision=3, commission_value= 0.1, commission_type=strategy.commission.percent, slippage= 1, currency=currency.USD, default_qty_type = strategy.percent_of_equity, default_qty_value = 10, initial_capital=10000)

// ————————
// User Inputs
// ————————
short_period = input.int(3, minval=1, title="Short Period")
long_period = input.int(7, minval=1, title="Long Period")
momentum_period = input.int(7, minval=1, title="Momentum Period")
atr_sma_period = input.int(7, minval=1, title="ATR SMA Period for Confirmation")
trend_strength_threshold = input.float(1.618, minval=0.0, title="Trend Strength Threshold", step=0.1)

// ————————
// Take Profit Inputs
// ————————
useMultiStepTP = input.bool(true, title="Enable Multi-Step Take Profit")

// ATR-based Take Profit Inputs
atrLengthTP = input.int(14, minval=1, title="ATR Length for Take Profit")
atrMultiplierTP1 = input.float(2.618, minval=0.1, title="ATR Multiplier for TP Level 1")
atrMultiplierTP2 = input.float(5.0, minval=0.1, title="ATR Multiplier for TP Level 2")
atrMultiplierTP3 = input.float(10.0, minval=0.1, title="ATR Multiplier for TP Level 3")
atrMultiplierTP4 = input.float(13.82, minval=0.1, title="ATR Multiplier for TP Level 4")

// Fixed Percentage Take Profit Inputs
tp_level_percent1 = input.float(3.0, minval=0.1, title="Fixed TP Level 1 (%)")
tp_level_percent2 = input.float(8.0, minval=0.1, title="Fixed TP Level 2 (%)")
tp_level_percent3 = input.float(17.0, minval=0.1, title="Fixed TP Level 3 (%)")

// Take Profit Percentages for Each Level
tp_percent_atr = input.float(10.0, minval=0.1, maxval=100, title="Percentage to Exit at Each ATR TP Level")
tp_percent_fixed = input.float(10.0, minval=0.1, maxval=100, title="Percentage to Exit at Each Fixed TP Level")


// —————————————
// Helper Functions
// —————————————

// Function to calculate True Range with enhanced volatility detection
calculate_true_range() =>
    prev_close = close[1]
    tr1 = high - low
    tr2 = math.abs(high - prev_close)
    tr3 = math.abs(low - prev_close)
    true_range = math.max(tr1, tr2, tr3)
    true_range

// ———————————————
// Indicator Calculations
// ———————————————

// Calculate True Range
true_range = calculate_true_range()

// Calculate Momentum Factor
momentum = close - close[momentum_period]
stdev_close = ta.stdev(close, momentum_period)
normalized_momentum = stdev_close != 0 ? (momentum / stdev_close) : 0
momentum_factor = math.abs(normalized_momentum)

// Calculate Short and Long ATRs
short_atr = ta.sma(true_range, short_period)
long_atr = ta.sma(true_range, long_period)

// Calculate Adaptive ATR
adaptive_atr = (short_atr * momentum_factor + long_atr) / (1 + momentum_factor)

// Calculate Trend Strength
price_change = close - close[momentum_period]
atr_multiple = adaptive_atr != 0 ? (price_change / adaptive_atr) : 0
trend_strength = ta.sma(atr_multiple, momentum_period)

// Calculate Moving Averages
short_ma = ta.sma(close, short_period)
long_ma = ta.sma(close, long_period)

// Determine Trend Signal
trend_signal = (short_ma > long_ma and trend_strength > trend_strength_threshold) ? 1 :
               (short_ma < long_ma and trend_strength < -trend_strength_threshold) ? -1 : 0

// Calculate Adaptive ATR SMA for Confirmation
adaptive_atr_sma = ta.sma(adaptive_atr, atr_sma_period)

// Determine if Trend is Confirmed with Price Action
trend_confirmed = (trend_signal == 1 and close > short_ma and adaptive_atr > adaptive_atr_sma) or (trend_signal == -1 and close < short_ma and adaptive_atr > adaptive_atr_sma)

// —————————————
// Trading Logic
// —————————————

// Entry Conditions
long_entry = trend_confirmed and trend_signal == 1
short_entry = trend_confirmed and trend_signal == -1

// Exit Conditions
long_exit = strategy.position_size > 0 and short_entry
short_exit = strategy.position_size < 0 and long_entry



// Execute Long Trades
if long_entry
    strategy.entry("Long Entry", strategy.long)
if long_exit
    strategy.close("Long Entry")

// Execute Short Trades
if short_entry
    strategy.entry("Short Entry", strategy.short)
if short_exit
    strategy.close("Short Entry")

// ————————————————
// Multi-Step Take Profit Logic
// ————————————————

if useMultiStepTP
    // Calculate ATR for Take Profit Levels
    atrValueTP = ta.atr(atrLengthTP)

    // Long Position Take Profit Levels
    if strategy.position_size > 0
        // ATR-based Take Profit Prices
        tp_priceATR1_long = strategy.position_avg_price + atrMultiplierTP1 * atrValueTP
        tp_priceATR2_long = strategy.position_avg_price + atrMultiplierTP2 * atrValueTP
        tp_priceATR3_long = strategy.position_avg_price + atrMultiplierTP3 * atrValueTP
        tp_priceATR4_long = strategy.position_avg_price + atrMultiplierTP4 * atrValueTP

        // Fixed Percentage Take Profit Prices
        tp_pricePercent1_long = strategy.position_avg_price * (1 + tp_level_percent1 / 100)
        tp_pricePercent2_long = strategy.position_avg_price * (1 + tp_level_percent2 / 100)
        tp_pricePercent3_long = strategy.position_avg_price * (1 + tp_level_percent3 / 100)

        // Set ATR-based Take Profit Exits
        strategy.exit("TP ATR 1 Long", from_entry="Long Entry", qty_percent=tp_percent_atr, limit=tp_priceATR1_long)
        strategy.exit("TP ATR 2 Long", from_entry="Long Entry", qty_percent=tp_percent_atr, limit=tp_priceATR2_long)
        strategy.exit("TP ATR 3 Long", from_entry="Long Entry", qty_percent=tp_percent_atr, limit=tp_priceATR3_long)
        strategy.exit("TP ATR 4 Long", from_entry="Long Entry", qty_percent=tp_percent_atr, limit=tp_priceATR4_long)

        // Set Fixed Percentage Take Profit Exits
        strategy.exit("TP Percent 1 Long", from_entry="Long Entry", qty_percent=tp_percent_fixed, limit=tp_pricePercent1_long)
        strategy.exit("TP Percent 2 Long", from_entry="Long Entry", qty_percent=tp_percent_fixed, limit=tp_pricePercent2_long)
        strategy.exit("TP Percent 3 Long", from_entry="Long Entry", qty_percent=tp_percent_fixed, limit=tp_pricePercent3_long)

    // Short Position Take Profit Levels
    if strategy.position_size < 0
        // ATR-based Take Profit Prices
        tp_priceATR1_short = strategy.position_avg_price - atrMultiplierTP1 * atrValueTP
        tp_priceATR2_short = strategy.position_avg_price - atrMultiplierTP2 * atrValueTP
        tp_priceATR3_short = strategy.position_avg_price - atrMultiplierTP3 * atrValueTP
        tp_priceATR4_short = strategy.position_avg_price - atrMultiplierTP4 * atrValueTP

        // Fixed Percentage Take Profit Prices
        tp_pricePercent1_short = strategy.position_avg_price * (1 - tp_level_percent1 / 100)
        tp_pricePercent2_short = strategy.position_avg_price * (1 - tp_level_percent2 / 100)
        tp_pricePercent3_short = strategy.position_avg_price * (1 - tp_level_percent3 / 100)

        // Set ATR-based Take Profit Exits
        strategy.exit("TP ATR 1 Short", from_entry="Short Entry", qty_percent=tp_percent_atr, limit=tp_priceATR1_short)
        strategy.exit("TP ATR 2 Short", from_entry="Short Entry", qty_percent=tp_percent_atr, limit=tp_priceATR2_short)
        strategy.exit("TP ATR 3 Short", from_entry="Short Entry", qty_percent=tp_percent_atr, limit=tp_priceATR3_short)
        strategy.exit("TP ATR 4 Short", from_entry="Short Entry", qty_percent=tp_percent_atr, limit=tp_priceATR4_short)

        // Set Fixed Percentage Take Profit Exits
        strategy.exit("TP Percent 1 Short", from_entry="Short Entry", qty_percent=tp_percent_fixed, limit=tp_pricePercent1_short)
        strategy.exit("TP Percent 2 Short", from_entry="Short Entry", qty_percent=tp_percent_fixed, limit=tp_pricePercent2_short)
        strategy.exit("TP Percent 3 Short", from_entry="Short Entry", qty_percent=tp_percent_fixed, limit=tp_pricePercent3_short)


// ——————————
// Plotting
// ——————————

plot(short_ma, color=color.blue, title="Short MA")
plot(long_ma, color=color.orange, title="Long MA")

// Plot Buy and Sell Signals
//plotshape(long_entry, title="Long Entry", style=shape.triangleup, location=location.belowbar, color=color.green, size=size.small, text="Buy")
//plotshape(short_entry, title="Short Entry", style=shape.triangledown, location=location.abovebar, color=color.red, size=size.small, text="Sell")

// Optional: Plot Trend Strength for analysis
// Uncomment the lines below to display Trend Strength on a separate chart pane
// plot(trend_strength, title="Trend Strength", color=color.gray)
// hline(trend_strength_threshold, title="Trend Strength Threshold", color=color.gray, linestyle=hline.style_dashed)
// hline(-trend_strength_threshold, title="Trend Strength Threshold", color=color.gray, linestyle=hline.style_dashed)


```

> Detail

https://www.fmz.com/strategy/474064

> Last Modified

2024-12-05 16:49:57
