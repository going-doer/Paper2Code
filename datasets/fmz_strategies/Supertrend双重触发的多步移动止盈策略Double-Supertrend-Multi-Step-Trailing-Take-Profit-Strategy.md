
> Name

Supertrend双重触发的多步移动止盈策略Double-Supertrend-Multi-Step-Trailing-Take-Profit-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/16d759e95f0803c43d0.png)
[trans]
#### 概述

这是一个基于双重Supertrend指标的多步移动止盈策略。该策略利用两个参数不同的Supertrend指标来判断市场趋势,并根据趋势方向进行多空交易。策略的核心在于采用多步移动止盈机制,通过设置多个止盈目标来逐步锁定利润,同时保留部分仓位以把握更大的行情。这种方法既能降低风险,又能最大化盈利潜力。

#### 策略原理

1. 双重Supertrend指标:策略使用两个参数设置不同的Supertrend指标来判断趋势。当两个指标同时显示上升趋势时,触发做多信号;当两个指标同时显示下降趋势时,触发做空信号。这种双重确认机制可以有效减少假信号。

2. 多步移动止盈:策略设置了4个可调节的止盈目标。每个目标都有相应的止盈百分比和平仓比例。例如,第一个止盈目标可能设置为6%利润时平掉12%的仓位,第二个目标可能是12%利润时平掉8%的仓位,以此类推。这种机制既能逐步锁定利润,又能让部分仓位继续享受行情。

3. 灵活的交易方向:策略允许用户选择只做多、只做空或双向交易,以适应不同的市场环境和交易偏好。

4. 动态止损:虽然代码中没有明确的止损设置,但策略会在Supertrend指标反转时自动平仓,这实际上起到了动态止损的作用。

#### 策略优势

1. 风险管理优化:多步移动止盈机制大大改善了策略的风险收益比。通过逐步锁定利润,策略可以在保留上涨空间的同时降低回撤风险。

2. 减少假信号:双重Supertrend指标的使用显著降低了假信号的影响,提高了交易的准确性和可靠性。

3. 适应性强:策略可以根据用户偏好和市场状况灵活调整交易方向和止盈参数,适用于各种交易品种和时间周期。

4. 自动化程度高:策略完全自动化,从入场、止盈到出场都无需人工干预,大大降低了情绪影响和操作失误的可能性。

5. 资金管理灵活:通过设置不同的止盈比例,策略可以实现灵活的资金管理,既能保证快速锁定部分利润,又能让剩余仓位继续获利。

#### 策略风险

1. 参数敏感性:策略的性能很大程度上依赖于Supertrend指标和止盈参数的设置。不恰当的参数可能导致过度交易或错过重要机会。

2. 趋势依赖:作为一个趋势跟踪策略,在震荡市场中可能会频繁进出,造成不必要的交易成本。

3. 滑点风险:在快速行情中,多步止盈的执行可能受到滑点影响,实际执行价格可能与预期有所偏差。

4. 过度优化风险:策略有多个可调参数,容易陷入过度优化的陷阱,导致回测结果与实盘表现差异较大。

#### 策略优化方向

1. 引入波动率过滤:考虑结合ATR或其他波动率指标,在低波动率期间减少交易频率,提高策略在不同市场环境下的适应性。

2. 动态参数调整:可以探索使用自适应算法动态调整Supertrend参数和止盈目标,以更好地适应市场变化。

3. 增加止损机制:虽然Supertrend反转提供了一定的止损功能,但可以考虑添加更灵活的止损机制,如跟踪止损,进一步控制风险。

4. 结合其他技术指标:可以考虑引入RSI、MACD等其他技术指标,通过多指标共振来提高入场和出场的准确性。

5. 优化资金管理:可以探索更复杂的资金管理策略,如根据账户盈利情况动态调整仓位大小,以更好地平衡风险和收益。

6. 回测优化:进行更全面的回测,包括不同时间周期、不同市场条件下的表现分析,以找出策略的最佳应用场景和参数设置。

#### 总结

该多步移动止盈策略基于双重Supertrend指标,通过灵活的多步止盈机制实现了风险和收益的平衡。策略的主要优势在于其优秀的风险管理能力和对趋势的敏感度。然而,用户在应用时需要注意参数设置和市场环境的影响。通过进一步优化和完善,该策略有潜力成为一个稳健可靠的自动化交易系统。在实际应用中,建议traders进行充分的回测和模拟交易,并根据具体的交易品种和市场状况进行适当的参数调整。

|| 

#### Overview

This is a multi-step trailing take-profit strategy based on dual Supertrend indicators. The strategy uses two Supertrend indicators with different parameters to determine market trends and execute long or short trades accordingly. The core of the strategy lies in its multi-step trailing take-profit mechanism, which sets multiple profit targets to progressively lock in profits while keeping a portion of the position open to capture larger market movements. This approach aims to reduce risk while maximizing profit potential.

#### Strategy Principles

1. Dual Supertrend Indicators: The strategy employs two Supertrend indicators with different parameter settings to identify trends. A long signal is triggered when both indicators show an uptrend, while a short signal is triggered when both indicate a downtrend. This dual confirmation mechanism effectively reduces false signals.

2. Multi-Step Trailing Take-Profit: The strategy sets up 4 adjustable take-profit targets. Each target has a corresponding profit percentage and position closing ratio. For example, the first target might close 12% of the position at 6% profit, the second might close 8% at 12% profit, and so on. This mechanism allows for gradual profit locking while keeping part of the position open to benefit from continued market movements.

3. Flexible Trade Direction: Users can choose to trade long-only, short-only, or both directions, adapting to different market environments and trading preferences.

4. Dynamic Stop-Loss: Although there's no explicit stop-loss setting in the code, the strategy automatically closes positions when the Supertrend indicators reverse, effectively acting as a dynamic stop-loss.

#### Strategy Advantages

1. Optimized Risk Management: The multi-step trailing take-profit mechanism greatly improves the strategy's risk-reward ratio. By progressively locking in profits, the strategy can reduce drawdown risk while maintaining upside potential.

2. Reduced False Signals: The use of dual Supertrend indicators significantly reduces the impact of false signals, improving trading accuracy and reliability.

3. High Adaptability: The strategy can flexibly adjust trading direction and take-profit parameters based on user preferences and market conditions, making it suitable for various trading instruments and timeframes.

4. High Degree of Automation: The strategy is fully automated, from entry to take-profit and exit, minimizing the impact of emotions and operational errors.

5. Flexible Capital Management: By setting different take-profit ratios, the strategy achieves flexible capital management, ensuring quick partial profit realization while allowing the remaining position to continue benefiting from market movements.

#### Strategy Risks

1. Parameter Sensitivity: The strategy's performance heavily depends on the settings of Supertrend indicators and take-profit parameters. Inappropriate parameters may lead to overtrading or missing important opportunities.

2. Trend Dependency: As a trend-following strategy, it may frequently enter and exit in choppy markets, causing unnecessary trading costs.

3. Slippage Risk: In fast-moving markets, the execution of multi-step take-profits may be affected by slippage, resulting in actual execution prices deviating from expectations.

4. Over-optimization Risk: With multiple adjustable parameters, the strategy is prone to over-optimization, potentially leading to significant differences between backtesting results and live trading performance.

#### Strategy Optimization Directions

1. Introduce Volatility Filtering: Consider incorporating ATR or other volatility indicators to reduce trading frequency during low volatility periods, improving the strategy's adaptability to different market conditions.

2. Dynamic Parameter Adjustment: Explore using adaptive algorithms to dynamically adjust Supertrend parameters and take-profit targets for better adaptation to market changes.

3. Enhance Stop-Loss Mechanism: While Supertrend reversals provide some stop-loss functionality, consider adding more flexible stop-loss mechanisms, such as trailing stops, for better risk control.

4. Integrate Additional Technical Indicators: Consider incorporating other technical indicators like RSI or MACD to improve entry and exit accuracy through multi-indicator confluence.

5. Optimize Capital Management: Explore more sophisticated capital management strategies, such as dynamically adjusting position sizes based on account performance, to better balance risk and reward.

6. Backtesting Optimization: Conduct more comprehensive backtests, including performance analysis across different timeframes and market conditions, to identify the strategy's optimal application scenarios and parameter settings.

#### Conclusion

This multi-step trailing take-profit strategy, based on dual Supertrend indicators, achieves a balance between risk and reward through its flexible multi-step profit-taking mechanism. The strategy's main advantages lie in its excellent risk management capabilities and sensitivity to trends. However, users need to pay attention to parameter settings and market environment impacts when applying it. Through further optimization and refinement, this strategy has the potential to become a robust and reliable automated trading system. In practical application, it is recommended that traders conduct thorough backtesting and paper trading, and make appropriate parameter adjustments based on specific trading instruments and market conditions.

[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-05-21 00:00:00
end: 2024-06-20 00:00:00
period: 4h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Strategic Multi-Step Supertrend Trader - Strategy [presentTrading]", overlay=true )

// this strategy utilizes a double Supertrend indicator to determine entry and exit conditions for both long and short trades, with user-configurable take profit levels and trade direction settings. 
// The strategy dynamically updates highest and lowest prices during trades and exits positions based on multi-step profit targets or opposing Supertrend signals.


// User inputs for take profit settings
// Grouping Take Profit settings
useTakeProfit = input.bool(true, title="Use Take Profit", group="Take Profit Settings")
takeProfitPercent1 = input.float(6.0, title="Take Profit % Step 1", group="Take Profit Settings")
takeProfitPercent2 = input.float(12.0, title="Take Profit % Step 2", group="Take Profit Settings")
takeProfitPercent3 = input.float(18.0, title="Take Profit % Step 3", group="Take Profit Settings")
takeProfitPercent4 = input.float(50.0, title="Take Profit % Step 4", group="Take Profit Settings")

takeProfitAmount1 = input.float(12, title="Take Profit Amount % Step 1", group="Take Profit Settings")
takeProfitAmount2 = input.float(8, title="Take Profit Amount % Step 2", group="Take Profit Settings")
takeProfitAmount3 = input.float(4, title="Take Profit Amount % Step 3", group="Take Profit Settings")
takeProfitAmount4 = input.float(0, title="Take Profit Amount % Step 4", group="Take Profit Settings")

numberOfSteps = input.int(3, title="Number of Take Profit Steps", minval=1, maxval=4, group="Take Profit Settings")

// Grouping Trade Direction
tradeDirection = input.string("Both", title="Trade Direction", options=["Long", "Short", "Both"], group="Trade Direction")

// Grouping Supertrend settings
atrPeriod1 = input(10, title="ATR Length for Supertrend 1", group="Supertrend Settings")
factor1 = input.float(3.0, title="Factor for Supertrend 1", step=0.01, group="Supertrend Settings")

atrPeriod2 = input(5, title="ATR Length for Supertrend 2", group="Supertrend Settings")
factor2 = input.float(4.0, title="Factor for Supertrend 2", step=0.01, group="Supertrend Settings")


// Function to calculate Supertrend
supertrend(factor, atrPeriod) =>
    [a, direction] = ta.supertrend(factor, atrPeriod)
    direction

// Calculate Double Supertrend
supertrend1 = supertrend(factor1, atrPeriod1)
supertrend2 = supertrend(factor2, atrPeriod2)

// Entry conditions
longCondition = (supertrend1 < 0 and supertrend2 < 0) and (tradeDirection == "Long" or tradeDirection == "Both")
shortCondition = (supertrend1 > 0 and supertrend2 > 0) and (tradeDirection == "Short" or tradeDirection == "Both")

// Exit conditions
longExitCondition = (supertrend1 > 0 and supertrend2 > 0) and (tradeDirection == "Long" or tradeDirection == "Both")
shortExitCondition = (supertrend1 < 0 and supertrend2 < 0) and (tradeDirection == "Short" or tradeDirection == "Both")

// Variables to store the highest and lowest prices during the trade
var float highestPrice = na
var float lowestPrice = na

// Get the entry price from open trades
entryPrice = strategy.opentrades.entry_price(strategy.opentrades - 1)

// Reset highestPrice or lowestPrice when entering new trades
if (longCondition and strategy.position_size <= 0)
    highestPrice := na // Reset the highest price
    strategy.entry("My Long Entry Id", strategy.long) // Enter long position
    strategy.close("My Short Entry Id", "Short Exit") // Close short position if any

if (shortCondition and strategy.position_size >= 0)
    lowestPrice := na // Reset the lowest price
    strategy.entry("My Short Entry Id", strategy.short) // Enter short position
    strategy.close("My Long Entry Id", "Long Exit") // Close long position if any

// Exit trades based on conditions
if (longExitCondition and strategy.position_size > 0)
    strategy.close("My Long Entry Id", "Long Exit") // Exit long position

if (shortExitCondition and strategy.position_size < 0)
    strategy.close("My Short Entry Id", "Short Exit") // Exit short position

if (strategy.position_size > 0)
    // Update the highest price for long positions
    highestPrice := na(highestPrice) ? high : math.max(highestPrice, high)

    // Step 1
    if (useTakeProfit and numberOfSteps >= 1)
        strategy.exit("Take Profit 1", from_entry="My Long Entry Id", qty_percent=takeProfitAmount1, limit=entryPrice * (1 + takeProfitPercent1 / 100))
    // Step 2
    if (useTakeProfit and numberOfSteps >= 2)
        strategy.exit("Take Profit 2", from_entry="My Long Entry Id", qty_percent=takeProfitAmount2, limit=entryPrice * (1 + takeProfitPercent2 / 100))
    // Step 3
    if (useTakeProfit and numberOfSteps >= 3)
        strategy.exit("Take Profit 3", from_entry="My Long Entry Id", qty_percent=takeProfitAmount3, limit=entryPrice * (1 + takeProfitPercent3 / 100))
    // Step 4
    if (useTakeProfit and numberOfSteps == 4)
        strategy.exit("Take Profit 4", from_entry="My Long Entry Id", qty_percent=takeProfitAmount4, limit=entryPrice * (1 + takeProfitPercent4 / 100))

if (strategy.position_size < 0)
    // Update the lowest price for short positions
    lowestPrice := na(lowestPrice) ? low : math.min(lowestPrice, low)

    // Step 1
    if (useTakeProfit and numberOfSteps >= 1)
        strategy.exit("Take Profit 1", from_entry="My Short Entry Id", qty_percent=takeProfitAmount1, limit=entryPrice * (1 - takeProfitPercent1 / 100))
    // Step 2
    if (useTakeProfit and numberOfSteps >= 2)
        strategy.exit("Take Profit 2", from_entry="My Short Entry Id", qty_percent=takeProfitAmount2, limit=entryPrice * (1 - takeProfitPercent2 / 100))
    // Step 3
    if (useTakeProfit and numberOfSteps >= 3)
        strategy.exit("Take Profit 3", from_entry="My Short Entry Id", qty_percent=takeProfitAmount3, limit=entryPrice * (1 - takeProfitPercent3 / 100))
    // Step 4
    if (useTakeProfit and numberOfSteps == 4)
        strategy.exit("Take Profit 4", from_entry="My Short Entry Id", qty_percent=takeProfitAmount4, limit=entryPrice * (1 - takeProfitPercent4 / 100))

```

> Detail

https://www.fmz.com/strategy/454736

> Last Modified

2024-06-21 14:36:35
