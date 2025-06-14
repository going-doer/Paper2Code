
> Name

趋势过滤多均线交叉ATR风控量化策略-Multi-Moving-Average-Crossover-Strategy-with-Trend-Filter-and-ATR-Risk-Management

> Author

ianzeng123

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/2d95ca9ea871461e878fd.png)
![IMG](https://www.fmz.com/upload/asset/2d81d0c078d2c68aafc5b.png)



[trans]

#### 概述

这是一个基于多条移动平均线交叉信号的量化交易策略，同时结合趋势过滤和ATR风险管理机制。该策略主要利用20周期简单移动平均线(SMA)与89周期指数移动平均线(EMA)的交叉来产生交易信号，并使用200周期简单移动平均线作为趋势过滤器，确保交易方向与主趋势一致。此外，策略采用平均真实波幅(ATR)来设置动态止损和止盈水平，有效控制每笔交易的风险回报比。

#### 策略原理

该策略的核心逻辑基于三条移动平均线和ATR指标的综合应用：

1. 移动平均线计算：
   - 20周期简单移动平均线(SMA)：反映短期价格趋势
   - 89周期指数移动平均线(EMA)：反映中期价格趋势
   - 200周期简单移动平均线(SMA)：作为长期趋势判断标准

2. 入场条件：
   - 多头入场：价格位于200周期移动平均线之上，且20周期SMA从下方向上穿越89周期EMA
   - 空头入场：价格位于200周期移动平均线之下，且20周期SMA从上方向下穿越89周期EMA

3. 风险管理设置：
   - 使用14周期ATR计算市场波动性
   - 止损位：入场价格 ± (ATR × 2)，多头在下方，空头在上方
   - 止盈位：入场价格 ± (ATR × 3)，多头在上方，空头在下方
   - 风险回报比固定为1:1.5

策略在图表上标记入场信号，并显示包含入场价格、止损和止盈水平的标签，便于交易者直观了解交易细节。

#### 策略优势

1. 多重趋势确认机制：通过三条不同周期的移动平均线，策略能够综合分析短期、中期和长期市场趋势，大幅降低假信号风险。

2. 顺势交易逻辑：200周期移动平均线作为趋势过滤器，确保只在主趋势方向上进行交易，避免逆势操作，提高胜率。

3. 动态风险管理：基于ATR的止损和止盈设置，能够根据市场实际波动性自动调整风险控制参数，在不同波动环境下保持策略的适应性。

4. 固定风险回报比：止损与止盈比例固定为2:3，确保每笔交易的预期收益大于预期风险，长期来看有利于资金增长。

5. 可视化交易信号：策略在图表上清晰标记入场点、止损点和止盈点，使交易决策过程更加直观和便捷。

6. 全自动化执行：策略逻辑清晰，易于编程实现，适合自动化交易系统部署，减少情绪干扰和人为操作失误。

#### 策略风险

1. 震荡市场表现不佳：在无明显趋势的横盘震荡市场中，移动平均线交叉可能产生频繁的假信号，导致连续止损。

2. 滞后性问题：所有基于移动平均线的策略都存在信号滞后的问题，可能错过趋势初期的最佳入场点，或在趋势逆转时反应不够迅速。

3. 固定倍数风险控制限制：虽然ATR能够反映市场波动，但固定的2倍ATR止损在某些极端行情下可能不足以避免重大损失，特别是在跳空行情中。

4. 参数优化困境：策略涉及多个参数（如20、89、200周期和ATR倍数），不同市场和时间框架可能需要不同的参数组合，存在过度拟合风险。

5. 趋势过滤器滞后：200周期移动平均线反应极其缓慢，可能导致趋势转变初期的错误判断，错失交易机会或产生错误信号。

针对这些风险，可以考虑以下解决方案：
- 增加市场环境识别机制，在震荡市场减少或暂停交易
- 引入其他技术指标作为确认信号，提高入场精度
- 考虑采用可变的ATR倍数或设置绝对最大亏损限制
- 引入自适应参数调整机制，根据不同市场条件自动优化参数

#### 策略优化方向

1. 市场环境自适应机制：引入波动率指标或趋势强度指标（如ADX），在不同市场环境下自动调整策略参数或暂停交易。这样可以解决策略在震荡市场表现不佳的问题。

2. 入场信号优化：可以考虑增加额外的确认指标，如RSI、MACD或成交量指标，只有在多重指标共同确认的情况下才入场，提高信号质量。

3. 动态风险管理：基于市场波动性和历史表现，实现自适应的止损和止盈倍数，在高波动市场增加止损距离，在低波动市场减小止损距离。

4. 部分止盈机制：引入分段止盈逻辑，在达到一定盈利目标后，移动止损至成本位或分批平仓，锁定部分利润的同时保留跟踪趋势的可能性。

5. 时间过滤器：增加交易时间过滤，避开重大经济数据公布或特定的低流动性时段，减少因市场异常波动引起的风险。

6. 资金管理优化：基于策略历史回测结果和当前市场条件，动态调整每笔交易的仓位大小，在有利条件下增加风险敞口，在不利条件下减少风险敞口。

7. 参数自优化：实现基于滚动回测的参数自动优化机制，定期根据最近市场数据调整移动平均线周期和ATR倍数，使策略持续适应变化的市场环境。

这些优化方向的核心目标是增强策略的自适应性和稳健性，减少对固定参数的依赖，提高在不同市场环境下的表现一致性。

#### 总结

趋势过滤多均线交叉ATR风控量化策略是一个结合了技术分析传统智慧和现代风险管理理念的交易系统。通过20/89/200三重移动平均线的配合，策略能够有效识别市场趋势并产生顺势交易信号；而基于ATR的动态风险控制机制，则确保了每笔交易具有合理的风险回报特性。

该策略最大的优势在于其系统性和纪律性，通过明确的规则消除了交易中的情绪因素，同时简洁的逻辑设计使其易于理解和执行。然而，策略也存在震荡市场表现不佳和信号滞后等固有缺陷，需要交易者在实际应用中保持警惕。

通过引入市场环境识别、多重确认信号和动态风险管理等优化措施，该策略有望在保持核心逻辑简洁的同时，实现更高的稳定性和适应性。无论是个人交易者还是机构投资者，都可以将此策略作为构建完整交易系统的基础框架，根据自身需求和风险偏好进行个性化调整。

最终，任何交易策略的成功都取决于严格的执行纪律和持续的优化改进。在市场环境不断变化的今天，保持对策略的监控和调整，比盲目追求完美参数更为重要。

|| 

#### Overview

This is a quantitative trading strategy based on multiple moving average crossover signals, combined with trend filtering and ATR risk management mechanisms. The strategy primarily uses the crossover of the 20-period Simple Moving Average (SMA) with the 89-period Exponential Moving Average (EMA) to generate trading signals, while employing the 200-period Simple Moving Average as a trend filter to ensure trade direction aligns with the main trend. Additionally, the strategy utilizes Average True Range (ATR) to set dynamic stop-loss and take-profit levels, effectively controlling the risk-reward ratio for each trade.

#### Strategy Principles

The core logic of this strategy is based on the comprehensive application of three moving averages and the ATR indicator:

1. Moving Average Calculations:
   - 20-period Simple Moving Average (SMA): Reflects short-term price trends
   - 89-period Exponential Moving Average (EMA): Reflects medium-term price trends
   - 200-period Simple Moving Average (SMA): Serves as a long-term trend judgment criterion

2. Entry Conditions:
   - Long Entry: Price is above the 200-period moving average, and the 20-period SMA crosses above the 89-period EMA
   - Short Entry: Price is below the 200-period moving average, and the 20-period SMA crosses below the 89-period EMA

3. Risk Management Settings:
   - Uses 14-period ATR to calculate market volatility
   - Stop Loss: Entry price ± (ATR × 2), below for long positions, above for short positions
   - Take Profit: Entry price ± (ATR × 3), above for long positions, below for short positions
   - Fixed risk-reward ratio of 1:1.5

The strategy marks entry signals on the chart and displays labels containing entry price, stop-loss, and take-profit levels, allowing traders to visually understand trade details.

#### Strategy Advantages

1. Multiple Trend Confirmation Mechanism: Through three moving averages of different periods, the strategy can comprehensively analyze short, medium, and long-term market trends, significantly reducing the risk of false signals.

2. Trend-Following Logic: The 200-period moving average serves as a trend filter, ensuring trades are only executed in the direction of the main trend, avoiding counter-trend operations and improving win rates.

3. Dynamic Risk Management: Stop-loss and take-profit settings based on ATR automatically adjust risk control parameters according to actual market volatility, maintaining strategy adaptability in different volatility environments.

4. Fixed Risk-Reward Ratio: The stop-loss to take-profit ratio is fixed at 2:3, ensuring that the expected return for each trade exceeds the expected risk, beneficial for long-term capital growth.

5. Visualized Trading Signals: The strategy clearly marks entry points, stop-loss points, and take-profit points on the chart, making the trading decision process more intuitive and convenient.

6. Fully Automated Execution: The strategy logic is clear and easy to program, suitable for automated trading system deployment, reducing emotional interference and human operational errors.

#### Strategy Risks

1. Poor Performance in Ranging Markets: In sideways, choppy markets without clear trends, moving average crossovers may generate frequent false signals, leading to consecutive stop-losses.

2. Lag Issues: All strategies based on moving averages suffer from signal lag problems, potentially missing optimal entry points at the beginning of trends or not responding quickly enough when trends reverse.

3. Fixed Multiplier Risk Control Limitations: Although ATR reflects market volatility, the fixed 2x ATR stop-loss may be insufficient to avoid significant losses in extreme market conditions, especially during gap openings.

4. Parameter Optimization Dilemma: The strategy involves multiple parameters (such as 20, 89, 200 periods and ATR multipliers), different markets and timeframes may require different parameter combinations, creating overfitting risks.

5. Trend Filter Lag: The 200-period moving average reacts extremely slowly, potentially leading to misjudgments during initial trend changes, missing trading opportunities or generating false signals.

To address these risks, consider the following solutions:
- Add market environment recognition mechanisms to reduce or pause trading in ranging markets
- Introduce other technical indicators as confirmation signals to improve entry precision
- Consider using variable ATR multipliers or setting absolute maximum loss limits
- Implement adaptive parameter adjustment mechanisms to automatically optimize parameters based on different market conditions

#### Strategy Optimization Directions

1. Market Environment Adaptive Mechanism: Introduce volatility indicators or trend strength indicators (such as ADX) to automatically adjust strategy parameters or pause trading in different market environments. This addresses the strategy's poor performance in ranging markets.

2. Entry Signal Optimization: Consider adding additional confirmation indicators such as RSI, MACD, or volume indicators, entering only when multiple indicators confirm, improving signal quality.

3. Dynamic Risk Management: Implement adaptive stop-loss and take-profit multipliers based on market volatility and historical performance, increasing stop-loss distance in high-volatility markets and decreasing it in low-volatility markets.

4. Partial Take-Profit Mechanism: Introduce staged take-profit logic to move stops to breakeven or partially close positions after reaching certain profit targets, securing partial profits while maintaining the possibility of following trends.

5. Time Filter: Add trading time filters to avoid major economic data releases or specific low-liquidity periods, reducing risks caused by abnormal market volatility.

6. Money Management Optimization: Dynamically adjust position size for each trade based on strategy historical backtest results and current market conditions, increasing risk exposure under favorable conditions and reducing it under unfavorable conditions.

7. Parameter Self-Optimization: Implement an automatic parameter optimization mechanism based on rolling backtests, periodically adjusting moving average periods and ATR multipliers according to recent market data, enabling the strategy to continuously adapt to changing market environments.

The core objective of these optimization directions is to enhance the strategy's adaptability and robustness, reduce dependence on fixed parameters, and improve performance consistency across different market environments.

#### Summary

The Multi-Moving Average Crossover Strategy with Trend Filter and ATR Risk Management combines traditional technical analysis wisdom with modern risk management concepts. Through the coordination of 20/89/200 triple moving averages, the strategy can effectively identify market trends and generate trend-following trading signals; while the dynamic risk control mechanism based on ATR ensures that each trade has reasonable risk-reward characteristics.

The strategy's greatest advantage lies in its systematic and disciplined approach, eliminating emotional factors in trading through clear rules, while its concise logical design makes it easy to understand and execute. However, the strategy also has inherent flaws such as poor performance in ranging markets and signal lag, requiring traders to remain vigilant in practical applications.

By introducing market environment recognition, multiple confirmation signals, and dynamic risk management optimization measures, this strategy has the potential to achieve greater stability and adaptability while maintaining its core logic simplicity. Both individual traders and institutional investors can use this strategy as a foundation framework for building complete trading systems, making personalized adjustments according to their needs and risk preferences.

Ultimately, the success of any trading strategy depends on strict execution discipline and continuous optimization improvements. In today's constantly changing market environment, maintaining monitoring and adjustment of the strategy is more important than blindly pursuing perfect parameters.

[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-03-26 00:00:00
end: 2025-03-25 00:00:00
period: 2h
basePeriod: 2h
exchanges: [{"eid":"Futures_Binance","currency":"ETH_USDT"}]
*/

//@version=5
strategy("EMA Cross Strategy (20MA & 89EMA with 200MA Filter)", overlay=true, initial_capital=10000, currency=currency.USD)

// 1. Moving Average Calculation
ma20  = ta.sma(close, 20)
ema89 = ta.ema(close, 89)
ma200 = ta.sma(close, 200)

// 2. Plot Moving Averages
plot(ma20, title="20MA", color=color.orange)
plot(ema89, title="89EMA", color=color.red)
plot(ma200, title="200MA", color=color.blue)

// 3. ATR and Multipliers
atrValue = ta.atr(14)
stopLossMultiplier  = 2.0   // Stop Loss: ATR × 2
takeProfitMultiplier = 3.0   // Take Profit: ATR × 3

// 4. Entry Signal Conditions
// Long Signal: Price is above the 200MA and 20MA crosses above 89EMA
longSignal  = (close > ma200) and (strategy.position_size == 0) and ta.crossover(ma20, ema89)
// Short Signal: Price is below the 200MA and 20MA crosses below 89EMA
shortSignal = (close < ma200) and (strategy.position_size == 0) and ta.crossunder(ma20, ema89)

// Plot Entry Signals (Circles for Reference)
plotshape(longSignal, title="Long Signal", style=shape.circle, location=location.belowbar, color=color.green, size=size.normal)
plotshape(shortSignal, title="Short Signal", style=shape.circle, location=location.abovebar, color=color.red, size=size.normal)

// 5. Position Entry and SL/TP Setup (Fixed ATR at Entry)
if longSignal
    entryPrice = close
    lockedATR  = atrValue
    longStopPrice = entryPrice - lockedATR * stopLossMultiplier
    longTakeProfitPrice = entryPrice + lockedATR * takeProfitMultiplier
    strategy.entry("Long", strategy.long)
    strategy.exit("Long_Exit", "Long", stop=longStopPrice, limit=longTakeProfitPrice)

if shortSignal
    entryPrice = close
    lockedATR  = atrValue
    shortStopPrice = entryPrice + lockedATR * stopLossMultiplier
    shortTakeProfitPrice = entryPrice - lockedATR * takeProfitMultiplier
    strategy.entry("Short", strategy.short)
    strategy.exit("Short_Exit", "Short", stop=shortStopPrice, limit=shortTakeProfitPrice)

```

> Detail

https://www.fmz.com/strategy/488268

> Last Modified

2025-03-26 13:42:55
