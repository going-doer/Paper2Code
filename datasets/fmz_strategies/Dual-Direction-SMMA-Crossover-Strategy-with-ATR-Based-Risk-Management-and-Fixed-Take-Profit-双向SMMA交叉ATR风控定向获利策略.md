
> Name

Dual-Direction-SMMA-Crossover-Strategy-with-ATR-Based-Risk-Management-and-Fixed-Take-Profit-双向SMMA交叉ATR风控定向获利策略

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/cd8b0f81437a916c1a.png)

[trans]
#### 概述
这是一个基于SMMA(平滑移动平均线)的双向趋势跟踪策略。该策略利用价格与SMMA的交叉来产生多空信号,并结合ATR动态止损和固定获利目标来管理风险和收益。策略设计简洁而有效,适合不同时间周期的趋势跟踪交易。

#### 策略原理
策略核心是通过17周期SMMA与价格的交叉来捕捉趋势变化。当价格上穿SMMA时,开启多头头寸;当价格下穿SMMA时,开启空头头寸。出场管理采用三重机制:1)基于ATR的动态止损,设定为SMMA上下0.75倍ATR;2)固定获利目标,多头为1150点,空头为1500点;3)反向交叉信号平仓。这种组合既保护了利润,又给予趋势充分发展空间。

#### 策略优势
1. 信号系统稳定可靠:SMMA相比简单移动平均线更平滑,能有效减少虚假信号
2. 风险管理全面:结合ATR动态止损和固定获利目标,既适应市场波动性变化,又锁定合理收益
3. 双向交易:充分把握市场双向机会,提高资金利用效率
4. 可扩展性强:策略框架清晰,易于在不同市场和时间周期上实施
5. 操作规则明确:入场出场条件客观,减少主观判断带来的干扰

#### 策略风险
1. 震荡市场风险:在横盘震荡市场可能频繁交易导致损失
2. 滑点风险:固定点数获利目标在快速市场可能面临滑点
3. 趋势反转风险:强趋势突然反转时,ATR止损可能不够快
4. 参数依赖:SMMA周期和ATR倍数的选择对策略表现影响较大
5. 资金管理风险:固定百分比仓位在波动性变化时可能不够灵活

#### 策略优化方向
1. 引入趋势强度过滤:可添加ADX等指标筛选强趋势,减少震荡市场假信号
2. 动态获利目标:考虑使用ATR动态调整获利目标,更好适应市场状态
3. 改进仓位管理:引入波动率加权的仓位计算,优化资金利用效率
4. 多时间周期确认:增加更长周期趋势确认,提高交易质量
5. 市场环境适应:增加市场类型判断逻辑,在不同市场条件下调整策略参数

#### 总结
这是一个设计合理的趋势跟踪策略,通过SMMA交叉捕捉趋势,运用ATR进行风险控制,配合固定获利目标管理收益。策略逻辑清晰,实现简单,具有良好的可操作性和扩展性。虽然在震荡市场表现可能欠佳,但通过建议的优化方向可以进一步提升策略的稳定性和适应性。对于偏好趋势跟踪的交易者来说,这是一个值得关注的策略框架。
||
#### Overview
This is a dual-direction trend-following strategy based on SMMA (Smoothed Moving Average). The strategy generates long and short signals through price-SMMA crossovers, combining ATR-based dynamic stop-loss and fixed take-profit targets for risk and reward management. The strategy design is concise and effective, suitable for trend following across different timeframes.

#### Strategy Principles
The core mechanism relies on price crossovers with a 17-period SMMA to capture trend changes. Long positions are initiated when price crosses above SMMA, while short positions are taken when price crosses below SMMA. Exit management employs a triple mechanism: 1) ATR-based dynamic stop-loss set at 0.75 times ATR above/below SMMA; 2) Fixed take-profit targets of 1150 points for longs and 1500 points for shorts; 3) Reverse crossover signals for position closure. This combination both protects profits and allows trends to develop fully.

#### Strategy Advantages
1. Stable signal system: SMMA provides smoother signals compared to simple moving averages, effectively reducing false signals
2. Comprehensive risk management: Combines ATR-based dynamic stops with fixed profit targets, adapting to market volatility while securing reasonable profits
3. Dual-direction trading: Fully captures bilateral market opportunities, improving capital efficiency
4. High scalability: Clear strategy framework, easily implementable across different markets and timeframes
5. Clear operational rules: Objective entry and exit conditions minimize subjective interference

#### Strategy Risks
1. Choppy market risk: Frequent trades in ranging markets may lead to losses
2. Slippage risk: Fixed point profit targets may face slippage in fast markets
3. Trend reversal risk: ATR stops may not react quickly enough to sudden trend reversals
4. Parameter dependency: Strategy performance heavily relies on SMMA period and ATR multiplier choices
5. Money management risk: Fixed percentage position sizing may lack flexibility during volatility changes

#### Optimization Directions
1. Introduce trend strength filtering: Add indicators like ADX to screen for strong trends, reducing false signals in ranging markets
2. Dynamic profit targets: Consider using ATR to dynamically adjust profit targets for better market adaptation
3. Improved position sizing: Implement volatility-weighted position calculation for optimized capital efficiency
4. Multi-timeframe confirmation: Add longer timeframe trend confirmation to improve trade quality
5. Market environment adaptation: Include market type classification logic to adjust strategy parameters under different market conditions

#### Summary
This is a well-designed trend-following strategy that captures trends through SMMA crossovers, implements risk control using ATR, and manages profits with fixed targets. The strategy logic is clear, implementation is straightforward, with good operability and extensibility. While performance may suffer in ranging markets, the suggested optimization directions can further enhance strategy stability and adaptability. For traders who prefer trend following, this represents a noteworthy strategy framework.[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-02-20 00:00:00
end: 2025-02-17 08:00:00
period: 4h
basePeriod: 4h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("SMMA 17 Crossover Strategy (Long & Short, ATR SL & Fixed TP)", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=200)

// ? SMMA Calculation
smmaLength = 17
smma = 0.0
smma := na(smma[1]) ? ta.sma(close, smmaLength) : (smma[1] * (smmaLength - 1) + close) / smmaLength

// ? ATR Calculation (For Dynamic Stop-Loss)
atrLength = 14
atr = ta.rma(ta.tr(true), atrLength)

// ? Long Entry Condition
longCondition = ta.crossover(close, smma)  // ✅ Price crosses above SMMA

// ? Long Exit Condition
longExit = ta.crossunder(close, smma)  // ✅ Price crosses below SMMA

// ? ATR-Based Stop-Loss (Dynamic) for Long
longStopLoss = smma - (atr * 0.75)  // ✅ Stop Loss below SMMA

// ? Fixed Take Profit for Long (1150 Points)
var float longEntryPrice = na
var float longTakeProfit = na
if longCondition
    longEntryPrice := close
    longTakeProfit := longEntryPrice + 1150  // ✅ TP 1150 points above entry

// ? Short Entry Condition
shortCondition = ta.crossunder(close, smma)  // ✅ Price crosses BELOW SMMA (Short trade)

// ? Short Exit Condition
shortExit = ta.crossover(close, smma)  // ✅ Price crosses ABOVE SMMA (Close Short trade)

// ? ATR-Based Stop-Loss (Dynamic) for Short
shortStopLoss = smma + (atr * 0.75)  // ✅ Stop Loss above SMMA

// ? Fixed Take Profit for Short (1500 Points) - Updated from 2000
var float shortEntryPrice = na
var float shortTakeProfit = na
if shortCondition
    shortEntryPrice := close
    shortTakeProfit := shortEntryPrice - 1500  // ✅ TP 1500 points below entry (Updated)

// ? Plot SMMA (For Visualization)
plot(smma, title="SMMA (17)", color=color.blue)

// ? Long Entry (Allow Multiple)
if longCondition
    strategy.entry("Long", strategy.long)

// ? Long Exit Conditions (Whichever Comes First)
strategy.exit("Long TP/SL", from_entry="Long", stop=longStopLoss, limit=longTakeProfit)
if longExit
    strategy.close("Long")

// ? Short Entry (Allow Multiple)
if shortCondition
    strategy.entry("Short", strategy.short)

// ? Short Exit Conditions (Whichever Comes First)
strategy.exit("Short TP/SL", from_entry="Short", stop=shortStopLoss, limit=shortTakeProfit)
if shortExit
    strategy.close("Short")

```

> Detail

https://www.fmz.com/strategy/482592

> Last Modified

2025-02-19 10:59:14
