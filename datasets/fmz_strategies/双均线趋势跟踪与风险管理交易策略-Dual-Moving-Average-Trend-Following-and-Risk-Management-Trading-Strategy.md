
> Name

双均线趋势跟踪与风险管理交易策略-Dual-Moving-Average-Trend-Following-and-Risk-Management-Trading-Strategy

> Author

ianzeng123

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/2d82e2a073ee80879754e.png)
![IMG](https://www.fmz.com/upload/asset/2d966fa396f81ad7ac0b6.png)




[trans]
#### 概述
该策略是一个结合了多周期趋势跟踪和风险管理的自动化交易系统。它主要通过5分钟和1分钟两个时间周期的指数移动平均线(EMA)来识别交易机会,同时应用了固定百分比的止损和获利设置来控制风险。该策略特别适合于短线交易者,尤其是那些专注于趋势跟踪的交易者。

#### 策略原理
策略的核心逻辑基于两个时间周期的趋势判断:
1. 使用5分钟周期的200周期EMA作为主要趋势过滤器,只有当价格位于该均线之上时才允许做多,位于均线之下时才允许做空。
2. 在1分钟周期上,使用20周期EMA作为入场触发器。当价格向上穿过该均线时触发做多信号,向下穿过时触发做空信号。
3. 风险管理采用固定比例方法,每次交易的止损设置为入场价格的0.5%,获利目标设置为止损距离的2倍,形成1:2的风险收益比。

#### 策略优势
1. 多周期分析提供了更可靠的趋势判断,降低了假突破带来的风险。
2. 使用固定比例的风险管理方法,使得资金管理更加规范和系统化。
3. 1:2的风险收益比设置,即使胜率只有40%也可能实现盈利。
4. 策略逻辑简单明确,易于理解和执行。
5. 可视化的交易信号标记便于回测验证。

#### 策略风险
1. 快速震荡市场可能导致频繁假信号。
2. 在波动率较低的时期,0.5%的止损可能过于紧密。
3. 依赖均线交叉可能存在滞后性。
4. 高频交易可能带来较高的交易成本。
5. 市场快速反转时可能面临较大回撤。

#### 策略优化方向
1. 引入波动率指标来动态调整止损距离。
2. 增加成交量确认信号来提高入场质量。
3. 可以考虑加入趋势强度指标如ADX来过滤弱趋势。
4. 在横盘市场增加振荡指标如RSI来过滤信号。
5. 根据不同市场特征开发动态的风险收益比设置。

#### 总结
这是一个结构完整、逻辑清晰的趋势跟踪策略。通过结合多周期分析和严格的风险管理,策略在保护资金的同时,能够有效捕捉市场趋势。虽然存在一些优化空间,但策略的基本框架是稳健的,适合作为一个基础策略进行进一步改进和定制。

|| 

#### Overview
This strategy is an automated trading system that combines multi-timeframe trend following with risk management. It primarily uses Exponential Moving Averages (EMA) on 5-minute and 1-minute timeframes to identify trading opportunities, while implementing fixed percentage stops and targets for risk control. The strategy is particularly suitable for short-term traders, especially those focused on trend following.

#### Strategy Principles
The core logic is based on trend determination across two timeframes:
1. Uses a 200-period EMA on the 5-minute timeframe as the main trend filter, allowing long positions only above this line and short positions only below it.
2. On the 1-minute timeframe, a 20-period EMA serves as the entry trigger. Long signals are generated when price crosses above this EMA, and short signals when it crosses below.
3. Risk management employs a fixed percentage approach, with stops set at 0.5% from entry price and profit targets at twice the stop distance, creating a 1:2 risk-reward ratio.

#### Strategy Advantages
1. Multi-timeframe analysis provides more reliable trend identification, reducing false breakout risks.
2. Fixed percentage risk management approach ensures systematic capital management.
3. 1:2 risk-reward ratio can generate profits even with a 40% win rate.
4. Simple and clear strategy logic makes it easy to understand and execute.
5. Visual trade signal markers facilitate backtesting verification.

#### Strategy Risks
1. Rapid oscillating markets may generate frequent false signals.
2. The 0.5% stop loss might be too tight during low volatility periods.
3. Reliance on EMA crossovers can lead to lag in signal generation.
4. High-frequency trading may incur significant transaction costs.
5. Quick market reversals could result in substantial drawdowns.

#### Strategy Optimization Directions
1. Incorporate volatility indicators for dynamic stop loss adjustment.
2. Add volume confirmation to improve entry quality.
3. Consider adding trend strength indicators like ADX to filter weak trends.
4. Implement oscillators like RSI for ranging market signal filtration.
5. Develop dynamic risk-reward ratios based on market characteristics.

#### Summary
This is a well-structured trend following strategy with clear logic. By combining multi-timeframe analysis with strict risk management, the strategy effectively captures market trends while protecting capital. While there is room for optimization, the basic framework is robust and serves as an excellent foundation for further improvements and customization.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2025-01-21 00:00:00
end: 2025-02-20 00:00:00
period: 1h
basePeriod: 1h
exchanges: [{"eid":"Binance","currency":"DOGE_USDT"}]
*/

//@version=5
strategy("Scalping Strategy: 1-min Entries with 5-min 200 EMA Filter", overlay=true, initial_capital=10000, currency=currency.USD, default_qty_type=strategy.percent_of_equity, default_qty_value=5, calc_on_every_tick=true)

// --- Higher Timeframe Trend Filter ---
// Get the 200-period EMA on a 5-minute timeframe
ema200_5 = request.security(syminfo.tickerid, "5", ta.ema(close, 200), lookahead=barmerge.lookahead_on)
plot(ema200_5, color=color.purple, title="5-min 200 EMA")

// --- Local (1-Minute) Indicators ---
// On a 1-minute chart, calculate a 20-period EMA for entry triggers
ema20_1 = ta.ema(close, 20)
plot(ema20_1, color=color.yellow, title="1-min 20 EMA")

// --- Entry Conditions ---
// For long entries:
//   - The overall trend is bullish: current close > 5-min 200 EMA
//   - The 1-min candle closes and crosses above its 20 EMA
longCondition = (close > ema200_5) and ta.crossover(close, ema20_1)

// For short entries:
//   - Overall bearish trend: current close < 5-min 200 EMA
//   - 1-min candle crosses below its 20 EMA
shortCondition = (close < ema200_5) and ta.crossunder(close, ema20_1)

// --- Risk Management Settings ---
// For scalping, use a tight stop loss. Here we set risk at 0.5% of the entry price.
var float riskPerc = 0.005  // 0.5% risk per trade

// Declare global variables for stop loss and take profit so they can be used outside the if-blocks
var float longStop  = na
var float longTP    = na
var float shortStop = na
var float shortTP   = na

// --- Trade Execution --- 
if (longCondition)
    entryPrice = close
    // Stop loss for long: 0.5% below entry
    longStop := entryPrice * (1 - riskPerc)
    // Take profit: twice the risk distance (1:2 risk-reward)
    longTP   := entryPrice + 2 * (entryPrice - longStop)
    strategy.entry("Long", strategy.long)
    strategy.exit("Long Exit", from_entry="Long", stop=longStop, limit=longTP)

if (shortCondition)
    entryPrice = close
    // Stop loss for short: 0.5% above entry
    shortStop := entryPrice * (1 + riskPerc)
    // Take profit: twice the risk distance
    shortTP   := entryPrice - 2 * (shortStop - entryPrice)
    strategy.entry("Short", strategy.short)
    strategy.exit("Short Exit", from_entry="Short", stop=shortStop, limit=shortTP)

// --- Visual Debug Markers ---
// Plot a green triangle below bars when a long signal is generated
plotshape(longCondition, title="Long Signal", location=location.belowbar, color=color.green, style=shape.triangleup, size=size.tiny)
// Plot a red triangle above bars when a short signal is generated
plotshape(shortCondition, title="Short Signal", location=location.abovebar, color=color.red, style=shape.triangledown, size=size.tiny)
```

> Detail

https://www.fmz.com/strategy/483025

> Last Modified

2025-02-27 17:18:43
