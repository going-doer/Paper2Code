
> Name

动态趋势跟踪多周期均线交叉策略-Dynamic-Trend-Following-Multi-Period-Moving-Average-Crossover-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/9000bb17aaeaefd868.png)

[trans]
#### 概述
该策略是一个基于多周期均线的趋势跟踪交易系统。策略利用89周期和21周期简单移动平均线(SMA)确定市场总体趋势方向,同时结合5周期指数移动平均线(EMA)的高点和低点来寻找具体的交易信号。策略采用双重仓位管理方式,并结合固定止损和追踪止盈来控制风险。

#### 策略原理
策略的核心逻辑包含以下几个关键要素:
1. 趋势判断:使用89周期和21周期SMA的相对位置以及价格位置来确定趋势。当价格和5周期EMA都位于21周期SMA之上,且21周期SMA位于89周期SMA之上时,判定为上升趋势;反之则为下降趋势。
2. 入场信号:在上升趋势中,当价格回调至5周期EMA低点时入场做多;在下降趋势中,当价格反弹至5周期EMA高点时入场做空。
3. 仓位管理:每次信号触发时开立两个相同数量的合约仓位。
4. 风险控制:对第一个仓位采用固定止损和获利目标,对第二个仓位采用追踪止损方式管理。

#### 策略优势
1. 多重时间周期确认:通过不同周期的移动平均线组合,能够更全面地判断市场趋势,减少虚假信号。
2. 灵活的止盈方式:结合固定止盈和追踪止盈,既能在短期波动中获利了结,又不会错过大趋势行情。
3. 风险可控:设置明确的止损位置,且每个交易信号的风险敞口固定。
4. 系统化操作:交易规则明确,不受主观判断影响,易于程序化实现。

#### 策略风险
1. 震荡市风险:在横盘整理行情中,频繁的均线交叉可能导致过多假信号。
2. 滑点风险:在市场波动较大时,实际成交价格可能与理论信号价格产生较大偏差。
3. 资金管理风险:固定合约数量的交易方式可能不适合所有资金规模。
4. 参数敏感性:移动平均线周期的选择对策略表现影响较大,需要针对不同市场进行优化。

#### 策略优化方向
1. 动态仓位管理:建议根据账户净值和市场波动率动态调整交易合约数量。
2. 市场环境过滤:增加趋势强度指标(如ADX),在震荡市场降低交易频率。
3. 止损优化:可考虑使用ATR动态调整止损距离,提高策略对不同市场环境的适应性。
4. 信号确认:增加成交量、动量等辅助指标,提高交易信号的可靠性。

#### 总结
该策略是一个结构完整的趋势跟踪系统,通过多周期均线组合捕捉市场趋势,并采用灵活的仓位管理和止盈止损方式控制风险。虽然存在一定的优化空间,但策略的基本框架具有较好的实用性和可扩展性。针对不同的交易品种和市场环境,可以通过调整参数和增加过滤条件来提升策略的稳定性。

|| 

#### Overview
This strategy is a trend-following trading system based on multiple-period moving averages. It utilizes 89-period and 21-period Simple Moving Averages (SMA) to determine the overall market trend direction, while incorporating 5-period Exponential Moving Average (EMA) highs and lows to identify specific trading signals. The strategy employs a dual position management approach combined with fixed stop-loss and trailing take-profit mechanisms.

#### Strategy Principles
The core logic includes the following key elements:
1. Trend Determination: Uses the relative position of 89-period and 21-period SMAs along with price position to identify trends. An uptrend is confirmed when price and 5-period EMA are above the 21-period SMA, which is above the 89-period SMA; the opposite confirms a downtrend.
2. Entry Signals: In uptrends, enter long positions when price retraces to the 5-period EMA low; in downtrends, enter short positions when price rebounds to the 5-period EMA high.
3. Position Management: Opens two identical contract positions for each triggered signal.
4. Risk Control: Applies fixed stop-loss and profit targets for the first position, and trailing stop-loss for the second position.

#### Strategy Advantages
1. Multiple Timeframe Confirmation: The combination of different period moving averages provides a more comprehensive trend assessment, reducing false signals.
2. Flexible Profit-Taking: Combines fixed and trailing take-profit methods to capture both short-term fluctuations and extended trends.
3. Controlled Risk: Sets clear stop-loss levels with fixed risk exposure for each trade signal.
4. Systematic Operation: Clear trading rules minimize subjective judgment, making it easy to implement programmatically.

#### Strategy Risks
1. Consolidation Risk: Frequent moving average crossovers during sideways markets may generate excessive false signals.
2. Slippage Risk: Significant price deviations between theoretical and actual execution prices during high volatility periods.
3. Money Management Risk: Fixed contract quantity trading may not suit all account sizes.
4. Parameter Sensitivity: Strategy performance heavily depends on moving average period selection, requiring optimization for different markets.

#### Optimization Directions
1. Dynamic Position Sizing: Recommend adjusting contract quantities based on account equity and market volatility.
2. Market Environment Filtering: Add trend strength indicators (like ADX) to reduce trading frequency in ranging markets.
3. Stop-Loss Enhancement: Consider using ATR for dynamic stop-loss adjustment to improve adaptability across different market conditions.
4. Signal Confirmation: Incorporate volume and momentum indicators to increase signal reliability.

#### Summary
This strategy represents a comprehensive trend-following system that captures market trends through multiple-period moving averages while implementing flexible position management and risk control methods. Although there is room for optimization, the basic framework demonstrates good practicality and scalability. The strategy's stability can be enhanced by adjusting parameters and adding filtering conditions for different trading instruments and market environments.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-11-12 00:00:00
end: 2024-12-11 08:00:00
period: 2h
basePeriod: 2h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © tobiashartemink2

//@version=5
strategy("High 5 Trading Technique", overlay=true)

// --- Input parameters ---
sma89Length = input.int(title="SMA 89 Length", defval=89)
sma21Length = input.int(title="SMA 21 Length", defval=21)
ema5HighLength = input.int(title="EMA 5 High Length", defval=5)
ema5LowLength = input.int(title="EMA 5 Low Length", defval=5)
contracts = input.int(title="Aantal Contracten", defval=1)
stopLossPoints = input.int(title="Stop Loss Points per Contract", defval=25)
takeProfitPoints = input.int(title="Take Profit Points per Contract", defval=25)

// --- Calculate moving averages ---
sma89 = ta.sma(close, sma89Length)
sma21 = ta.sma(close, sma21Length)
ema5High = ta.ema(high, ema5HighLength)
ema5Low = ta.ema(low, ema5LowLength)

// --- Identify trend and order of moving averages ---
longSetup = close > sma89 and close > sma21 and ema5High > sma21 and sma21 > sma89
shortSetup = close < sma89 and close < sma21 and ema5Low < sma21 and sma21 < sma89

// --- Entry signals ---
longTrigger = longSetup and close <= ema5Low
shortTrigger = shortSetup and close >= ema5High

// --- Entry orders ---
if (longTrigger)
    strategy.entry("Long 1", strategy.long, qty=contracts)
    strategy.entry("Long 2", strategy.long, qty=contracts)

if (shortTrigger)
    strategy.entry("Short 1", strategy.short, qty=contracts)
    strategy.entry("Short 2", strategy.short, qty=contracts)

// --- Stop-loss and take-profit for long positions ---
if (strategy.position_size > 0)
    strategy.exit("Exit Long 1", "Long 1", stop=strategy.position_avg_price - stopLossPoints, limit=strategy.position_avg_price + takeProfitPoints)
    strategy.exit("Exit Long 2", "Long 2", stop=strategy.position_avg_price - stopLossPoints, trail_offset=takeProfitPoints, trail_points=takeProfitPoints)

// --- Stop-loss and take-profit for short positions ---
if (strategy.position_size < 0)
    strategy.exit("Exit Short 1", "Short 1", stop=strategy.position_avg_price + stopLossPoints, limit=strategy.position_avg_price - takeProfitPoints)
    strategy.exit("Exit Short 2", "Short 2", stop=strategy.position_avg_price + stopLossPoints, trail_offset=takeProfitPoints, trail_points=takeProfitPoints)

// --- Plot moving averages ---
plot(sma89, color=color.blue, linewidth=2)
plot(sma21, color=color.red, linewidth=2)
plot(ema5High, color=color.green, linewidth=2)
plot(ema5Low, color=color.orange, linewidth=2)

```

> Detail

https://www.fmz.com/strategy/474962

> Last Modified

2024-12-13 10:40:30
