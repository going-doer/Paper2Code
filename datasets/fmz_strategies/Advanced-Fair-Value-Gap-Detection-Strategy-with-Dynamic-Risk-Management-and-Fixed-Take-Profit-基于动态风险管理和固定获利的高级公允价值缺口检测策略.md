
> Name

Advanced-Fair-Value-Gap-Detection-Strategy-with-Dynamic-Risk-Management-and-Fixed-Take-Profit-基于动态风险管理和固定获利的高级公允价值缺口检测策略

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/13e6bfcba3ff6117aab.png)

[trans]
#### 概述
这是一个基于公允价值缺口(FVG)的交易策略,结合了动态风险管理和固定获利目标。该策略在15分钟时间周期上运行,通过识别市场中的价格缺口来捕捉潜在的交易机会。根据回测数据显示,在2023年11月至2024年8月期间,该策略实现了284.40%的净收益率,总计完成153笔交易,其中盈利率达到71.24%,获利因子为2.422。

#### 策略原理
策略的核心是通过监测连续三根K线之间的价格关系来识别公允价值缺口。具体来说:
1. 多头FVG形成条件:当前一根K线的最高价低于前两根K线的最低价
2. 空头FVG形成条件:当前一根K线的最低价高于前两根K线的最高价
3. 入场信号由FVG阈值参数控制,只有当缺口大小超过价格一定百分比时才触发
4. 风险控制采用账户权益的固定比例(1%)作为止损标准
5. 获利目标采用固定点数(50点)设置

#### 策略优势
1. 风险管理科学合理：采用账户权益比例止损,可以实现动态风险控制
2. 交易规则明确：使用固定的获利目标,避免主观判断
3. 性能表现优异：较高的盈利率和获利因子表明策略具有良好的稳定性
4. 实现方式简单：代码逻辑清晰,易于理解和维护
5. 适应性强：可以通过参数调整适应不同的市场环境

#### 策略风险
1. 市场波动风险：在高波动市场中,固定点数的获利目标可能不够灵活
2. 滑点风险：频繁交易可能导致较高的滑点成本
3. 参数依赖：策略表现强烈依赖于FVG阈值的设置
4. 假突破风险：部分FVG信号可能是假突破,需要额外的确认指标
5. 资金管理风险：固定比例止损在连续亏损时可能导致资金快速缩水

#### 策略优化方向
1. 引入市场波动率指标,动态调整获利目标
2. 增加趋势过滤器,避免在横盘市场中交易
3. 开发多重时间周期确认机制
4. 优化仓位管理算法,引入浮动仓位系统
5. 增加交易时间过滤,避开高波动时段
6. 开发信号强度评分系统,筛选高质量交易机会

#### 总结
该策略通过结合公允价值缺口理论和科学的风险管理方法,展现出了良好的交易效果。策略的高盈利率和稳定的获利因子表明其具有实战价值。通过建议的优化方向,策略还有进一步提升空间。建议交易者在实盘使用前进行充分的参数优化和回测验证。 ||

#### Overview
This is a trading strategy based on Fair Value Gap (FVG) detection, combining dynamic risk management with fixed take profit targets. Operating on a 15-minute timeframe, the strategy identifies potential trading opportunities by detecting price gaps in the market. According to backtest data from November 2023 to August 2024, the strategy achieved a net profit of 284.40% with 153 total trades, maintaining a win rate of 71.24% and a profit factor of 2.422.

#### Strategy Principle
The core mechanism revolves around detecting Fair Value Gaps by monitoring price relationships across three consecutive candles:
1. Bullish FVG: When the high of the middle candle is below the low of the first candle
2. Bearish FVG: When the low of the middle candle is above the high of the first candle
3. Entry signals are controlled by an FVG threshold parameter
4. Risk control uses a fixed percentage (1%) of account equity as stop loss
5. Take profit is set at a fixed 50 points

#### Strategy Advantages
1. Scientific risk management: Uses account equity percentage for stop loss
2. Clear trading rules: Fixed take profit targets eliminate subjective judgment
3. Excellent performance: High win rate and profit factor indicate strategy stability
4. Simple implementation: Clear code logic, easy to understand and maintain
5. High adaptability: Can be adjusted for different market conditions

#### Strategy Risks
1. Market volatility risk: Fixed point take profit may be inflexible in highly volatile markets
2. Slippage risk: Frequent trading may lead to higher slippage costs
3. Parameter dependency: Performance heavily relies on FVG threshold settings
4. False breakout risk: Some FVG signals might be false breakouts
5. Money management risk: Fixed percentage stop loss might lead to quick drawdowns

#### Optimization Directions
1. Introduce volatility indicators for dynamic take profit adjustment
2. Add trend filters to avoid ranging market trades
3. Develop multiple timeframe confirmation mechanism
4. Optimize position sizing algorithm with floating position system
5. Add trading time filters to avoid high volatility periods
6. Develop signal strength scoring system for high-quality trade selection

#### Summary
This strategy demonstrates impressive results by combining Fair Value Gap theory with scientific risk management. The high win rate and stable profit factor indicate its practical value. Through the suggested optimization directions, there is potential for further improvement. Traders are advised to conduct thorough parameter optimization and backtesting before live implementation.[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2019-12-23 08:00:00
end: 2024-11-28 00:00:00
period: 1d
basePeriod: 1d
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Fair Value Gap Strategy with % SL and Fixed TP", overlay=true, initial_capital=500, default_qty_type=strategy.fixed, default_qty_value=1)

// Parameters
fvgThreshold = input.float(0.5, "FVG Threshold (%)", minval=0.1, step=0.1)

// Fixed take profit in pips
takeProfitPips = 50

// Function to convert pips to price
pipsToPriceChange(pips) =>
    syminfo.mintick * pips * 10

// Function to detect Fair Value Gap
detectFVG(dir) =>
    gap = 0.0
    if dir > 0  // Bullish FVG
        gap := low[2] - high[1]
    else  // Bearish FVG
        gap := low[1] - high[2]
    math.abs(gap) > (close * fvgThreshold / 100)

// Detect FVGs
bullishFVG = detectFVG(1)
bearishFVG = detectFVG(-1)

// Entry conditions
longCondition = bullishFVG
shortCondition = bearishFVG

// Calculate take profit level
longTakeProfit = strategy.position_avg_price + pipsToPriceChange(takeProfitPips)
shortTakeProfit = strategy.position_avg_price - pipsToPriceChange(takeProfitPips)

// Calculate stop loss amount (5% of capital)
stopLossAmount = strategy.equity * 0.01

// Execute trades
if (longCondition)
    strategy.entry("Long", strategy.long)

if (shortCondition)
    strategy.entry("Short", strategy.short)

// Set exit conditions
if (strategy.position_size > 0)
    strategy.exit("Long TP", "Long", limit=longTakeProfit)
    strategy.close("Long SL", when=strategy.openprofit < -stopLossAmount)
else if (strategy.position_size < 0)
    strategy.exit("Short TP", "Short", limit=shortTakeProfit)
    strategy.close("Short SL", when=strategy.openprofit < -stopLossAmount)

// Plot signals
plotshape(longCondition, "Buy Signal", location = location.belowbar, color = color.green, style = shape.triangleup, size = size.small)
plotshape(shortCondition, "Sell Signal", location = location.abovebar, color = color.red, style = shape.triangledown, size = size.small)
```

> Detail

https://www.fmz.com/strategy/473391

> Last Modified

2024-11-29 16:22:10
