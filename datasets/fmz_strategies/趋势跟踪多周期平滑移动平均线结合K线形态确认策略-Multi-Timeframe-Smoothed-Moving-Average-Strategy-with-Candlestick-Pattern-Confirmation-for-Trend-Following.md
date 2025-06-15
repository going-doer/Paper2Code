
> Name

趋势跟踪多周期平滑移动平均线结合K线形态确认策略-Multi-Timeframe-Smoothed-Moving-Average-Strategy-with-Candlestick-Pattern-Confirmation-for-Trend-Following

> Author

ianzeng123

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/2d8ef46879019048729f3.png)
![IMG](https://www.fmz.com/upload/asset/2d8733bacc449a4e51482.png)


[trans]
#### 概述
TMA策略是一种趋势跟踪交易系统，它巧妙地结合了多周期平滑移动平均线(SMMA)和K线形态分析，旨在识别高概率的交易机会。该策略采用21、50、100和200周期的平滑移动平均线作为趋势识别和支撑/阻力区域的基础，同时利用"三线反击"和"吞噬形态"这两种经典K线形态来确认入场信号。为了减少虚假信号并确保交易方向与主导趋势一致，策略还结合了2周期的指数移动平均线(EMA)作为动态趋势过滤器。此外，该策略还提供了可自定义的交易会话过滤功能，允许交易者选择在特定市场时段(如纽约、伦敦、东京等)内执行交易，以确保交易在流动性高的时间段内进行。

#### 策略原理
TMA策略的核心逻辑围绕多周期平滑移动平均线、K线形态确认和交易会话过滤展开。首先，策略计算四种不同周期(21、50、100和200)的平滑移动平均线，这些移动平均线共同构成了市场趋势的框架。其次，策略利用2周期EMA作为短期趋势指标，用于判断当前价格动向。

入场条件设计得非常严格，要求同时满足多个条件：
1. 对于多头入场：需要出现看涨吞噬形态或三线反击形态；价格必须位于200周期SMMA之上；2周期EMA必须确认上升趋势。
2. 对于空头入场：需要出现看跌吞噬形态或三线反击形态；价格必须位于200周期SMMA之下；2周期EMA必须确认下降趋势。

此外，如果启用了交易会话过滤器，还必须在指定的交易时段内才会执行入场操作。这种多层条件筛选的设计有效降低了错误信号的产生。

出场条件则相对简单明了：
- 多头仓位在2周期EMA跌破200周期SMMA时平仓。
- 空头仓位在2周期EMA突破200周期SMMA时平仓。

这种设计允许趋势充分发展，同时在趋势反转初期即时退出，有效保护已获利润。

#### 策略优势
TMA策略具有多方面的优势，使其成为一个强大的趋势跟踪工具：

1. **多层趋势确认**：通过组合使用多个时期的平滑移动平均线，策略能够全面评估市场趋势的强度和持续性，减少单一指标可能带来的误导。

2. **K线形态确认**：策略不仅依赖技术指标，还结合了经典的K线形态分析，这种双重确认机制显著提高了入场信号的可靠性。

3. **适应性强**：可调整的参数设置(如移动平均线周期、交易会话时段等)使策略能够适应不同市场和交易风格。

4. **风险管理完善**：基于移动平均线交叉的明确出场条件，为交易者提供了客观的风险控制机制，避免了主观判断可能导致的过度持仓。

5. **流动性管理**：通过交易会话过滤器，策略能够避开低流动性时段，降低滑点和价格操纵风险。

6. **减少噪音**：平滑移动平均线的使用减少了市场噪音的影响，使趋势信号更加清晰。

7. **多市场适用性**：策略设计适用于外汇、股票和加密货币等多种市场，尤其在较高时间框架(15分钟、1小时、4小时、日线)上效果更佳。

#### 策略风险
尽管TMA策略具有诸多优势，但也存在一些潜在风险需要注意：

1. **趋势延迟识别**：由于使用了移动平均线作为主要指标，可能导致趋势转变的信号相对滞后，错过趋势初期阶段的部分利润。解决方法：可以考虑结合更灵敏的指标(如MACD或RSI)来提前识别潜在的趋势转变。

2. **震荡市场表现欠佳**：作为趋势跟踪策略，在横盘整理或频繁震荡的市场环境中可能产生连续的亏损交易。解决方法：增加一个市场模式过滤器，在识别到震荡市场时暂停交易或调整为适合震荡市场的参数设置。

3. **假突破风险**：K线形态如吞噬形态和三线反击在某些情况下可能产生假信号。解决方法：可以增加额外的确认条件，如交易量确认或关键价格水平突破确认。

4. **过度优化风险**：多个可调参数可能导致过度拟合历史数据，但在未来市场表现不佳。解决方法：在不同市场和时间段进行充分的回测，并保持参数设置的稳健性。

5. **会话过滤器时区设置**：交易会话过滤器依赖于正确的时区设置，错误配置可能导致在不合适的时间段进行交易。解决方法：仔细验证时区设置，确保与目标市场的活跃时段一致。

#### 策略优化方向
基于对代码的深入分析，TMA策略还有以下几个可优化的方向：

1. **动态参数调整**：目前策略使用固定的移动平均线周期，可以考虑根据市场波动率自动调整这些参数。例如，在波动性较高的市场中使用更长的周期来减少噪音，在波动性较低的市场中使用更短的周期来提高灵敏度。这样可以使策略更好地适应不同市场条件。

2. **增加止损机制**：当前策略仅依靠移动平均线交叉作为出场条件，可以添加固定止损或追踪止损功能，以限制单笔交易的最大亏损，保护资金安全。

3. **引入波动率过滤器**：在入场条件中加入波动率指标(如ATR或标准差)，避免在异常波动期间进入市场，或根据波动率水平动态调整仓位大小，实现更精细的风险管理。

4. **优化交易量管理**：考虑根据趋势强度或信号质量调整仓位大小，而不是固定的资金百分比，这可以在高概率交易中增加收益，同时减少低概率交易的风险敞口。

5. **增加部分利润锁定机制**：当交易达到一定盈利时，可以考虑部分平仓或移动止损点到成本价，锁定部分利润，同时保留继续参与趋势的机会。

6. **多时间框架确认**：整合更高时间框架的趋势分析，只在更高时间框架趋势方向一致的情况下入场，这可以显著提高成功率，降低假突破的风险。

#### 总结
TMA策略是一个设计精良的趋势跟踪系统，通过多周期平滑移动平均线、K线形态确认和动态趋势过滤器的组合，为交易者提供了一种系统化的方法来识别和捕捉市场趋势。该策略特别注重确认机制，要求多重条件同时满足才执行交易，有效降低了误报率。

尽管存在一些固有的风险，如趋势识别延迟和震荡市场表现欠佳等问题，但这些风险可以通过本文提出的优化方向得到缓解。通过添加止损机制、波动率过滤器、多时间框架确认等功能，该策略的稳健性和适应性可以进一步提升。

最后，需要强调的是，任何交易策略都没有百分百的胜率，TMA策略也不例外。成功的交易不仅依赖于策略本身，还取决于交易者的纪律性、风险管理能力和对市场的理解。因此，建议交易者在实盘交易前，先在模拟账户中充分测试该策略，熟悉其特性和局限性，并根据个人风险承受能力和交易目标进行适当调整。

|| 

#### Overview
The TMA Strategy is a trend-following trading system that cleverly combines multiple-period Smoothed Moving Averages (SMMA) with candlestick pattern analysis to identify high-probability trading opportunities. The strategy employs 21, 50, 100, and 200-period smoothed moving averages as the foundation for trend identification and support/resistance zones, while utilizing the "3-line strike" and "engulfing pattern" classic candlestick formations to confirm entry signals. To reduce false signals and ensure trade direction aligns with the dominant trend, the strategy also incorporates a 2-period Exponential Moving Average (EMA) as a dynamic trend filter. Additionally, the strategy provides a customizable trading session filter function, allowing traders to choose to execute trades within specific market sessions (such as New York, London, Tokyo, etc.) to ensure transactions occur during high-liquidity time periods.

#### Strategy Principles
The core logic of the TMA Strategy revolves around multiple-period smoothed moving averages, candlestick pattern confirmation, and trading session filtering. First, the strategy calculates four different period (21, 50, 100, and 200) smoothed moving averages, which together form the framework of market trends. Second, the strategy uses a 2-period EMA as a short-term trend indicator to determine current price direction.

Entry conditions are designed to be very strict, requiring multiple conditions to be met simultaneously:
1. For long entries: A bullish engulfing pattern or 3-line strike pattern must appear; price must be above the 200-period SMMA; the 2-period EMA must confirm an uptrend.
2. For short entries: A bearish engulfing pattern or 3-line strike pattern must appear; price must be below the 200-period SMMA; the 2-period EMA must confirm a downtrend.

Furthermore, if the trading session filter is enabled, the entry operation must be executed within the specified trading session. This multi-layered condition filtering design effectively reduces the generation of erroneous signals.

Exit conditions are relatively straightforward:
- Long positions are closed when the 2-period EMA crosses below the 200-period SMMA.
- Short positions are closed when the 2-period EMA crosses above the 200-period SMMA.

This design allows trends to fully develop while exiting at the early stages of trend reversal, effectively protecting profits already gained.

#### Strategy Advantages
The TMA Strategy has multiple advantages that make it a powerful trend-following tool:

1. **Multi-layer Trend Confirmation**: By combining multiple periods of smoothed moving averages, the strategy can comprehensively assess the strength and persistence of market trends, reducing potential misleading signals from a single indicator.

2. **Candlestick Pattern Confirmation**: The strategy relies not only on technical indicators but also incorporates classic candlestick pattern analysis, and this dual confirmation mechanism significantly improves the reliability of entry signals.

3. **Strong Adaptability**: Adjustable parameter settings (such as moving average periods, trading session times, etc.) allow the strategy to adapt to different markets and trading styles.

4. **Comprehensive Risk Management**: Clear exit conditions based on moving average crossovers provide traders with an objective risk control mechanism, avoiding excessive position holding that might result from subjective judgment.

5. **Liquidity Management**: Through the trading session filter, the strategy can avoid low liquidity periods, reducing slippage and price manipulation risks.

6. **Noise Reduction**: The use of smoothed moving averages reduces the impact of market noise, making trend signals clearer.

7. **Multi-market Applicability**: The strategy design is applicable to various markets such as forex, stocks, and cryptocurrencies, particularly effective on higher timeframes (15-minute, 1-hour, 4-hour, daily).

#### Strategy Risks
Despite its many advantages, the TMA Strategy also has some potential risks that need attention:

1. **Delayed Trend Identification**: Due to the use of moving averages as the main indicator, signals of trend changes may be relatively lagging, missing part of the profits in the early stages of trends. Solution: Consider combining more sensitive indicators (such as MACD or RSI) to identify potential trend changes in advance.

2. **Poor Performance in Oscillating Markets**: As a trend-following strategy, it may produce consecutive losing trades in sideways or frequently oscillating market environments. Solution: Add a market mode filter that pauses trading or adjusts to parameters suitable for oscillating markets when such conditions are detected.

3. **False Breakout Risk**: Candlestick patterns such as engulfing patterns and 3-line strikes may produce false signals in some cases. Solution: Additional confirmation conditions can be added, such as volume confirmation or key price level breakout confirmation.

4. **Over-optimization Risk**: Multiple adjustable parameters may lead to overfitting historical data but perform poorly in future markets. Solution: Conduct thorough backtesting across different markets and time periods, and maintain robustness in parameter settings.

5. **Session Filter Timezone Setting**: The trading session filter relies on correct timezone settings, and incorrect configuration may lead to trading during inappropriate time periods. Solution: Carefully verify timezone settings to ensure consistency with the active sessions of the target market.

#### Strategy Optimization Directions
Based on in-depth analysis of the code, the TMA Strategy has the following potential optimization directions:

1. **Dynamic Parameter Adjustment**: Currently, the strategy uses fixed moving average periods. Consider automatically adjusting these parameters based on market volatility. For example, use longer periods in high-volatility markets to reduce noise and shorter periods in low-volatility markets to increase sensitivity. This would allow the strategy to better adapt to different market conditions.

2. **Add Stop Loss Mechanism**: The current strategy relies solely on moving average crossovers as exit conditions. Fixed stop loss or trailing stop loss functionality could be added to limit the maximum loss per trade, protecting capital safety.

3. **Introduce Volatility Filter**: Add volatility indicators (such as ATR or standard deviation) to entry conditions to avoid entering the market during abnormal volatility periods, or dynamically adjust position sizes based on volatility levels for more refined risk management.

4. **Optimize Position Management**: Consider adjusting position size based on trend strength or signal quality, rather than a fixed percentage of funds. This can increase returns on high-probability trades while reducing risk exposure on low-probability ones.

5. **Add Partial Profit-Locking Mechanism**: When a trade reaches a certain profit level, consider partial position closure or moving the stop loss to breakeven, locking in some profits while retaining the opportunity to continue participating in the trend.

6. **Multi-timeframe Confirmation**: Integrate trend analysis from higher timeframes, entering only when the trend direction is consistent with higher timeframes. This can significantly improve success rates and reduce false breakout risks.

#### Summary
The TMA Strategy is a well-designed trend-following system that provides traders with a systematic approach to identifying and capturing market trends through the combination of multiple-period smoothed moving averages, candlestick pattern confirmation, and dynamic trend filters. The strategy particularly emphasizes confirmation mechanisms, requiring multiple conditions to be met simultaneously before executing trades, effectively reducing the false signal rate.

Although there are some inherent risks, such as delayed trend identification and poor performance in oscillating markets, these risks can be mitigated through the optimization directions proposed in this article. By adding stop loss mechanisms, volatility filters, multi-timeframe confirmation, and other features, the robustness and adaptability of the strategy can be further enhanced.

Finally, it should be emphasized that no trading strategy has a 100% win rate, and the TMA Strategy is no exception. Successful trading depends not only on the strategy itself but also on the trader's discipline, risk management abilities, and understanding of the market. Therefore, it is recommended that traders thoroughly test this strategy in a demo account before live trading, familiarize themselves with its characteristics and limitations, and make appropriate adjustments according to individual risk tolerance and trading objectives.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2025-02-26 00:00:00
end: 2025-03-05 00:00:00
period: 1m
basePeriod: 1m
exchanges: [{"eid":"Futures_Binance","currency":"SOL_USDT"}]
*/

//@version=5
strategy("TMA Strategy", shorttitle="TMA Strategy", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.1)

// Smoothed MAs Inputs
len1 = input.int(21, title="Length 1", group="Smoothed MA Inputs")
src1 = input.source(close, title="Source 1", group="Smoothed MA Inputs")
len2 = input.int(50, title="Length 2", group="Smoothed MA Inputs")
src2 = input.source(close, title="Source 2", group="Smoothed MA Inputs")
h100 = input.bool(true, title="Show 100 Line", group="Smoothed MA Inputs")
len3 = input.int(100, title="Length 3", group="Smoothed MA Inputs")
src3 = input.source(close, title="Source 3", group="Smoothed MA Inputs")
len4 = input.int(200, title="Length 4", group="Smoothed MA Inputs")
src4 = input.source(close, title="Source 4", group="Smoothed MA Inputs")

// Calculate Smoothed MAs
smma1 = ta.sma(src1, len1)
plot(smma1, color=color.white, linewidth=2, title="21 SMMA")

smma2 = ta.sma(src2, len2)
plot(smma2, color=color.green, linewidth=2, title="50 SMMA")

smma3 = ta.sma(src3, len3)
plot(h100 ? smma3 : na, color=color.yellow, linewidth=2, title="100 SMMA")

smma4 = ta.sma(src4, len4)
plot(smma4, color=color.red, linewidth=2, title="200 SMMA")

// Trend Filter
ema2 = ta.ema(close, 2)

// 3 Line Strike Signals
bullSig = close[3] < open[3] and close[2] < open[2] and close[1] < open[1] and close > open[1]
bearSig = close[3] > open[3] and close[2] > open[2] and close[1] > open[1] and close < open[1]

// Engulfing Candles Signals
bullishEngulfing = open <= close[1] and open < open[1] and close > open[1]
bearishEngulfing = open >= close[1] and open > open[1] and close < open[1]

// Trading Session Filter
ts = input.bool(true, title="Enable Session Filter", group="Trade Session")
tz = input.string("America/Chicago", title="Timezone", options=["America/New_York", "America/Chicago", "Europe/London", "Europe/Frankfurt", "Asia/Tokyo", "Asia/Sydney", "UTC"], group="Trade Session")
startH = input.int(8, title="Session Start Hour", minval=0, maxval=23, group="Trade Session")
startM = input.int(30, title="Session Start Minute", minval=0, maxval=59, group="Trade Session")
endH = input.int(12, title="Session End Hour", minval=0, maxval=23, group="Trade Session")
endM = input.int(0, title="Session End Minute", minval=0, maxval=59, group="Trade Session")

startTime = timestamp(year, month, dayofmonth, startH, startM)
endTime = timestamp(year, month, dayofmonth, endH, endM)
inSession = (time >= startTime and time <= endTime)

// Entry Conditions
longCondition = (bullishEngulfing or bullSig) and (ema2 > smma4) and (not ts or inSession)
shortCondition = (bearishEngulfing or bearSig) and (ema2 < smma4) and (not ts or inSession)

// Exit Conditions
exitLong = ta.crossunder(ema2, smma4)
exitShort = ta.crossover(ema2, smma4)

// Strategy Execution
if (longCondition)
    strategy.entry("Long", strategy.long, comment="Long Entry")

if (shortCondition)
    strategy.entry("Short", strategy.short, comment="Short Entry")

if (exitLong)
    strategy.close("Long", comment="Exit Long")

if (exitShort)
    strategy.close("Short", comment="Exit Short")

// Debugging Plots
plotshape(series=longCondition, location=location.belowbar, color=color.green, style=shape.labelup, title="Long Signal")
plotshape(series=shortCondition, location=location.abovebar, color=color.red, style=shape.labeldown, title="Short Signal")

// Visuals
plot(ema2, color=color.blue, linewidth=1, title="EMA(2)")
bgcolor(inSession and ts ? color.new(color.blue, 90) : na, title="Session Background")

```

> Detail

https://www.fmz.com/strategy/485125

> Last Modified

2025-03-06 11:02:41
