
> Name

历史新高突破结合月线均线过滤的趋势跟踪量化交易策略-Historical-High-Breakthrough-with-Monthly-Moving-Average-Filter-Trend-Following-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/16fa6f3711ab74299a1.png)

[trans]
#### 概述
该策略是一个基于历史新高突破和月线均线过滤的趋势跟踪策略。它通过监控价格是否突破之前的历史最高点来寻找买入信号，同时利用月线8周期简单移动平均线(8 SMA)作为卖出过滤条件，以此降低假突破带来的风险。这种策略设计理念符合"趋势延续性"这一市场特征，特别适合在强势上涨趋势中捕捉大级别行情。

#### 策略原理
策略的核心逻辑包含两个关键部分:
1. 买入信号：当最新收盘价突破前期历史最高点(不包含当前K线的最高价)时，系统产生买入信号。这个条件确保只在明确的上升趋势中入场。
2. 卖出信号：当月线收盘价跌破8周期简单移动平均线时，系统触发卖出信号。这个条件帮助及时止损，防止趋势反转造成更大损失。
策略还设计了信号状态跟踪机制，避免在同一状态下重复产生信号，提高了策略的稳定性。

#### 策略优势
1. 趋势把握能力强：通过历史新高突破判断，能够有效捕捉强势上涨趋势。
2. 风险控制完善：结合月线均线作为过滤条件，可以有效过滤虚假突破。
3. 信号稳定性高：通过lastSignal变量追踪信号状态，避免重复信号产生。
4. 可视化效果好：策略提供了清晰的图形界面，包括历史高点线、均线以及买卖信号标记。
5. 适应性强：策略可以应用于不同的时间周期和品种。

#### 策略风险
1. 滞后性风险：历史新高突破信号本质上具有一定滞后性，可能错过最佳入场时机。
2. 假突破风险：虽然有月线均线过滤，但在震荡市场中仍可能遭遇假突破。
3. 回撤风险：在趋势转折点，策略可能承受较大回撤。
4. 资金管理风险：策略未包含仓位管理机制，需要额外的资金管理规则。

#### 策略优化方向
1. 引入量能确认：可以添加成交量指标作为突破确认条件，提高信号可靠性。
2. 完善止损机制：可以设计更灵活的止损规则，如跟踪止损或波动率止损。
3. 添加仓位管理：根据市场波动率和趋势强度动态调整仓位大小。
4. 优化信号过滤：可以添加趋势强度指标，如ADX，进一步过滤弱势信号。
5. 增加时间过滤：可以添加时间周期过滤，避免在不适合的时间段交易。

#### 总结
这是一个设计合理、逻辑清晰的趋势跟踪策略。通过历史新高突破和月线均线的配合使用，既保证了对趋势的有效把握，又实现了风险的合理控制。虽然存在一定的滞后性和假突破风险，但通过建议的优化方向，策略的整体表现有望得到进一步提升。该策略特别适合在明确趋势的市场环境中应用，可以作为中长期投资的重要参考工具。

||

#### Overview
This strategy is a trend following system based on historical high breakthrough and monthly moving average filter. It generates buy signals by monitoring price breakouts above previous historical highs, while using the 8-period Simple Moving Average (8 SMA) on monthly timeframe as a sell filter to reduce false breakout risks. The strategy design aligns with the market characteristic of "trend continuation" and is particularly suitable for capturing major trends in strong upward markets.

#### Strategy Principles
The core logic consists of two key components:
1. Buy Signal: Generated when the latest closing price breaks above the previous historical high (excluding the current bar's high). This condition ensures entry only in clear upward trends.
2. Sell Signal: Triggered when the monthly closing price falls below the 8-period Simple Moving Average. This condition helps with timely stop-loss and prevents larger losses from trend reversals.
The strategy also includes a signal state tracking mechanism to avoid repeated signals in the same state, improving strategy stability.

#### Strategy Advantages
1. Strong Trend Capture: Effectively captures strong upward trends through historical high breakout detection.
2. Robust Risk Control: Incorporates monthly moving average as a filter to effectively screen out false breakouts.
3. High Signal Stability: Uses lastSignal variable to track signal states, preventing signal repetition.
4. Good Visualization: Provides clear graphical interface including historical high lines, moving averages, and buy/sell markers.
5. High Adaptability: Can be applied to different timeframes and instruments.

#### Strategy Risks
1. Lag Risk: Historical high breakout signals are inherently somewhat lagging, potentially missing optimal entry points.
2. False Breakout Risk: Despite monthly moving average filtering, false breakouts may still occur in ranging markets.
3. Drawdown Risk: Strategy may experience significant drawdowns at trend reversal points.
4. Position Management Risk: Strategy lacks position sizing mechanisms, requiring additional money management rules.

#### Strategy Optimization Directions
1. Volume Confirmation: Add volume indicators as breakout confirmation conditions to improve signal reliability.
2. Enhanced Stop-Loss: Design more flexible stop-loss rules, such as trailing stops or volatility-based stops.
3. Position Management: Dynamically adjust position sizes based on market volatility and trend strength.
4. Signal Filtering: Add trend strength indicators like ADX to further filter weak signals.
5. Time Filtering: Add time period filters to avoid trading during unsuitable time periods.

#### Summary
This is a well-designed trend following strategy with clear logic. Through the combination of historical high breakouts and monthly moving averages, it achieves both effective trend capture and reasonable risk control. While there are inherent risks of lag and false breakouts, the suggested optimization directions offer potential for further performance improvement. The strategy is particularly suitable for markets with clear trends and can serve as an important reference tool for medium to long-term investment.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2019-12-23 08:00:00
end: 2024-12-11 08:00:00
period: 1d
basePeriod: 1d
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Buy Signal on Close Greater Than Previous All-Time High Strategy", overlay=true)

// Initialize the previous all-time high
var float prevAllTimeHigh = na

// Update the all-time high, excluding the current bar's high (use previous bar's high)
if (na(prevAllTimeHigh) or high[1] > prevAllTimeHigh)
    prevAllTimeHigh := high[1]

// Monthly closing price and 8 SMA on monthly time frame
monthlyClose = request.security(syminfo.tickerid, "M", close)
monthlySMA = ta.sma(monthlyClose, 8)

// Variables to track the last signal type
var int lastSignal = 0 // 0 = None, 1 = Buy, 2 = Sell

// Debugging output to check the all-time high and conditions
plot(prevAllTimeHigh, color=color.blue, linewidth=1, title="Previous All-Time High")
plot(monthlySMA, color=color.green, linewidth=1, title="8 SMA (Monthly)")

// Buy signal: when the latest close is greater than the previous all-time high
buySignal = close > prevAllTimeHigh and lastSignal != 1

// Sell signal: when the monthly close is below the 8 SMA
sellSignal = monthlyClose < monthlySMA and lastSignal != 2

// Update the last signal type after triggering a signal
if (buySignal)
    lastSignal := 1
if (sellSignal)
    lastSignal := 2

// Execute the strategy orders
if (buySignal)
    strategy.entry("Buy", strategy.long)

if (sellSignal)
    strategy.close("Buy")

// Optional: Plot buy and sell signals on the chart for visual reference
plotshape(series=buySignal, style=shape.labelup, location=location.belowbar, color=color.green, text="BUY", size=size.small)
plotshape(series=sellSignal, style=shape.labeldown, location=location.abovebar, color=color.red, text="SELL", size=size.small)

```

> Detail

https://www.fmz.com/strategy/474955

> Last Modified

2024-12-13 10:25:18
