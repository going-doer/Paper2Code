
> Name

SMA交叉多空策略结合峰值回撤控制与自动终止-SMA-Crossover-Long-Short-Strategy-with-Peak-Drawdown-Control-and-Auto-Termination

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/b4583dc611520d7634.png)

[trans]
#### 概述

这个策略是一个结合了简单移动平均线(SMA)交叉信号和峰值回撤控制的多空交易系统。它使用14期和28期SMA的交叉来生成多空交易信号,同时实时监控策略的峰值回撤情况。当回撤超过预设阈值时,策略会自动停止交易。此外,策略还包含了一个详细的峰值-谷底周期分析功能,可以帮助交易者更好地理解策略的风险特征。

#### 策略原理

1. 交易信号生成:
   - 当14期SMA上穿28期SMA时,产生做多信号。
   - 当14期SMA下穿28期SMA时,产生做空信号。

2. 峰值回撤控制:
   - 实时追踪策略的权益曲线,记录历史最高点(峰值)。
   - 当当前权益低于峰值时,进入回撤状态,记录最低点(谷底)。
   - 计算回撤百分比 = (峰值 - 谷底) / 峰值 * 100%。
   - 如果回撤百分比超过预设的最大回撤阈值,策略停止开新仓。

3. 峰值-谷底周期分析:
   - 设定最小回撤百分比,用于定义有效的峰值-谷底周期。
   - 每当完成一个有效周期时,记录周期编号、之前的上涨幅度、回撤幅度和结束时间。
   - 将分析结果以表格形式展示,便于交易者查看策略的历史表现。

#### 策略优势

1. 结合趋势跟踪和风险控制:
   SMA交叉策略是一种经典的趋势跟踪方法,而峰值回撤控制则提供了额外的风险管理层面。这种结合可以在捕捉市场趋势的同时,有效控制下行风险。

2. 自适应性强:
   通过参数化设置最大回撤和最小回撤阈值,策略可以根据不同市场环境和个人风险偏好进行灵活调整。

3. 透明的风险指标:
   峰值-谷底周期分析提供了详细的历史回撤信息,让交易者能够直观地了解策略的风险特征,有助于做出更明智的交易决策。

4. 自动化风险控制:
   当回撤超过预设阈值时,策略自动停止交易,这种机制可以有效防止在不利市场环境下持续亏损。

5. 全面的绩效分析:
   除了常规的回测指标,策略还提供了详细的峰值-谷底周期数据,包括上涨幅度、回撤幅度和时间信息,有助于深入分析策略表现。

#### 策略风险

1. 过度依赖历史数据:
   SMA交叉策略基于历史价格数据,在快速变化的市场中可能反应滞后,导致错误信号。

2. 频繁交易:
   在震荡市场中,SMA可能频繁交叉,导致过多交易和高昂的交易成本。

3. 潜在的大幅回撤:
   虽然有最大回撤控制,但在市场剧烈波动时,单次大跌仍可能导致较大损失。

4. 参数敏感性:
   策略性能高度依赖于SMA周期和回撤阈值的选择,不当的参数设置可能导致次优结果。

5. 错失反转机会:
   当达到最大回撤阈值停止交易后,策略可能错过市场反转带来的机会。

#### 策略优化方向

1. 引入动态参数调整:
   可以考虑根据市场波动率动态调整SMA周期和回撤阈值,以适应不同市场环境。

2. 增加额外的市场过滤器:
   结合其他技术指标或基本面因素,如RSI或成交量,来过滤潜在的虚假信号。

3. 实现分批入场和出场:
   而不是全仓操作,可以实现分批建仓和平仓,以减少单一决策的风险。

4. 加入止盈机制:
   在回撤控制的基础上,增加动态止盈功能,以锁定利润并提高整体收益率。

5. 优化资金管理:
   实现基于账户规模和市场波动性的动态仓位管理,以更好地控制风险。

6. 引入机器学习算法:
   使用机器学习技术优化参数选择和信号生成过程,提高策略的适应性和准确性。

#### 总结

SMA交叉多空策略结合峰值回撤控制与自动终止是一个兼具趋势跟踪和风险管理的量化交易系统。它通过简单移动平均线的交叉来捕捉市场趋势,同时利用峰值回撤控制来管理下行风险。策略的独特之处在于其详细的峰值-谷底周期分析功能,为交易者提供了深入了解策略风险特征的工具。

虽然策略存在一些固有的风险,如过度依赖历史数据和参数敏感性,但通过适当的优化和改进,如引入动态参数调整、增加额外的市场过滤器和实现更智能的资金管理,可以显著提高其稳健性和盈利能力。

总的来说,这个策略为交易者提供了一个良好的起点,可以在此基础上进行进一步的定制和优化,以满足个人的交易目标和风险偏好。策略的模块化设计也使得它易于与其他交易策略或风险管理技术集成,为构建更复杂、更全面的交易系统奠定了基础。

|| 

#### Overview

This strategy is a long-short trading system that combines Simple Moving Average (SMA) crossover signals with peak drawdown control. It uses the crossover of 14-period and 28-period SMAs to generate long and short trading signals while simultaneously monitoring the strategy's peak drawdown. When the drawdown exceeds a preset threshold, the strategy automatically stops trading. Additionally, the strategy includes a detailed peak-to-trough cycle analysis feature to help traders better understand the risk characteristics of the strategy.

#### Strategy Principle

1. Trade Signal Generation:
   - A long signal is generated when the 14-period SMA crosses above the 28-period SMA.
   - A short signal is generated when the 14-period SMA crosses below the 28-period SMA.

2. Peak Drawdown Control:
   - Real-time tracking of the strategy's equity curve, recording historical high points (peaks).
   - When current equity falls below the peak, it enters a drawdown state, recording the lowest point (trough).
   - Drawdown percentage is calculated as: (Peak - Trough) / Peak * 100%.
   - If the drawdown percentage exceeds the preset maximum drawdown threshold, the strategy stops opening new positions.

3. Peak-to-Trough Cycle Analysis:
   - Set a minimum drawdown percentage to define valid peak-to-trough cycles.
   - For each completed valid cycle, record the cycle number, previous run-up percentage, drawdown percentage, and end time.
   - Display analysis results in a table format for easy review of the strategy's historical performance.

#### Strategy Advantages

1. Combines Trend Following and Risk Control:
   The SMA crossover strategy is a classic trend-following method, while peak drawdown control provides an additional layer of risk management. This combination can effectively control downside risk while capturing market trends.

2. High Adaptability:
   By parameterizing the maximum drawdown and minimum drawdown thresholds, the strategy can be flexibly adjusted to different market environments and personal risk preferences.

3. Transparent Risk Indicators:
   The peak-to-trough cycle analysis provides detailed historical drawdown information, allowing traders to intuitively understand the strategy's risk characteristics, aiding in more informed trading decisions.

4. Automated Risk Control:
   When drawdown exceeds the preset threshold, the strategy automatically stops trading. This mechanism can effectively prevent continued losses in unfavorable market conditions.

5. Comprehensive Performance Analysis:
   In addition to conventional backtesting metrics, the strategy provides detailed peak-to-trough cycle data, including run-up percentages, drawdown percentages, and time information, facilitating in-depth analysis of strategy performance.

#### Strategy Risks

1. Over-reliance on Historical Data:
   The SMA crossover strategy is based on historical price data and may react slowly in rapidly changing markets, leading to false signals.

2. Frequent Trading:
   In oscillating markets, SMAs may cross frequently, resulting in excessive trading and high transaction costs.

3. Potential for Large Drawdowns:
   Despite maximum drawdown control, a single large drop during severe market volatility can still result in significant losses.

4. Parameter Sensitivity:
   Strategy performance is highly dependent on the choice of SMA periods and drawdown thresholds. Improper parameter settings may lead to suboptimal results.

5. Missing Reversal Opportunities:
   When trading stops after reaching the maximum drawdown threshold, the strategy may miss opportunities brought by market reversals.

#### Strategy Optimization Directions

1. Introduce Dynamic Parameter Adjustment:
   Consider dynamically adjusting SMA periods and drawdown thresholds based on market volatility to adapt to different market environments.

2. Add Additional Market Filters:
   Incorporate other technical indicators or fundamental factors, such as RSI or volume, to filter potential false signals.

3. Implement Phased Entry and Exit:
   Instead of all-or-nothing operations, implement phased position building and closing to reduce the risk of single decisions.

4. Add Take-Profit Mechanism:
   On top of drawdown control, add a dynamic take-profit function to lock in profits and improve overall returns.

5. Optimize Money Management:
   Implement dynamic position sizing based on account size and market volatility for better risk control.

6. Introduce Machine Learning Algorithms:
   Use machine learning techniques to optimize parameter selection and signal generation processes, improving strategy adaptability and accuracy.

#### Conclusion

The SMA crossover long-short strategy combined with peak drawdown control and auto-termination is a quantitative trading system that balances trend following and risk management. It captures market trends through simple moving average crossovers while managing downside risk using peak drawdown control. The strategy's unique feature lies in its detailed peak-to-trough cycle analysis, providing traders with a tool to deeply understand the strategy's risk characteristics.

While the strategy has some inherent risks, such as over-reliance on historical data and parameter sensitivity, it can significantly improve its robustness and profitability through appropriate optimization and improvements. These include introducing dynamic parameter adjustments, adding additional market filters, and implementing smarter money management.

Overall, this strategy provides traders with a good starting point that can be further customized and optimized to meet individual trading goals and risk preferences. The modular design of the strategy also makes it easy to integrate with other trading strategies or risk management techniques, laying the foundation for building more complex and comprehensive trading systems.

[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-07-23 00:00:00
end: 2024-07-28 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/

capital = 10000

//@version=5
strategy(title = "Correct Strategy Peak-Drawdown Cycles [Tradingwhale]", shorttitle = "Peak-Draw [Tradingwhale]", initial_capital = capital, overlay=true, margin_long=100, margin_short=100)

// The code below is from Tradingwhale LLC
/// ==============================================================================
//  Peak-Trough Cycles with Date and Prev. RunUp
// Initialize variables
showTable = input.bool(true, title = "Plot Peak to Bottom Drawdown Cycles table?")
min_trough = input.float(3.0, title = "Define Minimum Drawdown/Trough to Display (%)", minval = 1, maxval = 100, step = 0.5, tooltip = "Peaks and Trough Cycles have to be roped in by either a lookback period or minmimum troughs to show. If you don't then every bar could be a peak or trough/bottom. I've decided to use minimum declines here because lookback seems more arbitrary.")
maxdraw = input.float(40.0, title = "Max Drawdown", minval = 1, maxval = 100, step = 0.5, tooltip = "Define the drawdown level where the srtategy stops executing trades.")

var float equityPeak = na
var float equityTrough = na
var int cycleCount = 0
var bool inDrawdown = false
var float initialCapital = capital
var float prevTrough = initialCapital
var float prevRunUp = na
var bool useLighterGray = true
var int lastYear = na

// Variable to indicate whether the strategy should end
var bool end_strategy = false

// Table to display data
var table resultTable = table.new(position.top_right, 5, 30, bgcolor=#ffffff00, frame_color=#4f4040, frame_width=1)

// Function to convert float to percentage string
f_to_percent(value) =>
    str.tostring(value, "#.##") + "%"

// Function to get month/year string without commas
get_month_year_string() =>
    str.tostring(year) + "/" + str.tostring(month)

// Update the table headers
if (bar_index == 0 and showTable)
    table.cell(resultTable, 0, 0, "Show Min Trough: " + f_to_percent(min_trough), bgcolor=#a8a8a88f, text_size=size.normal)
    table.cell(resultTable, 1, 0, "Cycle Count", bgcolor=#a8a8a88f, text_size=size.normal)
    table.cell(resultTable, 2, 0, "Prev.RunUp(%)", bgcolor=#a8a8a88f, text_size=size.normal)
    table.cell(resultTable, 3, 0, "Drawdown(%)", bgcolor=#a8a8a88f, text_size=size.normal)
    table.cell(resultTable, 4, 0, "Year/Month", bgcolor=#a8a8a88f, text_size=size.normal)

// Track peaks and troughs in equity
if (na(equityPeak) or strategy.equity > equityPeak)
    if (inDrawdown and strategy.equity > equityPeak and not na(equityTrough)) // Confirm end of drawdown cycle
        drawdownPercentage = (equityPeak - equityTrough) / equityPeak * 100
        if drawdownPercentage > min_trough
            cycleCount += 1
            prevRunUp := (equityPeak - prevTrough) / prevTrough * 100
            if cycleCount <= 20 and showTable
                currentYear = year
                if na(lastYear) or currentYear != lastYear
                    useLighterGray := not useLighterGray
                    lastYear := currentYear
                rowColor = useLighterGray ? color.new(color.gray, 80) : color.new(color.gray, 50)
                table.cell(resultTable, 1, cycleCount, str.tostring(cycleCount), bgcolor=rowColor, text_size=size.normal)
                table.cell(resultTable, 2, cycleCount, f_to_percent(prevRunUp), bgcolor=rowColor, text_size=size.normal)
                table.cell(resultTable, 3, cycleCount, f_to_percent(drawdownPercentage), bgcolor=rowColor, text_size=size.normal)
                table.cell(resultTable, 4, cycleCount, get_month_year_string(), bgcolor=rowColor, text_size=size.normal)
            prevTrough := equityTrough
    equityPeak := strategy.equity
    equityTrough := na
    inDrawdown := false
else if (strategy.equity < equityPeak)
    equityTrough := na(equityTrough) ? strategy.equity : math.min(equityTrough, strategy.equity)
    inDrawdown := true

// Calculate if the strategy should end
if not na(equityPeak) and not na(equityTrough)
    drawdownPercentage = (equityPeak - equityTrough) / equityPeak * 100
    if drawdownPercentage >= maxdraw
        end_strategy := true


// This code below is from Tradingview, but with additions where commented (see below)

longCondition = ta.crossover(ta.sma(close, 14), ta.sma(close, 28))
if (longCondition) and not end_strategy // Add 'and not end_strategy' to your order conditions to automatically end the strategy if max_draw is exceeded/
    strategy.entry("My Long Entry Id", strategy.long)

shortCondition = ta.crossunder(ta.sma(close, 14), ta.sma(close, 28))
if (shortCondition) and not end_strategy // Add 'and not end_strategy' to your order conditions to automatically end the strategy if max_draw is exceeded/
    strategy.entry("My Short Entry Id", strategy.short)


```

> Detail

https://www.fmz.com/strategy/458038

> Last Modified

2024-07-29 14:16:58
