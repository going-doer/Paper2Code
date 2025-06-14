
> Name

趋势跟踪与动量结合的双指标交易策略-Dual-Indicator-Trading-Strategy-Combining-Trend-Following-and-Momentum

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1061d8ce5190b17b8c1.png)

[trans]
#### 概述

本策略结合了趋势跟踪和动量分析两种方法,利用简单移动平均线(SMA)和移动平均收敛散度(MACD)指标来识别潜在的交易机会。策略通过Trendilo指标(一种基于SMA的趋势指示器)来确定整体市场趋势,同时使用MACD的零线交叉来捕捉短期动量变化。此外,策略还运用了平均真实范围(ATR)来动态设置止损和获利水平,以适应市场波动性的变化。

#### 策略原理

策略的核心逻辑基于以下几个关键元素:

1. Trendilo指标:使用50周期的简单移动平均线来确定中长期趋势方向。
2. MACD零线交叉:用于捕捉短期动量的变化,作为入场信号。
3. ATR止损/获利设置:利用14周期的ATR来动态调整风险管理参数。

具体而言,当MACD线从下方穿过零线(上穿),且收盘价高于Trendilo线时,触发做多信号。相反,当MACD线从上方穿过零线(下穿),且收盘价低于Trendilo线时,触发做空信号。入场后,策略使用基于ATR的止损和获利水平来管理风险和锁定利润。

#### 策略优势

1. 趋势确认:通过结合Trendilo和MACD,策略能够在确认整体趋势的同时捕捉短期动量变化,有效减少假信号。
2. 动态风险管理:使用ATR来设置止损和获利水平,使策略能够根据市场波动性自动调整,提高了策略的适应性。
3. 多周期分析:结合了中长期(Trendilo)和短期(MACD)指标,提供了更全面的市场视角。
4. 可视化支持:策略在图表上标记了买卖信号和趋势线,便于交易者直观理解市场状况。

#### 策略风险

1. 趋势反转风险:在强趋势市场中表现良好,但可能在横盘或快速反转的市场中产生亏损。
2. 参数敏感性:策略性能可能对输入参数(如Trendilo周期、ATR乘数等)的选择高度敏感。
3. 过度交易:在波动剧烈的市场中,可能会产生频繁的交易信号,增加交易成本。
4. 滞后性:由于使用移动平均线,策略可能在趋势初期错过部分机会。

#### 策略优化方向

1. 引入筛选器:可以添加额外的技术指标或市场情绪指标,以过滤掉低质量的交易信号。
2. 优化参数选择:通过历史数据回测,寻找最优的Trendilo周期和ATR乘数组合。
3. 加入波动率调整:根据当前市场波动率动态调整策略参数,提高策略的适应性。
4. 实现部分仓位管理:考虑根据信号强度或市场条件调整每次交易的仓位大小。
5. 增加时间过滤:添加交易时间窗口限制,避开波动性较大或流动性较差的时段。

#### 总结

该策略巧妙地结合了趋势跟踪和动量分析,通过Trendilo和MACD的协同作用,为交易者提供了一个相对全面的市场分析框架。动态的风险管理方法增强了策略的适应性,使其能够在不同的市场环境中保持稳定性。然而,交易者在使用此策略时仍需谨慎,特别是在参数优化和风险控制方面。通过持续的监控和优化,该策略有潜力成为一个可靠的交易工具,特别适合那些希望在趋势市场中捕捉机会的投资者。

|| 

#### Overview

This strategy combines trend following and momentum analysis methods, utilizing Simple Moving Average (SMA) and Moving Average Convergence Divergence (MACD) indicators to identify potential trading opportunities. The strategy uses the Trendilo indicator (a trend indicator based on SMA) to determine the overall market trend, while employing MACD zero line crossovers to capture short-term momentum changes. Additionally, the strategy incorporates Average True Range (ATR) to dynamically set stop-loss and take-profit levels, adapting to changes in market volatility.

#### Strategy Principles

The core logic of the strategy is based on the following key elements:

1. Trendilo Indicator: Uses a 50-period simple moving average to determine medium to long-term trend direction.
2. MACD Zero Line Crossover: Used to capture changes in short-term momentum as entry signals.
3. ATR Stop-Loss/Take-Profit Setting: Utilizes a 14-period ATR to dynamically adjust risk management parameters.

Specifically, a long signal is triggered when the MACD line crosses above zero and the closing price is above the Trendilo line. Conversely, a short signal is triggered when the MACD line crosses below zero and the closing price is below the Trendilo line. After entry, the strategy uses ATR-based stop-loss and take-profit levels to manage risk and lock in profits.

#### Strategy Advantages

1. Trend Confirmation: By combining Trendilo and MACD, the strategy can capture short-term momentum changes while confirming the overall trend, effectively reducing false signals.
2. Dynamic Risk Management: Using ATR to set stop-loss and take-profit levels allows the strategy to automatically adjust to market volatility, enhancing its adaptability.
3. Multi-timeframe Analysis: Combines medium to long-term (Trendilo) and short-term (MACD) indicators, providing a more comprehensive market perspective.
4. Visual Support: The strategy marks buy and sell signals and trend lines on the chart, facilitating intuitive understanding of market conditions for traders.

#### Strategy Risks

1. Trend Reversal Risk: Performs well in strong trending markets but may generate losses in ranging or rapidly reversing markets.
2. Parameter Sensitivity: Strategy performance may be highly sensitive to the choice of input parameters (such as Trendilo period, ATR multipliers, etc.).
3. Overtrading: In highly volatile markets, frequent trading signals may be generated, increasing transaction costs.
4. Lagging Nature: Due to the use of moving averages, the strategy may miss some opportunities at the beginning of trends.

#### Strategy Optimization Directions

1. Introduce Filters: Additional technical indicators or market sentiment indicators can be added to filter out low-quality trading signals.
2. Optimize Parameter Selection: Through historical data backtesting, find the optimal combination of Trendilo period and ATR multipliers.
3. Incorporate Volatility Adjustment: Dynamically adjust strategy parameters based on current market volatility to improve strategy adaptability.
4. Implement Partial Position Management: Consider adjusting the size of each trade based on signal strength or market conditions.
5. Add Time Filtering: Implement trading time window restrictions to avoid periods of high volatility or poor liquidity.

#### Conclusion

This strategy cleverly combines trend following and momentum analysis, providing traders with a relatively comprehensive market analysis framework through the synergy of Trendilo and MACD. The dynamic risk management approach enhances the strategy's adaptability, allowing it to maintain stability in different market environments. However, traders should still exercise caution when using this strategy, especially in terms of parameter optimization and risk control. Through continuous monitoring and optimization, this strategy has the potential to become a reliable trading tool, particularly suitable for investors looking to capture opportunities in trending markets.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2019-12-23 08:00:00
end: 2024-09-24 08:00:00
period: 1d
basePeriod: 1d
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("NNFX Trendilo + Zero MACD Strategy", overlay=true)

// --- Inputs ---
atrPeriod = input.int(14, title="ATR Period")
stopLossMultiplier = input.float(1.5, minval=0.0, maxval = 20.0, step = 0.1 ,title="Stop Loss Multiplier")
takeProfitMultiplier = input.float(2.0, minval=0.0 , maxval = 20.0, step = 0.1,title="Take Profit Multiplier")

// --- Trendilo ---
trendiloPeriod = input.int(50, title="Trendilo Period")
trendilo = ta.sma(close, trendiloPeriod)

// --- MACD ---
[macdLine, signalLine, _] = ta.macd(close, 12, 26, 9)
macdZeroCrossUp = ta.crossover(macdLine, 0)
macdZeroCrossDown = ta.crossunder(macdLine, 0)

// --- ATR for Stop Loss and Take Profit ---
atr = ta.atr(atrPeriod)
stopLoss = atr * stopLossMultiplier
takeProfit = atr * takeProfitMultiplier

// --- Trading Logic ---
longCondition = macdZeroCrossUp and close > trendilo
shortCondition = macdZeroCrossDown and close < trendilo

// --- Execute Long Trades ---
if (longCondition)
    strategy.entry("Buy", strategy.long)
    strategy.exit("Take Profit/Stop Loss", "Buy", limit=close + takeProfit, stop=close - stopLoss)

// --- Execute Short Trades ---
if (shortCondition)
    strategy.entry("Sell", strategy.short)
    strategy.exit("Take Profit/Stop Loss", "Sell", limit=close - takeProfit, stop=close + stopLoss)

// --- Plot Signals ---
plotshape(series=longCondition, location=location.belowbar, color=color.green, style=shape.labelup, text="Buy")
plotshape(series=shortCondition, location=location.abovebar, color=color.red, style=shape.labeldown, text="Sell")

// --- Plot Trendilo ---
plot(trendilo, color=color.blue, linewidth=2)

```

> Detail

https://www.fmz.com/strategy/468334

> Last Modified

2024-09-26 16:14:22
