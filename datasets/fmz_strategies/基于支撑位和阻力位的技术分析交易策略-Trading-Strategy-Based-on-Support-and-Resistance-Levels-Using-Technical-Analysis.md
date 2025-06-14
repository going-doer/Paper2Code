
> Name

基于支撑位和阻力位的技术分析交易策略-Trading-Strategy-Based-on-Support-and-Resistance-Levels-Using-Technical-Analysis

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1a4e3ea950138517099.png)

[trans]
#### 概述

该策略是一个基于技术分析的交易策略,利用支撑位和阻力位来做出交易决策。策略使用pivothigh()和pivotlow()指标来确定支撑位和阻力位,当收盘价高于阻力位时进行做多,当收盘价低于支撑位且前一个最高价也低于支撑位时进行做空。当价格反向穿过支撑位或阻力位时平仓。该策略适用于俄罗斯股票市场,使用日线数据。

#### 策略原理

1. 使用request.security()函数获取日线收盘价数据。
2. 使用ta.pivothigh()和ta.pivotlow()函数计算7日时间窗口内的支撑位和阻力位。
3. 当收盘价高于阻力位时,执行做多交易。
4. 当收盘价低于支撑位,且前一个最高价也低于支撑位时,执行做空交易。
5. 当价格反向穿过支撑位或阻力位时,平仓所有头寸。
6. 在图表上绘制支撑位和阻力位,以绿色和红色表示。

#### 策略优势

1. 该策略基于技术分析,利用市场价格行为来做出交易决策,适用于趋势性市场。
2. 支撑位和阻力位是市场参与者广泛认可的重要价位,策略围绕这些关键价位建立交易信号,有助于捕捉趋势性机会。
3. 策略逻辑清晰,易于理解和实现,适合初学者学习和使用。
4. 通过图表绘制支撑位和阻力位,可以直观地观察市场结构和价格行为,辅助交易决策。

#### 策略风险

1. 该策略完全依赖历史价格数据,在市场出现重大基本面变化或黑天鹅事件时,可能失效。
2. 支撑位和阻力位可能被突破,导致策略出现连续亏损。
3. 策略缺乏风险管理措施,如止损和头寸规模控制,在市场剧烈波动时可能导致大额亏损。
4. 策略在震荡市中表现可能欠佳,频繁交易可能导致高昂的交易成本。

#### 策略优化方向

1. 引入趋势确认指标,如移动平均线,以过滤噪音和识别主要趋势,提高信号质量。
2. 设置合理的止损位,控制单笔交易风险,提高策略稳健性。
3. 优化支撑位和阻力位的计算方法,如采用多时间尺度组合,以提高价位的可靠性。
4. 引入仓位管理和资金管理规则,根据市场波动性动态调整头寸规模,控制整体风险敞口。
5. 对策略进行参数优化和回测,找到最佳参数组合,提高策略绩效。

#### 总结

该策略是一个基于支撑位和阻力位的技术分析交易策略,通过识别关键价格区域来建立交易信号。策略逻辑清晰,适合初学者学习,但在实际应用中需要注意风险管理和优化。通过引入其他技术指标、风控措施和仓位管理等,可以进一步提升策略的稳健性和盈利能力。在实盘部署前,建议在历史数据上进行全面的回测和参数优化。

|| 

#### Overview

This strategy is a technical analysis-based trading strategy that uses support and resistance levels to make trading decisions. The strategy utilizes the pivothigh() and pivotlow() indicators to determine support and resistance levels. It goes long when the closing price is above the resistance level and goes short when the closing price is below the support level, and the previous high is also below the support level. Positions are closed when the price crosses the support or resistance levels in the opposite direction. The strategy is suitable for the Russian stock market and uses daily data.

#### Strategy Principle

1. Use the request.security() function to obtain daily closing price data.
2. Calculate support and resistance levels using the ta.pivothigh() and ta.pivotlow() functions with a 7-day time window.
3. Execute a long trade when the closing price is above the resistance level.
4. Execute a short trade when the closing price is below the support level, and the previous high is also below the support level.
5. Close all positions when the price crosses the support or resistance levels in the opposite direction.
6. Plot support and resistance levels on the chart, represented by green and red colors.

#### Strategy Advantages

1. The strategy is based on technical analysis and uses market price behavior to make trading decisions, suitable for trending markets.
2. Support and resistance levels are widely recognized by market participants as important price levels. The strategy builds trading signals around these key levels, helping to capture trending opportunities.
3. The strategy logic is clear, easy to understand and implement, making it suitable for beginners to learn and use.
4. By plotting support and resistance levels on the chart, market structure and price behavior can be visually observed, aiding in trading decisions.

#### Strategy Risks

1. The strategy relies entirely on historical price data and may fail when significant fundamental changes or black swan events occur in the market.
2. Support and resistance levels may be breached, leading to consecutive losses for the strategy.
3. The strategy lacks risk management measures, such as stop-loss and position sizing control, which may lead to substantial losses during extreme market volatility.
4. The strategy may perform poorly in choppy markets, and frequent trading may result in high transaction costs.

#### Strategy Optimization Directions

1. Introduce trend confirmation indicators, such as moving averages, to filter out noise and identify the main trend, improving signal quality.
2. Set reasonable stop-loss levels to control individual trade risk and enhance strategy robustness.
3. Optimize the calculation method for support and resistance levels, such as using a combination of multiple time scales, to improve the reliability of price levels.
4. Incorporate position sizing and money management rules to dynamically adjust position sizes based on market volatility and control overall risk exposure.
5. Perform parameter optimization and backtesting on the strategy to find the optimal parameter combination and improve strategy performance.

#### Summary

This strategy is a technical analysis-based trading strategy that uses support and resistance levels to generate trading signals. The strategy logic is straightforward, making it suitable for beginners to learn. However, when applying the strategy in practice, risk management and optimization need to be considered. By introducing other technical indicators, risk control measures, position sizing, and other enhancements, the robustness and profitability of the strategy can be further improved. Before deploying the strategy in a live trading environment, it is recommended to conduct comprehensive backtesting and parameter optimization on historical data.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-04-01 00:00:00
end: 2024-04-30 23:59:59
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Торговая стратегия от уровней", overlay=true)

// Функция для определения уровней поддержки и сопротивления
findSR() =>
    // Получаем данные для поиска уровней
    data = request.security(syminfo.tickerid, "D", close)
    // Находим уровни поддержки и сопротивления
    pivot_high = ta.pivothigh(data, 7, 7)
    pivot_low = ta.pivotlow(data, 7, 7)
    [pivot_high, pivot_low]

[support, resistance] = findSR()

// Условия входа в длинную позицию
longCondition = close > resistance
// Условия входа в короткую позицию
shortCondition = close < support and high[1] < support

// Условия выхода из позиции
exitCondition = close < resistance and close > support

// Отображение уровней поддержки и сопротивления на графике
plot(support, color=color.green, style=plot.style_stepline)
plot(resistance, color=color.red, style=plot.style_stepline)

// Вход в позицию
if (longCondition)
    strategy.entry("Длинная", strategy.long)
if (shortCondition)
    strategy.entry("Короткая", strategy.short)

// Выход из позиции
if (exitCondition)
    strategy.close("Длинная")
    strategy.close("Короткая")

```

> Detail

https://www.fmz.com/strategy/451028

> Last Modified

2024-05-11 11:53:34
