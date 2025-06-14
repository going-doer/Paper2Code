
> Name

斐波那契回撤与延展多重指标量化交易策略-Fibonacci-Retracement-and-Extension-Multi-Indicator-Quantitative-Trading-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/14bbd3c75136d173766.png)

[trans]
#### 概述
本策略是一个基于斐波那契回撤和延展水平,结合EMA均线趋势判断的复合型量化交易系统。策略通过识别市场的重要支撑阻力位,结合趋势信号进行交易。系统使用20周期和50周期EMA均线判断市场趋势,并在此基础上利用斐波那契回撤水平寻找最佳交易机会。

#### 策略原理
策略的核心逻辑包含三个主要部分:首先计算近10个周期的最高价和最低价,用于确定价格波动区间;其次基于该区间计算五个关键的斐波那契回撤水平(0.236、0.382、0.5、0.618、0.786);最后通过20和50周期EMA的交叉确定趋势方向。在上升趋势中,当价格突破回撤水平时发出做多信号;在下降趋势中,当价格跌破回撤水平时发出做空信号。

#### 策略优势
1. 结合了趋势跟踪和价格回撤两种交易理念,提高了交易的准确性
2. 使用斐波那契数列作为关键价格水平,这些水平在市场中具有较强的心理意义
3. 通过EMA均线判断趋势,避免了在横盘市场中频繁交易
4. 系统设计简洁,易于理解和维护
5. 可以适用于不同的时间周期,具有较强的适应性

#### 策略风险
1. 在剧烈波动的市场中可能产生虚假信号
2. 依赖于趋势的持续性,在震荡市场中表现可能不佳
3. 回撤水平的计算基于历史高低点,可能滞后于市场
4. 入场点的选择可能不够精确,导致止损位置较远
5. 系统缺乏动态的仓位管理机制

#### 策略优化方向
1. 引入成交量指标,提高趋势判断的准确性
2. 增加动态止损机制,更好地控制风险
3. 优化回撤水平的计算周期,使其更符合市场节奏
4. 加入波动率过滤器,避免在高波动期间交易
5. 设计更灵活的仓位管理系统,根据市场条件调整持仓量

#### 总结
该策略通过结合经典的技术分析工具,构建了一个相对完整的交易系统。虽然存在一些需要优化的地方,但整体框架具有良好的市场适应性。通过持续优化和改进,该策略有望在实际交易中取得更好的表现。建议在实盘交易前,进行充分的历史数据回测和参数优化。

||

#### Overview
This strategy is a composite quantitative trading system based on Fibonacci retracement and extension levels, combined with EMA trend determination. The strategy identifies important support and resistance levels in the market and executes trades based on trend signals. The system uses 20-period and 50-period EMAs to determine market trends and utilizes Fibonacci retracement levels to find optimal trading opportunities.

#### Strategy Principles
The core logic consists of three main components: First, it calculates the highest and lowest prices over the past 10 periods to determine the price range; Second, it computes five key Fibonacci retracement levels (0.236, 0.382, 0.5, 0.618, 0.786) based on this range; Finally, it determines trend direction through the crossover of 20 and 50-period EMAs. Buy signals are generated when price breaks above retracement levels in an uptrend, while sell signals are triggered when price breaks below retracement levels in a downtrend.

#### Strategy Advantages
1. Combines trend following and price retracement concepts, improving trading accuracy
2. Uses Fibonacci sequences as key price levels, which hold strong psychological significance in markets
3. Employs EMA crossovers for trend identification, avoiding frequent trades in ranging markets
4. Features a clean system design that's easy to understand and maintain
5. Adaptable to different timeframes, demonstrating strong versatility

#### Strategy Risks
1. May generate false signals in highly volatile markets
2. Relies on trend continuation, potentially underperforming in ranging markets
3. Retracement level calculations based on historical highs and lows may lag behind the market
4. Entry point selection might not be precise enough, leading to wider stop losses
5. System lacks dynamic position management mechanisms

#### Strategy Optimization Directions
1. Incorporate volume indicators to improve trend determination accuracy
2. Implement dynamic stop-loss mechanisms for better risk control
3. Optimize retracement level calculation periods to better match market rhythm
4. Add volatility filters to avoid trading during high volatility periods
5. Design more flexible position management systems that adjust holdings based on market conditions

#### Summary
The strategy builds a relatively complete trading system by combining classic technical analysis tools. While there are areas for optimization, the overall framework demonstrates good market adaptability. Through continuous optimization and improvement, the strategy shows promise for better performance in actual trading. It is recommended to conduct thorough historical data backtesting and parameter optimization before live trading.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2019-12-23 08:00:00
end: 2024-11-11 00:00:00
period: 1d
basePeriod: 1d
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Fibonacci Retracement and Extension Strategy", overlay=true)

// Define the Fibonacci levels for retracement and extension
fibRetracementLevels = array.new_float(5)
array.set(fibRetracementLevels, 0, 0.236)
array.set(fibRetracementLevels, 1, 0.382)
array.set(fibRetracementLevels, 2, 0.5)
array.set(fibRetracementLevels, 3, 0.618)
array.set(fibRetracementLevels, 4, 0.786)

fibExtensionLevels = array.new_float(5)
array.set(fibExtensionLevels, 0, 1.618)
array.set(fibExtensionLevels, 1, 2.618)
array.set(fibExtensionLevels, 2, 3.618)
array.set(fibExtensionLevels, 3, 4.236)
array.set(fibExtensionLevels, 4, 5.618)

// Calculate the high and low prices for the last 10 bars
highPrice = ta.highest(high, 10)
lowPrice = ta.lowest(low, 10)

// Calculate the Fibonacci retracement levels
fibRetracement = array.new_float(5)
for i = 0 to 4
    array.set(fibRetracement, i, highPrice - (highPrice - lowPrice) * array.get(fibRetracementLevels, i))

// Calculate the trend using the Exponential Moving Average (EMA)
shortEMA = ta.ema(close, 20)
longEMA = ta.ema(close, 50)

// Define the trend conditions
isUptrend = shortEMA > longEMA
isDowntrend = shortEMA < longEMA

// Generate buy and sell signals
var float lastFibRetracementLevel = na
var float lastFibExtensionLevel = na

// Buy condition: price crosses above the highest retracement level
if (isUptrend)
    for i = 0 to 4
        if (close > array.get(fibRetracement, i))
            lastFibRetracementLevel := array.get(fibRetracement, i)
            strategy.entry("Buy", strategy.long)

// Sell condition: price crosses below the lowest retracement level
if (isDowntrend)
    for i = 0 to 4
        if (close < array.get(fibRetracement, i))
            lastFibRetracementLevel := array.get(fibRetracement, i)
            strategy.entry("Sell", strategy.short)

// Plotting the Fibonacci levels on the chart
// for i = 0 to 4
//     line.new(bar_index[10], array.get(fibRetracement, i), bar_index, array.get(fibRetracement, i), color=color.new(color.blue, 70), width=1)

// Plot the EMAs
plot(shortEMA, color=color.red, title="Short EMA")
plot(longEMA, color=color.blue, title="Long EMA")
```

> Detail

https://www.fmz.com/strategy/471662

> Last Modified

2024-11-12 10:51:02
