
> Name

双均线交叉策略-Dual-Moving-Average-Crossover-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1afceaa164a9b8804b9.png)
[trans]
#### 概述
双均线交叉策略是一个经典的趋势跟踪交易策略。该策略使用两条移动平均线,一条是快速移动平均线,另一条是慢速移动平均线。当快速移动平均线从下往上穿过慢速移动平均线时,称为"黄金交叉",表明上升趋势可能形成,此时开仓做多。当快速移动平均线从上往下穿过慢速移动平均线时,称为"死亡交叉",表明下降趋势可能形成,此时开仓做空。该策略代码支持使用简单移动平均线(SMA)和指数移动平均线(EMA),并且可以设置止损。

#### 策略原理 
该策略的核心是利用移动平均线的趋势特性和交叉信号来判断趋势方向和开仓时机。首先通过参数设置快速移动平均线(默认50)和慢速移动平均线(默认200)的周期,以及选择使用SMA还是EMA。然后计算两条移动平均线,判断它们的交叉情况:
1. 当快速移动平均线向上穿过慢速移动平均线(黄金交叉)时,若当前无持仓则开仓做多,同时设置止损价格(根据止损百分比计算)。
2. 当快速移动平均线向下穿过慢速移动平均线(死亡交叉)时,若当前无持仓则开仓做空,同时设置止损价格。
3. 若已有多头仓位,当死亡交叉发生时平仓。
4. 若已有空头仓位,当黄金交叉发生时平仓。
通过移动平均线交叉信号开仓,并设置止损,以趋势跟踪的方式捕捉价格的中长期趋势。

#### 策略优势
1. 逻辑简单清晰,容易理解和实现,是趋势跟踪策略的基础。 
2. 通过两条不同周期的移动平均线的交叉,可以较好地判断趋势的形成和反转。
3. 支持SMA和EMA两种类型的移动平均线,可以灵活选择。
4. 设置了止损,一定程度上控制了损失风险。
5. 适合把握中长期趋势,trend-following风格。

#### 策略风险
1. 参数选择不当(如移动平均线周期选择不当)可能导致信号频繁或者趋势判断滞后。
2. 快速震荡行情可能导致频繁交易,表现不佳。
3. 趋势反转或结束时可能出现较大回撤。
4. 固定百分比止损可能无法很好地控制风险。

#### 策略优化方向  
1. 对参数进行优化,包括移动平均线周期、止损百分比等,提高稳定性和收益风险比。
2. 可以考虑引入ATR等与波动率相关的指标来动态调整止损位置。
3. 趋势确认后再开仓而不是交叉时立即开仓,或者再加入其他趋势确认指标辅助判断,提高趋势把握的准确性。
4. 可以通过加仓、减仓等资金管理策略来改进。
5. 考虑与其他信号结合,形成多因子策略。

#### 总结
双均线交叉策略是一个简单经典的趋势跟踪策略,通过两条不同周期移动平均线的交叉来判断趋势方向和开平仓时机,适合把握中长期趋势。但是固定参数可能在变化的市场环境中表现不稳定,需要进一步优化完善,如优化参数、改进止损、引入其他信号等,才能成为一个相对稳健的交易策略。该策略可以作为趋势策略的基础,并在此基础上不断改进和扩展。

|| 

#### Overview
The Dual Moving Average Crossover strategy is a classic trend-following trading strategy. It uses two moving averages: a fast-moving average and a slow-moving average. When the fast-moving average crosses above the slow-moving average, it's called a "golden cross", indicating a potential uptrend, and a long position is opened. When the fast-moving average crosses below the slow-moving average, it's called a "death cross", indicating a potential downtrend, and a short position is opened. The strategy code supports the use of Simple Moving Average (SMA) and Exponential Moving Average (EMA), and allows setting stop-loss.

#### Strategy Principle
The core of this strategy is to use the trend characteristics of moving averages and crossover signals to determine the trend direction and entry timing. First, set the periods of the fast-moving average (default 50) and slow-moving average (default 200) through parameters, and choose to use SMA or EMA. Then calculate the two moving averages and determine their crossover situations:
1. When the fast-moving average crosses above the slow-moving average (golden cross), open a long position if there's no current position, and set the stop-loss price (calculated based on stop-loss percentage).
2. When the fast-moving average crosses below the slow-moving average (death cross), open a short position if there's no current position, and set the stop-loss price.
3. If there's an existing long position, close the position when a death cross occurs.
4. If there's an existing short position, close the position when a golden cross occurs.
Open positions based on moving average crossover signals and set stop-loss to capture the medium to long-term price trends in a trend-following manner.

#### Strategy Advantages
1. The logic is simple and clear, easy to understand and implement, and is the foundation of trend-following strategies.
2. By using the crossover of two moving averages with different periods, it can better determine the formation and reversal of trends.
3. It supports both SMA and EMA, which can be flexibly selected.
4. Setting stop-loss controls the risk of loss to a certain extent.
5. Suitable for capturing medium to long-term trends, trend-following style.

#### Strategy Risks
1. Improper parameter selection (such as inappropriate moving average periods) may lead to frequent signals or lagging trend judgment.
2. Fast fluctuating markets may lead to frequent trading and poor performance.
3. When the trend reverses or ends, there may be larger drawdowns.
4. Fixed percentage stop-loss may not control risks well.

#### Strategy Optimization Directions
1. Optimize parameters, including moving average periods, stop-loss percentage, etc., to improve stability and risk-return ratio.
2. Consider introducing volatility-related indicators such as ATR to dynamically adjust stop-loss positions.
3. Confirm the trend before opening a position instead of immediately opening at crossover, or add other trend confirmation indicators to assist judgment and improve the accuracy of trend capture.
4. Can be improved through money management strategies such as adding or reducing positions.
5. Consider combining with other signals to form a multi-factor strategy.

#### Summary
The Dual Moving Average Crossover strategy is a simple and classic trend-following strategy that determines the trend direction and entry/exit timing based on the crossover of two moving averages with different periods, suitable for capturing medium to long-term trends. However, fixed parameters may perform unstably in changing market environments and need further optimization and improvement, such as optimizing parameters, improving stop-loss, introducing other signals, etc., to become a relatively robust trading strategy. This strategy can serve as the foundation for trend strategies and be continuously improved and expanded upon.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-05-11 00:00:00
end: 2024-05-16 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
//==============================================================================
// A baseline strategy with a well known concept, golden cross & death cross.
// Support for both Simple & Exponential moving averages.
// Support for long & short stop losses as a percentage.:well
//==============================================================================
strategy("Basic Moving Average Crosses", overlay=true)

//------------------------------------------------------------------------------
// configuration
//------------------------------------------------------------------------------
maQuickLength = input(50, title="Quick MA Length") 
maSlowLength  = input(200, title="Quick MA Length") 
useSma        = input(true, title="Use SMA? If false, EMA is used.")

maQuick = useSma ? ta.sma(close, maQuickLength) : ta.ema(close, maQuickLength)
maSlow  = useSma ? ta.sma(close, maSlowLength) : ta.ema(close, maSlowLength)

stop_loss_percentage = input(2.0, title="Stop Loss (%)")

var float longStopLevel = na
var float shortStopLevel = na

bool isGoldenCross = ta.crossover(maQuick, maSlow)
bool isDeathCross  = ta.crossunder(maQuick, maSlow)

//------------------------------------------------------------------------------
// position opening logic
//------------------------------------------------------------------------------

if(strategy.position_size == 0)
    // Golden cross, enter a long position
    if(isGoldenCross)
        strategy.entry("Buy", strategy.long)
        longStopLevel := close - close * stop_loss_percentage/100.0
        strategy.exit("StopLossLong", "Buy", stop=longStopLevel)
    // Death cross, enter short position
    else if(isDeathCross)
        strategy.entry("Sell", strategy.short)
        shortStopLevel := close + close * stop_loss_percentage/100.0
        strategy.exit("StopLossShort", "Sell", stop=shortStopLevel)

//------------------------------------------------------------------------------
// position closing logic
//------------------------------------------------------------------------------
else
    // Close long position on death cross
    if(strategy.position_size > 0 and isDeathCross)
        strategy.close("Buy")
    
    // Close short position on golden cross
    else if(strategy.position_size < 0 and isGoldenCross)
        strategy.close("Sell")

//------------------------------------------------------------------------------
// ploting
//------------------------------------------------------------------------------
plot(maQuick, color=color.yellow)
plot(maSlow, color=color.blue)
```

> Detail

https://www.fmz.com/strategy/451733

> Last Modified

2024-05-17 15:48:04
