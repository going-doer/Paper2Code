
> Name

基于移动平均线和供需区域的动态风险管理交易系统-Moving-Average-Based-Supply-Demand-Zone-Trading-System-with-Dynamic-Risk-Management

> Author

ianzeng123

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/2d864836d1fa901c7542b.png)
![IMG](https://www.fmz.com/upload/asset/2d913b7242c39228235a6.png)



[trans]
#### 概述
这是一个结合了移动平均线交叉、供需区域识别以及动态止损止盈的综合交易策略。该策略通过短期和长期移动平均线的交叉来确定交易方向,同时利用供需区域作为重要的价格支撑位和阻力位,并配合百分比止损止盈来管理风险。策略的核心在于只在特定的供需区域附近开仓,从而提高交易的胜率。

#### 策略原理
策略使用9周期和21周期的简单移动平均线(SMA)来判断趋势方向。当价格处于需求区域(支撑位)1%范围内,且短期均线向上穿越长期均线时,系统发出做多信号;当价格处于供给区域(阻力位)1%范围内,且短期均线向下穿越长期均线时,系统发出做空信号。供需区域的识别基于50个周期内的显著高低点,并要求该点位具有至少2根确认蜡烛图。系统会根据入场价格自动设置动态的止损位(默认1%)和止盈位(默认2%)。

#### 策略优势
1. 多重确认机制: 结合了技术指标(均线交叉)和价格结构(供需区域),降低假突破风险
2. 动态风险管理: 止损止盈基于入场价格的百分比设置,适应不同市场环境
3. 可视化交易信号: 在图表上清晰标示供需区域和交易信号,便于分析和验证
4. 参数灵活可调: 均线周期、供需区域确认条件、止损止盈比例等都可根据不同市场特征调整
5. 策略逻辑清晰: 入场和出场条件明确,便于回测和优化

#### 策略风险
1. 震荡市场风险: 频繁的均线交叉可能导致过多假信号
2. 滑点风险: 在供需区域附近的交易可能面临较大滑点
3. 参数敏感性: 不同市场环境下最优参数可能差异较大
4. 止损幅度风险: 固定百分比止损可能不适合所有市场环境
5. 资金管理风险: 策略未包含头寸规模管理功能

#### 策略优化方向
1. 引入成交量确认: 在均线交叉和供需区域分析中加入成交量指标,提高信号可靠性
2. 动态参数优化: 根据市场波动率自动调整止损止盈比例和供需区域范围
3. 增加趋势过滤: 添加更长周期的趋势判断,避免在大趋势相反方向交易
4. 完善资金管理: 加入基于波动率的头寸规模计算
5. 增强供需区域识别: 引入更多技术指标来确认供需区域的有效性

#### 总结
这是一个将经典技术分析方法与现代风险管理理念相结合的策略系统。通过在重要价格区域附近进行交易,并结合移动平均线交叉信号,策略提供了一个相对可靠的交易框架。动态止损止盈的设计有助于适应不同市场环境,但策略的实际应用还需要根据具体市场特征进行优化。建议在实盘交易前进行充分的参数优化和回测验证。

|| 

#### Overview
This is a comprehensive trading strategy that combines moving average crossovers, supply/demand zone identification, and dynamic risk management. The strategy determines trade direction through short-term and long-term moving average crossovers, utilizes supply/demand zones as key support and resistance levels, and manages risk with percentage-based stop-loss and take-profit levels. The strategy's core principle is to only enter trades near specific supply/demand zones to improve win rate.

#### Strategy Principles
The strategy employs 9-period and 21-period Simple Moving Averages (SMA) to determine trend direction. Buy signals are generated when price is within 1% of a demand zone (support) and the short-term MA crosses above the long-term MA; sell signals occur when price is within 1% of a supply zone (resistance) and the short-term MA crosses below the long-term MA. Supply/demand zones are identified based on significant highs/lows within 50 periods, requiring at least 2 confirmation candles. The system automatically sets dynamic stop-loss (default 1%) and take-profit (default 2%) levels based on entry price.

#### Strategy Advantages
1. Multiple confirmation mechanism: Combines technical indicators (MA crossover) and price structure (supply/demand zones) to reduce false breakout risks
2. Dynamic risk management: Stop-loss and take-profit levels based on entry price percentages, adapting to different market conditions
3. Visual trade signals: Clear visualization of supply/demand zones and trade signals for analysis and verification
4. Flexible parameters: MA periods, supply/demand zone confirmation conditions, and risk management levels can be adjusted for different markets
5. Clear strategy logic: Well-defined entry and exit conditions for backtesting and optimization

#### Strategy Risks
1. Choppy market risk: Frequent MA crossovers may generate excessive false signals
2. Slippage risk: Trading near supply/demand zones may face significant slippage
3. Parameter sensitivity: Optimal parameters may vary significantly across different market conditions
4. Fixed stop-loss risk: Percentage-based stops may not suit all market environments
5. Money management risk: Strategy lacks position sizing functionality

#### Optimization Directions
1. Volume confirmation: Incorporate volume indicators in MA crossover and supply/demand zone analysis to improve signal reliability
2. Dynamic parameter optimization: Automatically adjust stop-loss/take-profit percentages and zone ranges based on market volatility
3. Trend filtering: Add longer-term trend analysis to avoid trading against major trends
4. Enhanced money management: Include volatility-based position sizing calculations
5. Improved zone identification: Introduce additional technical indicators to confirm supply/demand zone validity

#### Summary
This strategy system combines classical technical analysis methods with modern risk management concepts. By trading near significant price zones and incorporating moving average crossover signals, the strategy provides a relatively reliable trading framework. The dynamic stop-loss and take-profit design helps adapt to different market conditions, but practical application requires optimization based on specific market characteristics. Thorough parameter optimization and backtesting are recommended before live trading.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-12-01 00:00:00
end: 2025-02-01 00:00:00
period: 1h
basePeriod: 1h
exchanges: [{"eid":"Binance","currency":"ETH_USDT"}]
*/

//@version=5
strategy("MA Crossover with Demand/Supply Zones + Stop Loss/Take Profit", overlay=true)

// Input parameters for Moving Averages
shortLength = input.int(9, title="Short MA Length", minval=1)
longLength = input.int(21, title="Long MA Length", minval=1)

// Input parameters for Demand/Supply Zones
zoneLookback = input.int(50, title="Zone Lookback Period", minval=10)
zoneStrength = input.int(2, title="Zone Strength (Candles)", minval=1)

// Input parameters for Stop Loss and Take Profit
stopLossPerc = input.float(1.0, title="Stop Loss (%)", minval=0.1) / 100
takeProfitPerc = input.float(2.0, title="Take Profit (%)", minval=0.1) / 100

// Calculate moving averages
shortMA = ta.sma(close, shortLength)
longMA = ta.sma(close, longLength)

// Plot moving averages
plot(shortMA, color=color.blue, title="Short MA")
plot(longMA, color=color.red, title="Long MA")

// Identify Demand and Supply Zones
var float demandZone = na
var float supplyZone = na

// Detect Demand Zones (Price makes a significant low and bounces up)
if (ta.lowest(low, zoneLookback) == low[zoneStrength] and close[zoneStrength] > open[zoneStrength])
    demandZone := low[zoneStrength]

// Detect Supply Zones (Price makes a significant high and drops down)
if (ta.highest(high, zoneLookback) == high[zoneStrength] and close[zoneStrength] < open[zoneStrength])
    supplyZone := high[zoneStrength]

// Draw Demand and Supply Zones using lines
var line demandLine = na
var line supplyLine = na


// Trade Logic: Only open trades near Demand/Supply Zones
isNearDemand = demandZone > 0 and close <= demandZone * 1.01  // Within 1% of demand zone
isNearSupply = supplyZone > 0 and close >= supplyZone * 0.99  // Within 1% of supply zone

// Calculate Stop Loss and Take Profit levels
stopLossLevel = strategy.position_avg_price * (1 - stopLossPerc)  // Stop loss for long positions
takeProfitLevel = strategy.position_avg_price * (1 + takeProfitPerc)  // Take profit for long positions

stopLossLevelShort = strategy.position_avg_price * (1 + stopLossPerc)  // Stop loss for short positions
takeProfitLevelShort = strategy.position_avg_price * (1 - takeProfitPerc)  // Take profit for short positions

// Generate buy/sell signals based on MA crossover and zones
if (ta.crossover(shortMA, longMA) and isNearDemand)
    strategy.entry("Buy", strategy.long)
    strategy.exit("Take Profit/Stop Loss", from_entry="Buy", stop=stopLossLevel, limit=takeProfitLevel)

if (ta.crossunder(shortMA, longMA) and isNearSupply)
    strategy.entry("Sell", strategy.short)
    strategy.exit("Take Profit/Stop Loss", from_entry="Sell", stop=stopLossLevelShort, limit=takeProfitLevelShort)

// Optional: Plot buy/sell signals on the chart
plotshape(series=ta.crossover(shortMA, longMA) and isNearDemand, title="Buy Signal", location=location.belowbar, color=color.green, style=shape.labelup, text="BUY")
plotshape(series=ta.crossunder(shortMA, longMA) and isNearSupply, title="Sell Signal", location=location.abovebar, color=color.red, style=shape.labeldown, text="SELL")
```

> Detail

https://www.fmz.com/strategy/482859

> Last Modified

2025-02-20 15:39:27
