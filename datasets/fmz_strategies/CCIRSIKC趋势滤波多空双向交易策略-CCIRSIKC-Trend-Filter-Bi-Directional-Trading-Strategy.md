
> Name

CCIRSIKC趋势滤波多空双向交易策略-CCIRSIKC-Trend-Filter-Bi-Directional-Trading-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1a9ea437fbea8fd3397.png)

[trans]
#### 概述
本策略采用CCI、RSI和肯特纳通道(KC)三种技术指标,结合趋势过滤器,实现在AUDNZD和GBPNZD货币对上的多空双向交易。策略通过CCI和RSI来判断超买超卖情况,KC作为止损止盈的参考依据,同时使用移动平均线作为趋势过滤,在顺势的情况下进行开仓操作。该策略已经在过去5年的历史数据上进行了回测,取得了稳定的收益。

#### 策略原理
1. 计算CCI、RSI和KC指标。KC上轨为中线加上ATR,下轨为中线减去ATR。
2. 根据输入参数选择移动平均线类型(SMA、EMA、SMMA、CMA或TMA)和趋势过滤方法(关闭、正向或反向)。
3. 多头开仓条件:允许做多、CCI<超卖线、收盘价<KC下轨、RSI<超卖线、成交量>50周期均量*倍数、当前无多头仓位。
4. 空头开仓条件:允许做空、CCI>超买线、收盘价>KC上轨、RSI>超买线、成交量>50周期均量*倍数、当前无空头仓位。
5. 多头平仓条件:CCI>0。空头平仓条件:CCI<0。
6. 开仓时发出警报,平仓时也发出警报。

#### 策略优势
1. 结合多个指标进行综合判断,提高信号准确性。
2. 采用趋势过滤方法,可以根据市场趋势灵活调整,顺势而为。
3. 移动平均线类型可选,适应不同的市场特点。
4. 已经过长时间的历史数据验证,稳定性好,适合长期使用。
5. 双向交易,适应各种行情,获利机会多。
6. 自动化程度高,不需要人工干预,省时省力。

#### 策略风险
1. 缺乏传统的止损止盈,遇到极端行情可能出现较大回撤。
2. 在震荡市可能出现频繁开平仓,交易成本增加。 
3. 使用了相对较短的CCI周期,可能出现噪音信号。
4. 在趋势不明朗或市场波动加大时,趋势过滤作用有限。
5. 固定仓位,不能自适应市场波动率的变化。

#### 策略优化方向 
1. 可以考虑增加移动止损或固定点数止损,控制单笔交易风险。
2. RSI和CCI的参数可以进一步优化,以减少噪音信号。
3. 可以考虑引入波动率指标如ATR,根据市场波动调整仓位和止损。
4. 增加更多货币对,并根据每个品种特点单独优化参数。
5. 尝试引入机器学习等人工智能技术,自适应优化参数。

#### 总结
该策略采用了多个经典指标,trading view上编写和回测都比较方便。回测效果不错,但实盘中还需注意控制风险,调整参数。建议先小资金测试,积累经验后再逐步加大投入。机械化程度高,适合稳健型投资者长期使用。

|| 

#### Overview
This strategy combines three technical indicators: CCI, RSI, and Keltner Channels (KC), along with a trend filter to achieve bi-directional trading on AUDNZD and GBPNZD currency pairs. It uses CCI and RSI to determine overbought and oversold conditions, KC as a reference for stop-loss and take-profit, and a moving average as a trend filter to open positions in line with the trend. The strategy has been backtested on historical data over the past 5 years, achieving stable returns.

#### Strategy Principles
1. Calculate CCI, RSI, and KC indicators. The upper KC line is the midline plus ATR, and the lower line is the midline minus ATR.
2. Select the moving average type (SMA, EMA, SMMA, CMA, or TMA) and trend filter method (OFF, Normal, or Reversed) based on input parameters.
3. Long entry conditions: allow long, CCI < oversold line, close < KC lower line, RSI < oversold line, volume > 50-period average volume * multiplier, no current long position.
4. Short entry conditions: allow short, CCI > overbought line, close > KC upper line, RSI > overbought line, volume > 50-period average volume * multiplier, no current short position.
5. Long exit condition: CCI > 0. Short exit condition: CCI < 0.
6. Send alerts when opening and closing positions.

#### Strategy Advantages
1. Combines multiple indicators for comprehensive analysis, improving signal accuracy.
2. Uses trend filter methods, allowing flexible adjustments based on market trends.
3. Offers multiple moving average types, adapting to different market characteristics.
4. Validated through long-term historical data, demonstrating good stability and suitability for long-term use.
5. Bi-directional trading, suitable for various market conditions, providing more profit opportunities.
6. Highly automated, requiring no manual intervention, saving time and effort.

#### Strategy Risks
1. Lacks traditional stop-loss and take-profit, potentially leading to significant drawdowns in extreme market conditions.
2. May experience frequent opening and closing of positions in choppy markets, increasing trading costs.
3. Uses relatively short CCI periods, potentially generating noise signals.
4. Trend filters may have limited effectiveness when trends are unclear or market volatility increases.
5. Fixed position sizing, unable to adapt to changes in market volatility.

#### Strategy Optimization Directions
1. Consider adding trailing stops or fixed-point stop-losses to control single-trade risk.
2. Further optimize RSI and CCI parameters to reduce noise signals.
3. Consider introducing volatility indicators like ATR to adjust position sizing and stop-losses based on market volatility.
4. Add more currency pairs and optimize parameters individually based on each instrument's characteristics.
5. Attempt to introduce machine learning and other AI technologies for adaptive parameter optimization.

#### Summary
This strategy employs multiple classic indicators and is relatively easy to code and backtest on TradingView. While the backtesting results are good, risk control and parameter adjustments are still necessary for live trading. It is recommended to start with small funds for testing and gradually increase investment as experience accumulates. With a high degree of automation, it is suitable for conservative investors to use over the long term.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-04-01 00:00:00
end: 2024-04-30 23:59:59
period: 2h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy('CCI Strategy with Trend Filter AUDNZD, GBPNZD', overlay=true, default_qty_type=strategy.cash, default_qty_value=50000, commission_value=0.0005, slippage=2, initial_capital=10000)

// State variables to ensure one entry per signal
var bool isLongOpen = false
var bool isShortOpen = false

// Input Parameters for allowing long and short trades
allowLong = input(true, title='Allow Long Trades')
allowShort = input(true, title='Allow Short Trades')

// Trend Filter Inputs
maType = input.string(title='MA Type', options=['OFF', 'SMA', 'EMA', 'SMMA', 'CMA', 'TMA'], defval='OFF')
trendFilterMethod = input.string(title='Trend Filter Method', options=['OFF', 'Normal', 'Reversed'], defval='OFF')
maLength = input(14, title='MA Length')

// Other Input Parameters
lengthKC = input(30, title='Keltner Channels Length')
multKC = input(0.7, title='Keltner Channels Multiplier')
lengthCCI = input(5, title='CCI Length')
overboughtCCI = input(75, title='CCI Overbought Level')
oversoldCCI = input(-75, title='CCI Oversold Level')
rsiPeriod = input(30, title='RSI Period')
rsiOverbought = input(60, title='RSI Overbought Level')
rsiOversold = input(60, title='RSI Oversold Level')
volumeMultiplier = input.float(0, title='Volume Multiplier', step=0.1, minval=0)

// Define Moving Averages
var float maValue = na
if maType == 'SMA'
    maValue := ta.sma(close, maLength)
else if maType == 'EMA'
    maValue := ta.ema(close, maLength)
else if maType == 'SMMA'
    float initialSMMA = ta.sma(close, maLength)
    maValue := na(maValue[1]) ? initialSMMA : (maValue[1] * (maLength - 1) + close) / maLength
else if maType == 'CMA'
    float firstSMA = ta.sma(close, maLength)
    float secondSMA = ta.sma(close, maLength)
    maValue := na(maValue[1]) ? firstSMA : (firstSMA + secondSMA - maValue[1]) / 2
else if maType == 'TMA'
    maValue := ta.sma(ta.sma(close, math.round(maLength / 2)), math.round(maLength / 2) + 1)

// Entry Conditions with Trend Filter
longCondition = allowLong and (trendFilterMethod == 'OFF' or trendFilterMethod == 'Normal' and close > maValue or trendFilterMethod == 'Reversed' and close < maValue)
shortCondition = allowShort and (trendFilterMethod == 'OFF' or trendFilterMethod == 'Normal' and close < maValue or trendFilterMethod == 'Reversed' and close > maValue)

// Keltner Channels
typicalPrice = hlc3
middleLine = ta.sma(typicalPrice, lengthKC)
range_1 = multKC * ta.atr(lengthKC)
upperChannel = middleLine + range_1
lowerChannel = middleLine - range_1

// CCI
cci = ta.cci(close, lengthCCI)

// RSI
rsi = ta.rsi(close, rsiPeriod)

// Volume
volCondition = volume > ta.sma(volume, 50) * volumeMultiplier

// Combined Entry Conditions with Trend Filter and state check
longCondition := longCondition and cci < oversoldCCI and low < lowerChannel and rsi < rsiOversold and volCondition and not isLongOpen
shortCondition := shortCondition and cci > overboughtCCI and high > upperChannel and rsi > rsiOverbought and volCondition and not isShortOpen

// Execute orders at the open of the new bar after conditions are met
if longCondition
    strategy.entry('Long', strategy.long)
    alert('LicenseID,buy,AUDNZD,risk=1')
    isLongOpen := true
if shortCondition
    strategy.entry('Short', strategy.short)
    alert('LicenseID,sell,AUDNZD,risk=1')
    isShortOpen := true

// Exit Conditions and Alerts
longExitCondition = cci > 0
shortExitCondition = cci < 0
if (longExitCondition and isLongOpen)
    strategy.close('Long')
    alert('LiceneseID,closelong,AUDNZD')
    isLongOpen := false
if (shortExitCondition and isShortOpen)
    strategy.close('Short')
    alert('LicenseID,closeshort,AUDNZD')
    isShortOpen := false

// Plotting
plot(upperChannel, color=color.new(color.red, 0), linewidth=1)
plot(lowerChannel, color=color.new(color.green, 0), linewidth=1)
hline(overboughtCCI, 'Overbought', color=color.red)
hline(oversoldCCI, 'Oversold', color=color.green)

```

> Detail

https://www.fmz.com/strategy/451530

> Last Modified

2024-05-15 16:56:03
