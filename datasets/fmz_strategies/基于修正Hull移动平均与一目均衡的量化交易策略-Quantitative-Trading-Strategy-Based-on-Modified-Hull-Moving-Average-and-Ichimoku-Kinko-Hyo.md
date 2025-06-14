
> Name

基于修正Hull移动平均与一目均衡的量化交易策略-Quantitative-Trading-Strategy-Based-on-Modified-Hull-Moving-Average-and-Ichimoku-Kinko-Hyo

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/d81d0903d5d9151893.png)

[trans]
#### 概述
该策略结合了修正的Hull移动平均(HMA)和一目均衡(Ichimoku Kinko Hyo)两种技术指标,旨在捕捉市场的中长期趋势。策略的主要思路是利用HMA与一目均衡中的基准线(Kijun Sen)的交叉信号,同时结合一目均衡的云层(Kumo)作为过滤条件,以此来判断市场的趋势方向并进行交易。

#### 策略原理
1. 计算修正的Hull移动平均(HMA)
   - 计算WMA(加权移动平均)并进行双重平滑处理,得到修正的HMA
2. 计算一目均衡的各项指标
   - 计算转折线(Tenkan Sen)、基准线(Kijun Sen)、先行上线(Senkou Span A)和先行下线(Senkou Span B)
3. 生成交易信号
   - 当HMA上穿基准线,且收盘价位于云层之上时,产生做多信号
   - 当HMA下穿基准线,且收盘价位于云层之下时,产生做空信号
4. 执行交易
   - 根据做多或做空信号进行相应的交易操作
5. 退出交易
   - 当HMA在相反方向上穿越基准线时,退出当前持仓

#### 策略优势
1. 结合了HMA和一目均衡两种有效的趋势跟踪指标,能够更好地捕捉市场趋势
2. 利用一目均衡的云层作为过滤条件,可以有效减少虚假信号,提高交易的胜率
3. 修正的HMA相比传统的移动平均线具有更快的响应速度和更低的延迟,能够及时反映市场的变化
4. 策略逻辑清晰,易于理解和实现,适用于各种市场和时间周期

#### 策略风险
1. 在市场震荡或趋势不明朗时,该策略可能会产生较多的虚假信号,导致频繁交易和资金损失
2. 策略的参数设置对交易结果有较大影响,不同的参数组合可能导致不同的表现
3. 该策略未考虑市场的突发事件和非理性行为,在极端市场条件下可能面临较大风险

#### 策略优化方向
1. 引入其他技术指标或市场情绪指标,以提高信号的可靠性和稳定性
2. 优化策略参数,如通过机器学习或遗传算法等方法寻找最优参数组合
3. 考虑加入风险管理模块,如设置止损止盈、仓位管理等,以控制策略的风险敞口
4. 根据不同市场和时间周期的特点,对策略进行针对性的调整和优化

#### 总结
该策略通过结合修正的Hull移动平均和一目均衡,构建了一个相对稳健的趋势跟踪交易系统。策略逻辑清晰,易于实现,同时也具有一定的优势。然而,策略的表现仍然受到市场条件和参数设置的影响,需要进一步的优化和改进。在实际应用中,应结合具体的市场特点和风险偏好,对策略进行适当的调整和管理,以期获得更好的交易结果。

|| 

#### Overview
This strategy combines two technical indicators: the modified Hull Moving Average (HMA) and Ichimoku Kinko Hyo (IKHS), aiming to capture medium to long-term market trends. The main idea is to utilize the crossover signals between the HMA and the Kijun Sen (baseline) of IKHS, while using the Kumo (cloud) of IKHS as a filtering condition to determine the trend direction and make trading decisions.

#### Strategy Principles
1. Calculate the modified Hull Moving Average (HMA)
   - Calculate the Weighted Moving Average (WMA) and apply double smoothing to obtain the modified HMA
2. Calculate the various indicators of Ichimoku Kinko Hyo
   - Calculate the Tenkan Sen (conversion line), Kijun Sen (baseline), Senkou Span A (leading span A), and Senkou Span B (leading span B)
3. Generate trading signals
   - When the HMA crosses above the Kijun Sen and the closing price is above the Kumo, generate a long signal
   - When the HMA crosses below the Kijun Sen and the closing price is below the Kumo, generate a short signal
4. Execute trades
   - Perform corresponding trading operations based on the long or short signals
5. Exit trades
   - When the HMA crosses the Kijun Sen in the opposite direction, exit the current position

#### Strategy Advantages
1. Combines two effective trend-following indicators, HMA and IKHS, to better capture market trends
2. Utilizes the Kumo of IKHS as a filtering condition to effectively reduce false signals and improve the win rate of trades
3. The modified HMA has a faster response speed and lower lag compared to traditional moving averages, enabling timely reflection of market changes
4. The strategy logic is clear, easy to understand, and implement, suitable for various markets and time frames

#### Strategy Risks
1. During market fluctuations or unclear trends, the strategy may generate more false signals, leading to frequent trading and capital losses
2. The parameter settings of the strategy have a significant impact on trading results, and different parameter combinations may lead to different performances
3. The strategy does not consider market emergencies and irrational behaviors, and may face greater risks under extreme market conditions

#### Strategy Optimization Directions
1. Introduce other technical indicators or market sentiment indicators to improve the reliability and stability of signals
2. Optimize strategy parameters, such as using machine learning or genetic algorithms to find the optimal parameter combination
3. Consider adding a risk management module, such as setting stop-loss and take-profit levels, position sizing, etc., to control the risk exposure of the strategy
4. Based on the characteristics of different markets and time frames, make targeted adjustments and optimizations to the strategy

#### Summary
By combining the modified Hull Moving Average and Ichimoku Kinko Hyo, this strategy constructs a relatively stable trend-following trading system. The strategy logic is clear and easy to implement, while also having certain advantages. However, the performance of the strategy is still affected by market conditions and parameter settings, requiring further optimization and improvement. In practical applications, it is necessary to make appropriate adjustments and management based on specific market characteristics and risk preferences to obtain better trading results.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_1|12|Double HullMA|
|v_input_2|9|Tenkan Sen Periods|
|v_input_3|26|Kijun Sen Periods|
|v_input_4|52|Senkou Span B Periods|
|v_input_5|26|Displacement|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-04-20 00:00:00
end: 2024-04-27 00:00:00
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=4
strategy("Hull MA_X + Ichimoku Kinko Hyo Strategy", shorttitle="HMX+IKHS", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=100, pyramiding=0)

// Hull Moving Average Parameters
keh = input(12, title="Double HullMA")
n2ma = 2 * wma(close, round(keh/2)) - wma(close, keh)
sqn = round(sqrt(keh))
hullMA = wma(n2ma, sqn)

// Ichimoku Kinko Hyo Parameters
tenkanSenPeriods = input(9, title="Tenkan Sen Periods")
kijunSenPeriods = input(26, title="Kijun Sen Periods")
senkouSpanBPeriods = input(52, title="Senkou Span B Periods")
displacement = input(26, title="Displacement")

// Ichimoku Calculations
highestHigh = highest(high, max(tenkanSenPeriods, kijunSenPeriods))
lowestLow = lowest(low, max(tenkanSenPeriods, kijunSenPeriods))
tenkanSen = (highest(high, tenkanSenPeriods) + lowest(low, tenkanSenPeriods)) / 2
kijunSen = (highestHigh + lowestLow) / 2
senkouSpanA = ((tenkanSen + kijunSen) / 2)
senkouSpanB = (highest(high, senkouSpanBPeriods) + lowest(low, senkouSpanBPeriods)) / 2

// Plot Ichimoku
p1 = plot(tenkanSen, color=color.blue, title="Tenkan Sen")
p2 = plot(kijunSen, color=color.red, title="Kijun Sen")
p3 = plot(senkouSpanA, color=color.green, title="Senkou Span A", offset=displacement)
p4 = plot(senkouSpanB, color=color.orange, title="Senkou Span B", offset=displacement)
fill(p3, p4, color=color.gray, title="Kumo Shadow")

// Trading Logic
longCondition = crossover(hullMA, kijunSen) and close > senkouSpanA[displacement] and close > senkouSpanB[displacement]
shortCondition = crossunder(hullMA, kijunSen) and close < senkouSpanA[displacement] and close < senkouSpanB[displacement]

// Strategy Execution
if (longCondition)
    strategy.entry("Long", strategy.long)
if (shortCondition)
    strategy.entry("Short", strategy.short)

// Exit Logic - Exit if HullMA crosses KijunSen in the opposite direction
exitLongCondition = crossunder(hullMA, kijunSen)
exitShortCondition = crossover(hullMA, kijunSen)

if (exitLongCondition)
    strategy.close("Long")
if (exitShortCondition)
    strategy.close("Short")

```

> Detail

https://www.fmz.com/strategy/449710

> Last Modified

2024-04-28 13:39:00
