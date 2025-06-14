
> Name

HalfTrend多空趋势追踪止损限价买入策略-HalfTrend-Bullish-and-Bearish-Trend-Following-Stop-Limit-Buy-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/af540cf5634aa10733.png)

[trans]
####概述
该策略基于HalfTrend指标,通过判断多空趋势来识别买入信号。当HalfTrend指标由空转多时,在之前空头趋势的HalfTrend值位置下单止损限价买入。该策略利用AmplitudeTrend指标(ATR)来动态调整趋势判断的幅度参数。

####策略原理
1. 计算HalfTrend指标值,需要设置回看周期length和幅度参数amplitude。
2. 比较当前收盘价与上一周期HalfTrend指标值,判断多空趋势。
   - 当收盘价上穿HalfTrend指标值amplitude点时,趋势转多。
   - 当收盘价下穿HalfTrend指标值amplitude点时,趋势转空。
3. 记录趋势转空时的HalfTrend指标值,作为未来潜在买入位置。
4. 当HalfTrend指标再次由空转多时,在步骤3记录的位置下达止损限价买入单。

####策略优势
1. 基于完整的多空趋势来判断投资方向,最大程度地适应当前行情。
2. 使用limit order下单,可以在预设位置买入以获得更优的成交价。
3. 买入位置基于之前空头HalfTrend趋势确定,保证了买入点的低位安全性。
4. 使用amplitude参数来控制区分多空趋势所需要的最小幅度,可以有效过滤噪音信号。

####策略风险
1. 趋势转向判断依赖amplitude参数,不当的参数值可能导致过早或过晚下单。
2. limit order可能因为价格波动而无法成交,错失上涨行情。
3. 止损设置位置过于靠近买入位置,可能承担较大亏损。

####策略优化方向
1. 对amplitude参数进行优化,寻找最佳的趋势判断幅度。可以使用AmplitudeTrend指标(ATR)动态调整幅度。
2. 在止损买入同时设置take profit卖出,及时锁定利润。
3. 止损位置可以设置得更低一些,给予更大的亏损空间,同时也提高了获利空间。
4. 可以加入移动止损逻辑,在价格向有利方向移动时提升止损位,减小风险。

####总结
HalfTrend多空趋势追踪止损限价买入策略通过判断HalfTrend指标的多空趋势变化来决定买入时机,利用之前空头趋势的低点作为买入位置,以求在相对安全的低位入场做多。该策略包含趋势判断、限价单、止损单等常用策略要素,可以进一步优化以提高风险收益比。

|| 

####Overview
This strategy is based on the HalfTrend indicator and identifies buy signals by determining bullish and bearish trends. When the HalfTrend indicator switches from bearish to bullish, a stop-limit buy order is placed at the HalfTrend value of the previous bearish trend. The strategy uses the AmplitudeTrend (ATR) indicator to dynamically adjust the amplitude parameter for trend determination.

####Strategy Principle
1. Calculate the HalfTrend indicator value, which requires setting the lookback period length and amplitude parameter.
2. Compare the current closing price with the previous period's HalfTrend indicator value to determine the bullish or bearish trend.
   - When the closing price crosses above the HalfTrend indicator value by amplitude points, the trend turns bullish.
   - When the closing price crosses below the HalfTrend indicator value by amplitude points, the trend turns bearish.
3. Record the HalfTrend indicator value when the trend turns bearish, which serves as a potential future buy position.
4. When the HalfTrend indicator switches from bearish to bullish again, place a stop-limit buy order at the position recorded in step 3.

####Strategy Advantages
1. Based on complete bullish and bearish trends to determine the investment direction, maximally adapting to the current market condition.
2. Using limit orders for buying, which can achieve better execution prices at predetermined positions.
3. The buy position is determined based on the previous bearish HalfTrend trend, ensuring the safety of buying at a low level.
4. The amplitude parameter is used to control the minimum amplitude required to distinguish between bullish and bearish trends, effectively filtering out noise signals.

####Strategy Risks
1. The trend reversal determination relies on the amplitude parameter, and inappropriate parameter values may lead to premature or delayed order placement.
2. Limit orders may fail to execute due to price fluctuations, missing out on upward movements.
3. The stop-loss setting position may be too close to the buy position, potentially incurring significant losses.

####Strategy Optimization Directions
1. Optimize the amplitude parameter to find the best amplitude for trend determination. The AmplitudeTrend (ATR) indicator can be used to dynamically adjust the amplitude.
2. Set a take profit sell order along with the stop-loss buy order to lock in profits in a timely manner.
3. The stop-loss position can be set lower to allow for a larger loss margin while also increasing the profit potential.
4. Incorporate a trailing stop-loss logic to raise the stop-loss position when the price moves in a favorable direction, reducing risk.

####Summary
The HalfTrend Bullish and Bearish Trend Following Stop-Limit Buy Strategy determines the timing of buying based on changes in the bullish and bearish trends of the HalfTrend indicator. It uses the low point of the previous bearish trend as the buy position, aiming to enter long positions at relatively safe low levels. This strategy incorporates common strategy elements such as trend determination, limit orders, and stop-loss orders, and it can be further optimized to improve the risk-reward ratio.

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
strategy("HalfTrend Stop-Limit Buy", overlay=true)

// HalfTrend indicator parameters
length = 1
amplitude = 2.0

// HalfTrend calculation
float ph = na
float pl = na
var float dir = na
var float trend = na

if na(trend)
    trend := close
    ph := high
    pl := low
    dir := na
else
    if high > ph
        ph := high
    if low < pl
        pl := low
    if close > trend and na(dir)
        dir := 1
        trend := close
        ph := high
        pl := low
    if close < trend and na(dir)
        dir := -1
        trend := close
        ph := high
        pl := low
    if dir == 1 and close < trend - amplitude
        dir := -1
        trend := close
        ph := high
        pl := low
    if dir == -1 and close > trend + amplitude
        dir := 1
        trend := close
        ph := high
        pl := low

// Buy signal based on HalfTrend
buySignal = dir == 1 and ta.valuewhen(dir == -1, trend, 0)

// Plot HalfTrend
plot(dir == 1 ? trend : na, color=color.blue, linewidth=2, title="HalfTrend Bullish")
plot(dir == -1 ? trend : na, color=color.red, linewidth=2, title="HalfTrend Bearish")

// Place a stop-limit buy order
if (buySignal)
    stopPrice = ta.valuewhen(dir == -1, trend, 0)
    strategy.entry("HalfTrend Buy", strategy.long, stop=stopPrice, comment="HalfTrend Buy")


```

> Detail

https://www.fmz.com/strategy/451732

> Last Modified

2024-05-17 15:45:13
