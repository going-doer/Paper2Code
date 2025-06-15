
> Name

基于WaveTrend指标的趋势追踪策略WaveTrend-Cross-LazyBear-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/15f45942ea170c9d1f2.png)
[trans]

#### 概述

WaveTrend Cross LazyBear策略是一个基于WaveTrend指标的交易策略。该策略使用两条不同周期的WaveTrend指标线,当较快周期的WaveTrend指标线上穿较慢周期的WaveTrend指标线时产生买入信号,当较快周期的WaveTrend指标线下穿较慢周期的WaveTrend指标线时产生卖出信号。该策略还设置了超买和超卖区域,用于辅助判断市场状态。

#### 策略原理

该策略的核心是WaveTrend指标,该指标由以下步骤计算得出:

1. 计算典型价格(AP)，其等于最高价、最低价和收盘价的均值。 
2. 计算AP的指数移动平均线(ESA),周期为n1。
3. 计算AP与ESA的差值的绝对值的指数移动平均值d,周期为n1。
4. 计算指标CI,其等于(AP - ESA) / (0.015 * d)。
5. 计算CI的指数移动平均线TCI,周期为n2,得到WaveTrend指标。

该策略使用两条不同周期(默认为10和21)的WaveTrend指标线,分别记为WT1和WT2。当WT1上穿WT2时,产生买入信号;当WT1下穿WT2时,产生卖出信号。此外,该策略还设置了4个辅助判断的水平:超买水平1、超买水平2、超卖水平1和超卖水平2,用于辅助判断市场状态。

#### 策略优势

1. WaveTrend指标结合了动量和波动率的特点,能够较好地捕捉市场趋势。
2. 双重周期的WaveTrend指标能够有效过滤掉一些噪音信号。
3. 超买超卖水平的设置可以在一定程度上防止策略在市场波动较大时频繁交易。
4. 策略逻辑清晰,易于理解和实现。

#### 策略风险

1. 该策略在震荡市中可能会出现较多的假信号。
2. 参数的选择对策略表现有较大影响,不同的参数可能导致策略表现差异较大。
3. 策略未考虑风险控制,在极端行情下可能出现较大回撤。

#### 策略优化方向

1. 可以考虑加入趋势过滤条件,如长期均线的方向,以减少震荡市中的假信号。
2. 可以优化超买超卖水平的设置,使其能够更加动态地适应不同的市场状态。
3. 可以加入止损和止盈机制,控制单次交易的风险。
4. 可以通过参数优化来寻找最佳的参数组合。

#### 总结

WaveTrend Cross LazyBear策略是一个基于WaveTrend指标的趋势追踪策略,通过双重周期的指标设计和超买超卖水平的辅助判断,在捕捉趋势的同时也兼顾了一定的风险控制。但是,该策略在震荡市中可能出现较多假信号,且缺乏严格的风险管理措施,实际应用中还需要进一步的优化和改进。

|| 

#### Overview

The WaveTrend Cross LazyBear strategy is a trading strategy based on the WaveTrend indicator. The strategy uses two WaveTrend indicator lines with different periods. When the faster-period WaveTrend indicator line crosses above the slower-period WaveTrend indicator line, it generates a buy signal. When the faster-period WaveTrend indicator line crosses below the slower-period WaveTrend indicator line, it generates a sell signal. The strategy also sets overbought and oversold zones to assist in judging market conditions.

#### Strategy Principle

The core of this strategy is the WaveTrend indicator, which is calculated by the following steps:

1. Calculate the typical price (AP), which is equal to the average of the high, low, and close prices.
2. Calculate the exponential moving average (ESA) of AP with a period of n1.
3. Calculate the exponential moving average d of the absolute value of the difference between AP and ESA with a period of n1.
4. Calculate the indicator CI, which is equal to (AP - ESA) / (0.015 * d).
5. Calculate the exponential moving average TCI of CI with a period of n2 to get the WaveTrend indicator.

The strategy uses two WaveTrend indicator lines with different periods (default is 10 and 21), denoted as WT1 and WT2 respectively. When WT1 crosses above WT2, it generates a buy signal; when WT1 crosses below WT2, it generates a sell signal. In addition, the strategy also sets 4 auxiliary levels: overbought level 1, overbought level 2, oversold level 1, and oversold level 2, to assist in judging market conditions.

#### Strategy Advantages

1. The WaveTrend indicator combines the characteristics of momentum and volatility, which can better capture market trends.
2. The dual-period WaveTrend indicator can effectively filter out some noise signals.
3. The setting of overbought and oversold levels can prevent the strategy from trading frequently when the market fluctuates greatly to a certain extent.
4. The strategy logic is clear and easy to understand and implement.

#### Strategy Risks

1. The strategy may generate more false signals in a oscillating market.
2. The choice of parameters has a great impact on the strategy performance, and different parameters may lead to large differences in strategy performance.
3. The strategy does not consider risk control and may experience large drawdowns in extreme market conditions.

#### Strategy Optimization Directions

1. Consider adding trend filtering conditions, such as the direction of the long-term moving average, to reduce false signals in oscillating markets.
2. Optimize the setting of overbought and oversold levels to make them more dynamically adapt to different market conditions.
3. Add stop-loss and take-profit mechanisms to control the risk of a single transaction.
4. Find the optimal parameter combination through parameter optimization.

#### Summary

The WaveTrend Cross LazyBear strategy is a trend-tracking strategy based on the WaveTrend indicator. Through the design of dual-period indicators and the auxiliary judgment of overbought and oversold levels, it captures trends while also taking into account certain risk control. However, the strategy may generate more false signals in oscillating markets and lacks strict risk management measures. Further optimization and improvement are needed in practical applications.

[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_1|10|Channel Length|
|v_input_2|21|Average Length|
|v_input_3|60|Over Bought Level 1|
|v_input_4|53|Over Bought Level 2|
|v_input_5|-60|Over Sold Level 1|
|v_input_6|-53|Over Sold Level 2|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-03-01 00:00:00
end: 2024-03-31 23:59:59
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © burakaydingr

//@version=5
strategy("WaveTrend with Crosses [LazyBear]", shorttitle="WT_CROSS_LB", overlay=true)

// Kullanıcı girişleri
n1 = input(10, title="Channel Length")
n2 = input(21, title="Average Length")
obLevel1 = input(60, title="Over Bought Level 1")
obLevel2 = input(53, title="Over Bought Level 2")
osLevel1 = input(-60, title="Over Sold Level 1")
osLevel2 = input(-53, title="Over Sold Level 2")

// Temel hesaplamalar
ap = hlc3
esa = ta.ema(ap, n1)
d = ta.ema(math.abs(ap - esa), n1)
ci = (ap - esa) / (0.015 * d)
tci = ta.ema(ci, n2)

// WaveTrend göstergeleri
wt1 = tci
wt2 = ta.sma(wt1, 4)

// Al ve Sat Sinyalleri
buySignal = ta.crossover(wt1, wt2)
sellSignal = ta.crossunder(wt1, wt2)

// Alım ve Satım pozisyonları
if (buySignal)
    if (strategy.position_size <= 0) // Eğer şu anda açık bir satış pozisyonu varsa, onu kapat
        strategy.close("Sell")
    strategy.entry("Buy", strategy.long, comment="Buy Signal: Price crossed above WT2")

if (sellSignal)
    if (strategy.position_size >= 0) // Eğer şu anda açık bir alım pozisyonu varsa, onu kapat
        strategy.close("Buy")
    strategy.entry("Sell", strategy.short, comment="Sell Signal: Price crossed below WT2")

// Renkler ve diğer görseller
plot(0, color=color.new(color.gray, 0), title="Zero Level")
plot(obLevel1, color=color.new(color.red, 0), title="Overbought Level 1")
plot(osLevel1, color=color.new(color.green, 0), title="Oversold Level 1")
plot(obLevel2, color=color.new(color.purple, 0), title="Overbought Level 2")
plot(osLevel2, color=color.new(color.orange, 0), title="Oversold Level 2")

plot(wt1, color=color.new(color.red, 0), title="WT1")
plot(wt2, color=color.new(color.blue, 0), title="WT2")
plot(wt1-wt2, color=color.new(color.purple, 80), style=plot.style_area, title="WT1-WT2 Area")

// İşaretler
plotshape(buySignal, location=location.absolute, color=color.new(color.yellow, 0), style=shape.circle, size=size.small, title="Buy Signal")
plotshape(sellSignal, location=location.absolute, color=color.new(color.red, 0), style=shape.circle, size=size.small, title="Sell Signal")

```

> Detail

https://www.fmz.com/strategy/449714

> Last Modified

2024-04-28 13:56:27
