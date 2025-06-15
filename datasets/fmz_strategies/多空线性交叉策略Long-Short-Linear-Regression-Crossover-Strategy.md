
> Name

多空线性交叉策略Long-Short-Linear-Regression-Crossover-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/e5055b9dfe6da41ab8.png)
[trans]
## 概述

多空线性交叉策略是一种技术分析策略,基于线性回归模型预测股票未来价格走势。策略的基本原理是:股价走势往往遵循一定的线性趋势,通过计算价格的线性回归,可以预测未来价格。当预测价格上穿当前价格时做多,下穿时平仓。

## 策略原理

该策略首先计算一段时间内股价的线性回归。线性回归用最小二乘法拟合出一条直线,这条直线代表了价格随时间变化的趋势。策略然后在图表上绘制预测价格线和当前价格。

策略定义了两个信号:

1. 做多信号:当预测价格上穿当前价格时触发  
2. 做空信号:当预测价格下穿当前价格时触发

当做多信号出现时,策略开仓做多;当做空信号出现时,平仓。

策略的关键步骤如下:

1. 计算一段时间内价格的线性回归
2. 在图表上画出预测价格线和当前价格  
3. 定义做多和做空信号
4. 做多信号触发时开仓做多
5. 做空信号触发时平仓

## 优势分析

多空线性交叉策略有以下优点:

1. 简单有效:该策略逻辑清晰,易于实现,可以捕捉到价格的线性趋势。
2. 适用性广:无论在趋势行情还是震荡行情,策略都能产生交易信号。  
3. 可优化性强:策略包含一些关键参数,如线性回归周期、移动平均线等,可以通过优化这些参数提高策略性能。

## 风险分析  

尽管多空线性交叉策略有诸多优点,但它也存在一些风险:

1. 趋势识别风险:当价格走势不遵循线性趋势时,例如震荡行情,策略可能会产生错误信号。可以通过结合其他指标如MACD等来降低风险。
2. 参数设置风险:策略性能对参数设置较为敏感,不当的参数可能导致亏损。因此实盘前需要对参数做充分的回测和优化。  
3. 过拟合风险:如果对参数优化过度,可能导致策略过拟合历史数据,未来表现不佳。避免过拟合的方法包括保持简单、数据集外测试等。

## 优化方向

1. 结合其他指标:线性回归信号可以与其他技术指标如MACD、布林带等结合,提高信号准确度。
2. 动态参数优化:可以设计一套参数自适应机制,根据市场情况动态调整参数,提高适应性。
3. 加入风控模块:在策略中加入止损、资金管理等风控措施,降低单次交易风险,提高累积收益。  
4. 机器学习优化:可以用机器学习算法不断优化线性回归模型,使其预测更加准确。

## 总结

多空线性交叉策略以价格线性回归为基础,通过比较预测价格和当前价格产生交易信号。该策略逻辑简单清晰,可以捕捉价格的线性趋势,适用于各类行情。同时,策略易于实现和优化,可以灵活调整参数,结合其他指标,加入风控模块等,不断提高策略性能。但策略也存在识别趋势不准、参数设置不当、过拟合历史数据等风险,实际运用时需谨慎。总的来说,多空线性交叉策略是一个简单有效的量化交易策略,值得进一步探索和优化。

|| 

## Overview

The Long-Short Linear Regression Crossover Strategy is a technical analysis strategy that uses a linear regression model to predict the future price movements of a stock. The basic principle of the strategy is: stock price movements often follow a certain linear trend, and by calculating the linear regression of the price, the future price can be predicted. The strategy goes long when the predicted price crosses above the current price, and exits the position when it crosses below.

## Strategy Principles

The strategy first calculates the linear regression of the stock price over a certain period of time. Linear regression fits a straight line using the least squares method, which represents the trend of price changing over time. The strategy then plots the predicted price line and the current price on the chart.

The strategy defines two signals:

1. Long signal: triggered when the predicted price crosses above the current price
2. Short signal: triggered when the predicted price crosses below the current price

When the long signal appears, the strategy opens a long position; when the short signal appears, it closes the position.

The key steps of the strategy are as follows:

1. Calculate the linear regression of the price over a period of time
2. Plot the predicted price line and the current price on the chart
3. Define the long and short signals 
4. Open a long position when the long signal is triggered
5. Close the position when the short signal is triggered

## Advantages Analysis

The Long-Short Linear Regression Crossover Strategy has the following advantages:

1. Simple and effective: The logic of the strategy is clear and easy to implement, and it can capture the linear trend of the price.
2. Wide applicability: The strategy can generate trading signals in both trending and ranging markets.
3. Strong optimizability: The strategy contains some key parameters, such as linear regression period, moving averages, etc., which can be optimized to improve the performance.

## Risk Analysis

Despite its many advantages, the Long-Short Linear Regression Crossover Strategy also has some risks:

1. Trend recognition risk: When the price movement does not follow a linear trend, such as in a ranging market, the strategy may generate false signals. This risk can be reduced by combining with other indicators such as MACD.
2. Parameter setting risk: The performance of the strategy is sensitive to parameter settings, and inappropriate parameters may lead to losses. Therefore, sufficient backtesting and optimization of parameters are necessary before live trading.
3. Overfitting risk: If the parameters are over-optimized, it may cause the strategy to overfit the historical data and perform poorly in the future. Methods to avoid overfitting include keeping it simple, out-of-sample testing, etc.

## Optimization Directions

1. Combine with other indicators: The linear regression signal can be combined with other technical indicators such as MACD, Bollinger Bands, etc., to improve the accuracy of the signals.
2. Dynamic parameter optimization: An adaptive mechanism for parameters can be designed to dynamically adjust parameters according to market conditions, improving adaptability.
3. Add a risk control module: Incorporate risk control measures such as stop-loss and money management into the strategy to reduce the risk of a single transaction and increase cumulative returns.
4. Machine learning optimization: Machine learning algorithms can be used to continuously optimize the linear regression model to make its predictions more accurate.

## Summary

The Long-Short Linear Regression Crossover Strategy generates trading signals based on the comparison of the predicted price from linear regression and the current price. The logic of the strategy is simple and clear, and it can capture the linear trend of the price and is applicable to various market conditions. At the same time, the strategy is easy to implement and optimize, and parameters can be flexibly adjusted, combined with other indicators, risk control modules can be added, etc., to continuously improve the performance of the strategy. However, the strategy also has risks such as inaccurate trend recognition, inappropriate parameter settings, and overfitting of historical data, so caution is needed in practical application. Overall, the Long-Short Linear Regression Crossover Strategy is a simple and effective quantitative trading strategy that is worth further exploration and optimization.

[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_1|21|Strategy Length|
|v_input_2|9|Linear Lookback|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-02-25 00:00:00
end: 2024-03-26 00:00:00
period: 3h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © stocktechbot
//@version=5
strategy("Linear Cross", overlay=true, margin_long=100, margin_short=0)

//Linear Regression

vol = volume

// Function to calculate linear regression
linregs(y, x, len) =>
    ybar = math.sum(y, len)/len
    xbar = math.sum(x, len)/len
    b = math.sum((x - xbar)*(y - ybar),len)/math.sum((x - xbar)*(x - xbar),len)
    a = ybar - b*xbar
    [a, b]

// Historical stock price data
price = close

// Length of linear regression
len = input(defval = 21, title = 'Strategy Length')
linearlen=input(defval = 9, title = 'Linear Lookback')
[a, b] = linregs(price, vol, len)

// Calculate linear regression for stock price based on volume
//eps = request.earnings(syminfo.ticker, earnings.actual)
//MA For double confirmation

out = ta.sma(close, 200)
outf = ta.sma(close, 50)
outn = ta.sma(close, 90)
outt = ta.sma(close, 21)
outthree = ta.sma(close, 9)

// Predicted stock price based on volume
predicted_price = a + b*vol

// Check if predicted price is between open and close
is_between = open < predicted_price and predicted_price < close

//MACD
//[macdLine, signalLine, histLine] = ta.macd(close, 12, 26, 9)

// Plot predicted stock price
plot(predicted_price, color=color.rgb(65, 59, 150), linewidth=2, title="Predicted Price")
plot(ta.sma(predicted_price,linearlen), color=color.rgb(199, 43, 64), linewidth=2, title="MA Predicted Price")
//offset = input.int(title="Offset", defval=0, minval=-500, maxval=500)
plot(out, color=color.blue, title="MA200")
[macdLine, signalLine, histLine] = ta.macd(predicted_price, 12, 26, 9)

//BUY Signal

longCondition=false
mafentry =ta.sma(close, 50) > ta.sma(close, 90)
//matentry = ta.sma(close, 21) > ta.sma(close, 50)
matwohun = close > ta.sma(close, 200)
twohunraise = ta.rising(out, 2)
twentyrise = ta.rising(outt, 2)
macdrise = ta.rising(macdLine,2)
macdlong = ta.crossover(predicted_price, ta.wma(predicted_price,linearlen))  and (signalLine < macdLine)
if macdlong and macdrise
    longCondition := true

if (longCondition)
    strategy.entry("My Long Entry Id", strategy.long)
//Sell Signal
lastEntryPrice = strategy.opentrades.entry_price(strategy.opentrades - 1)
daysSinceEntry = len
daysSinceEntry := int((time - strategy.opentrades.entry_time(strategy.opentrades - 1)) / (24 * 60 * 60 * 1000))
percentageChange = (close - lastEntryPrice) / lastEntryPrice * 100
//trailChange = (ta.highest(close,daysSinceEntry) - close) / close * 100

//label.new(bar_index, high, color=color.black, textcolor=color.white,text=str.tostring(int(trailChange)))
shortCondition=false
mafexit =ta.sma(close, 50) < ta.sma(close, 90)
matexit = ta.sma(close, 21) < ta.sma(close, 50)
matwohund = close < ta.sma(close, 200)
twohunfall = ta.falling(out, 3)
twentyfall = ta.falling(outt, 2)
shortmafall = ta.falling(outthree, 1)
macdfall = ta.falling(macdLine,1)
macdsell = macdLine < signalLine
if macdfall and macdsell and (macdLine < signalLine) and ta.falling(low,2)
    shortCondition := true

if (shortCondition)
    strategy.entry("My Short Entry Id", strategy.short)



```

> Detail

https://www.fmz.com/strategy/446331

> Last Modified

2024-03-27 17:52:02
