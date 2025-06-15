
> Name

基于相对强弱指数RSI和滑动平均SMA的波动率标准差DEV交易策略-Standard-Deviation-DEV-Trading-Strategy-Based-on-Relative-Strength-Index-RSI-and-Simple-Moving-Average-SMA

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1001c0360d7a17534bc.png)

[trans]
#### 概述
这个Pine Script策略基于相对强弱指数RSI和价格的波动率标准差DEV,通过比较价格与上下轨道来判断进场点,同时使用RSI作为辅助过滤指标,在价格触及上下轨道且RSI达到超买超卖区间时产生开仓信号,当价格反向突破退出轨道或RSI反向达到超买超卖区间时平仓。该策略能够根据市场波动情况动态调整,在波动率较高时及时止损,波动率较低时持仓获利,是一个能够适应不同市场状态的量化交易策略。

#### 策略原理
1. 计算价格在过去length个周期内的滑动平均SMA和标准差DEV。 
2. 以SMA为中轴,SMA+thresholdEntry*DEV为上轨,SMA-thresholdEntry*DEV为下轨,构建波动率通道。
3. 同时计算过去rsiLength个周期收盘价的RSI指标。
4. 当价格向上突破下轨且RSI小于超卖阈值rsiOversold时,产生开多仓信号。
5. 当价格向下突破上轨且RSI大于超买阈值rsiOverbought时,产生开空仓信号。
6. 以SMA为中轴,SMA+thresholdExit*DEV为上轨,SMA-thresholdExit*DEV为下轨,构建另一个较窄的退出通道。
7. 当持多仓时,如果价格向下突破退出下轨或RSI大于超买阈值,平多仓。
8. 当持空仓时,如果价格向上突破退出上轨或RSI小于超卖阈值,平空仓。

#### 优势分析
1. 同时使用价格行为和动量指标辅助判断,可以有效过滤假信号。
2. 通过波动率动态调整通道宽度,使策略能够适应不同的市场状态。
3. 设置两套通道,在价格反转初期即可止损,控制回撤,同时在趋势形成后仍能持仓获利。
4. 代码逻辑和参数设置清晰明了,容易理解和优化。

#### 风险分析
1. 当市场持续单边趋势运行时,该策略可能会过早止损,错失趋势利润。
2. 参数设置对策略表现影响很大,针对不同品种和周期需要分别进行参数优化。
3. 策略在震荡市更有优势,趋势市表现一般。若长期趋势突然反转,该策略可能会产生较大回撤。
4. 如果标的资产波动率发生剧烈变化,固定的参数设置可能会失效。

#### 优化方向  
1. 可以尝试引入趋势判断指标,如长短期均线交叉、ADX等,对趋势和震荡市进行区分,使用不同的参数设置。
2. 考虑使用适应性更强的波动率指标,如ATR,对波动率通道宽度进行动态调整。
3. 在开仓前对价格走势进行趋势判断,检测是否处于明确的趋势中,避免逆势交易。
4. 可以通过遗传算法、网格搜索等方法对不同参数组合进行优化,寻找最佳的参数设置。
5. 考虑分别针对多头和空头仓位使用不同的参数设置,控制风险敞口。

#### 总结
该策略通过波动率通道和相对强弱指数相结合的方式,在价格波动的同时参考RSI指标进行开平仓判断,能够较好地把握阶段性趋势,及时止损和获利了结。但是策略的表现对参数设置比较敏感,需要针对不同市场环境和标的资产进行优化,同时考虑引入其他指标对市场趋势进行辅助判断,才能充分发挥该策略的优势。总的来说,该策略思路清晰,逻辑严谨,是一个不错的量化交易策略。

|| 

#### Overview
This Pine Script strategy is based on the Relative Strength Index (RSI) and the standard deviation (DEV) of price volatility. It determines entry points by comparing the price with upper and lower bands, while using RSI as an auxiliary filtering indicator. It generates long entry signals when the price breaks above the lower band and RSI is below the oversold threshold, and short entry signals when the price breaks below the upper band and RSI is above the overbought threshold. The strategy closes long positions when the price breaks below the exit lower band or RSI exceeds the overbought threshold, and closes short positions when the price breaks above the exit upper band or RSI falls below the oversold threshold. This strategy can dynamically adjust according to market volatility conditions, cutting losses in time during high volatility and holding positions for profit during low volatility. It is a quantitative trading strategy that can adapt to different market states.

#### Strategy Principle
1. Calculate the Simple Moving Average (SMA) and Standard Deviation (DEV) of the price over the past "length" periods.
2. Construct a volatility channel with SMA as the centerline, SMA+thresholdEntry*DEV as the upper band, and SMA-thresholdEntry*DEV as the lower band.
3. Simultaneously calculate the RSI indicator of the closing price over the past "rsiLength" periods.
4. When the price breaks above the lower band and RSI is below the oversold threshold "rsiOversold", generate a long entry signal.
5. When the price breaks below the upper band and RSI is above the overbought threshold "rsiOverbought", generate a short entry signal.
6. Construct another narrower exit channel with SMA as the centerline, SMA+thresholdExit*DEV as the upper band, and SMA-thresholdExit*DEV as the lower band.
7. When holding a long position, if the price breaks below the exit lower band or RSI exceeds the overbought threshold, close the long position.
8. When holding a short position, if the price breaks above the exit upper band or RSI falls below the oversold threshold, close the short position.

#### Advantage Analysis
1. By using both price behavior and momentum indicators for auxiliary judgment, it can effectively filter out false signals.
2. By dynamically adjusting the channel width based on volatility, the strategy can adapt to different market states.
3. By setting two sets of channels, it can cut losses in the early stage of price reversal and control drawdowns, while still being able to hold positions for profit after a trend is formed.
4. The code logic and parameter settings are clear and easy to understand and optimize.

#### Risk Analysis 
1. When the market continues to run in a unilateral trend, the strategy may cut losses too early and miss out on trend profits.
2. Parameter settings have a significant impact on the strategy's performance, and parameter optimization needs to be performed separately for different varieties and time frames.
3. The strategy performs better in oscillating markets and average in trending markets. If a long-term trend suddenly reverses, the strategy may experience a larger drawdown.
4. If the volatility of the underlying asset changes drastically, the fixed parameter settings may become invalid.

#### Optimization Direction
1. Try introducing trend judgment indicators, such as long-short term moving average crossovers, ADX, etc., to distinguish between trending and oscillating markets and use different parameter settings.
2. Consider using more adaptive volatility indicators, such as ATR, to dynamically adjust the width of the volatility channel.
3. Before opening a position, perform trend judgment on the price movement to detect whether it is in a clear trend to avoid counter-trend trading.
4. Use genetic algorithms, grid search, and other methods to optimize different parameter combinations and find the best parameter settings.
5. Consider using different parameter settings for long and short positions to control risk exposure.

#### Summary
This strategy combines volatility channels and the Relative Strength Index to make entry and exit decisions based on price fluctuations while referencing the RSI indicator. It can better capture short-term trends and cut losses and take profits in a timely manner. However, the strategy's performance is relatively sensitive to parameter settings and needs to be optimized for different market environments and underlying assets. At the same time, consider introducing other indicators to assist in judging market trends in order to fully leverage the advantages of this strategy. Overall, this strategy has a clear idea, rigorous logic, and is a good quantitative trading strategy.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-05-20 00:00:00
end: 2024-05-27 00:00:00
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © tmalvao

//@version=5
strategy("Estratégia de Desvio Padrão com RSI", overlay=true, margin_long=100, margin_short=100)

// Parâmetros
length = input.int(20, title="Período do Desvio Padrão")
thresholdEntry = input.float(1.5, title="Limite de Entrada")
thresholdExit = input.float(0.5, title="Limite de Saída")
rsiLength = input.int(14, title="Período do RSI")
rsiOverbought = input.int(70, title="RSI Overbought")
rsiOversold = input.int(30, title="RSI Oversold")

// Cálculo do Desvio Padrão
price = close
stdDev = ta.stdev(price, length)

// Média Móvel Simples
sma = ta.sma(price, length)

// Limites baseados no Desvio Padrão
upperLimit = sma + thresholdEntry * stdDev
lowerLimit = sma - thresholdEntry * stdDev
exitUpperLimit = sma + thresholdExit * stdDev
exitLowerLimit = sma - thresholdExit * stdDev

// Cálculo do RSI
rsi = ta.rsi(price, rsiLength)

// Condições de Entrada com RSI
longCondition = ta.crossover(price, lowerLimit) and rsi < rsiOversold
shortCondition = ta.crossunder(price, upperLimit) and rsi > rsiOverbought

// Condições de Saída com RSI
exitLongCondition = ta.crossunder(price, exitLowerLimit) or rsi > rsiOverbought
exitShortCondition = ta.crossover(price, exitUpperLimit) or rsi < rsiOversold

// Plotar Linhas
plot(upperLimit, color=color.red, title="Limite Superior")
plot(lowerLimit, color=color.green, title="Limite Inferior")
plot(exitUpperLimit, color=color.orange, title="Limite de Saída Superior")
plot(exitLowerLimit, color=color.blue, title="Limite de Saída Inferior")
plot(sma, color=color.gray, title="SMA")
hline(rsiOverbought, "RSI Overbought", color=color.red)
hline(rsiOversold, "RSI Oversold", color=color.green)
plot(rsi, title="RSI", color=color.purple)

// Estratégia de Trade
if (longCondition)
    strategy.entry("Long", strategy.long)

if (shortCondition)
    strategy.entry("Short", strategy.short)

if (exitLongCondition)
    strategy.close("Long")

if (exitShortCondition)
    strategy.close("Short")



```

> Detail

https://www.fmz.com/strategy/452691

> Last Modified

2024-05-28 10:57:06
