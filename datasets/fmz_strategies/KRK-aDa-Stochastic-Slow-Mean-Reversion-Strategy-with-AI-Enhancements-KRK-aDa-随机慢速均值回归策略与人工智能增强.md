
> Name

KRK-aDa-Stochastic-Slow-Mean-Reversion-Strategy-with-AI-Enhancements-KRK-aDa-随机慢速均值回归策略与人工智能增强

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1adcd20e8cd597ea5b8.png)

[trans]

#### 概述

该策略使用随机慢速指标(Stochastic Slow)作为主要的交易信号,并结合200期简单移动平均线(SMA)作为趋势过滤器。此外,该策略还引入了一个虚拟的人工智能(AI)指标,以提供额外的入场信号。策略的主要思路是在超卖区域买入,超买区域卖出,同时确保价格在200 SMA上方买入,在200 SMA下方卖出,以顺应当前趋势。AI指标的加入为策略提供了更多的入场机会。

#### 策略原理

1. 计算随机慢速指标的K值和D值,其中K值的周期为26,D值为K值的3周期SMA。

2. 设定超买区(OverBought)为81,超卖区(OverSold)为20,最小K值(minKValue)为11。

3. 当K线上穿D线且K值小于超卖区且大于最小K值时,产生买入信号。

4. 当K线下穿D线且K值大于超买区且大于最小K值时,产生卖出信号。

5. 使用200期SMA作为趋势过滤器,当价格在200 SMA上方时允许买入,在200 SMA下方时允许卖出。

6. 引入虚拟AI指标(使用RSI>50表示看涨,RSI<50表示看跌),当AI指标看涨时买入,看跌时卖出。

7. 综合随机指标、趋势过滤器和AI指标的信号,产生最终的交易信号。

8. 买入时设置10%的止损,卖出时设置10%的止损。

#### 策略优势

1. 随机慢速指标有效地识别了市场的超买和超卖区域,为交易提供了良好的入场点。

2. 引入200 SMA作为趋势过滤器,确保交易与当前趋势保持一致,提高了成功率。

3. 加入AI指标为策略提供了更多的入场机会,有可能提高策略的收益。

4. 设置止损订单,有效控制了风险。

#### 策略风险

1. 随机指标在震荡市场中可能会产生较多的虚假信号。

2. AI指标目前只是一个虚拟指标,实际效果有待验证。

3. 止损设置可能会导致部分收益被过早止损出场。

#### 策略优化方向

1. 对随机指标的参数进行优化,找到最佳的周期和超买超卖阈值设置。

2. 引入更加复杂和有效的AI模型,提高AI信号的准确性。

3. 优化止损和止盈设置,以更好地控制风险和锁定收益。

4. 考虑引入其他有效的技术指标或基本面数据,以提高策略的robustness。

#### 总结

该策略通过结合随机慢速指标、趋势过滤器和AI信号,形成了一个多因子交易策略。随机指标提供了有效的超买超卖信号,趋势过滤器确保交易方向与大趋势一致,AI信号则为策略提供了更多的入场机会。尽管该策略还有一些潜在的风险和优化空间,但其整体思路清晰,逻辑合理,值得进一步探索和改进。

|| 

#### Overview

This strategy utilizes the Stochastic Slow indicator as the primary trading signal, combined with a 200-period Simple Moving Average (SMA) as a trend filter. Additionally, the strategy introduces a dummy Artificial Intelligence (AI) indicator to provide extra entry signals. The main idea is to buy in oversold areas and sell in overbought areas, while ensuring that the price is above the 200 SMA for long entries and below the 200 SMA for short entries, aligning with the current trend. The inclusion of the AI indicator offers more entry opportunities.

#### Strategy Principles

1. Calculate the K and D values of the Stochastic Slow indicator, with the K period set to 26 and the D value being a 3-period SMA of the K value.

2. Set the overbought level (OverBought) to 81, the oversold level (OverSold) to 20, and the minimum K value (minKValue) to 11.

3. Generate a buy signal when the K line crosses above the D line, and the K value is below the oversold level and above the minimum K value.

4. Generate a sell signal when the K line crosses below the D line, and the K value is above the overbought level and above the minimum K value.

5. Use the 200-period SMA as a trend filter, allowing long entries only when the price is above the 200 SMA and short entries when the price is below the 200 SMA.

6. Introduce a dummy AI indicator (using RSI>50 for bullish and RSI<50 for bearish), entering long when the AI signal is bullish and short when it is bearish.

7. Combine the signals from the Stochastic indicator, trend filter, and AI indicator to generate the final trading signals.

8. Set a 10% stop loss for long entries and a 10% stop loss for short entries.

#### Strategy Advantages

1. The Stochastic Slow indicator effectively identifies overbought and oversold areas in the market, providing good entry points for trades.

2. The 200 SMA trend filter ensures that trades align with the current trend, increasing the success rate.

3. The inclusion of the AI indicator offers more entry opportunities, potentially increasing the strategy's profitability.

4. The use of stop-loss orders effectively manages risk.

#### Strategy Risks

1. The Stochastic indicator may generate false signals in choppy markets.

2. The AI indicator is currently a dummy indicator, and its actual effectiveness needs to be verified.

3. The stop-loss settings may lead to some profits being cut short prematurely.

#### Strategy Optimization Directions

1. Optimize the parameters of the Stochastic indicator to find the best period and overbought/oversold threshold settings.

2. Introduce more complex and effective AI models to improve the accuracy of AI signals.

3. Fine-tune the stop-loss and take-profit settings for better risk control and profit capture.

4. Consider incorporating other effective technical indicators or fundamental data to enhance the strategy's robustness.

#### Summary

This strategy combines the Stochastic Slow indicator, trend filter, and AI signals to form a multi-factor trading approach. The Stochastic indicator provides effective overbought and oversold signals, the trend filter ensures that trades align with the overall trend, and the AI signals offer additional entry opportunities. Although the strategy has some potential risks and room for improvement, its overall logic is clear and reasonable, making it worth further exploration and refinement.

[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_int_1|26|length|
|v_input_1|81|OverBought|
|v_input_2|20|OverSold|
|v_input_int_2|3|smoothK|
|v_input_int_3|3|smoothD|
|v_input_3|11|Minimum K Value|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-03-01 00:00:00
end: 2024-03-31 23:59:59
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Stochastic Slow Strategy with More Entries and AI", overlay=true)

length = input.int(26, minval=1)
OverBought = input(81)
OverSold = input(20)
smoothK = input.int(3, minval=1)
smoothD = input.int(3, minval=1)
minKValue = input(11, title="Minimum K Value")

// Stochastic calculations
k = ta.sma(ta.stoch(close, high, low, length), smoothK)
d = ta.sma(k, smoothD)
co = ta.crossover(k, d)
cu = ta.crossunder(k, d)

// Trend filter (200-period simple moving average)
ema200 = ta.sma(close, 200)

// Artificial Intelligence indicator (dummy example)
// Aquí puedes colocar la lógica de tu red neuronal artificial
// Por ahora, simplemente usaremos una señal aleatoria
ai_signal = ta.rsi(close, 14) > 50 ? 1 : -1

// Entry conditions
longCondition = ta.crossover(close, ema200) and k < OverSold and k > minKValue and ai_signal == 1
shortCondition = ta.crossunder(close, ema200) and k > OverBought and k > minKValue and ai_signal == -1

if (not na(k) and not na(d))
    if (co and k < OverSold and k > minKValue)
        strategy.entry("StochLE", strategy.long, comment="StochLE")
    if (cu and k > OverBought and k > minKValue)
        strategy.entry("StochSE", strategy.short, comment="StochSE")
    if (longCondition)
        strategy.entry("LongEntry", strategy.long, comment="LongEntry")
        strategy.exit("StopLoss", "LongEntry", loss = close * 0.9) // Stop loss del 10%
    if (shortCondition)
        strategy.entry("ShortEntry", strategy.short, comment="ShortEntry")
        strategy.exit("StopLoss", "ShortEntry", loss = close * 1.1) // Stop loss del 10%

// Plotting
plot(ema200, color=color.blue, title="200 SMA")

```

> Detail

https://www.fmz.com/strategy/449521

> Last Modified

2024-04-26 15:41:18
