
> Name

基于时间序列数据的自适应动态阈值策略-Time-Series-Adaptive-Dynamic-Threshold-Strategy-Based-on-Equity-Data

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1d6ce7d21ce3a67b7e0.png)
[trans]
#### 概述
该策略是一种基于股票或其他金融资产的净值时间序列数据,通过动态计算效率比(ER)作为指数移动平均(EMA)的平滑因子,从而自适应地调整上下轨,实现买卖信号的触发。该策略的主要思路是利用净值数据本身包含的全部信息,通过计算净值变化的复杂程度(ER)来动态调整EMA平滑因子,进而得到动态变化的上下轨。当价格突破上轨时开仓做多,突破下轨时平仓。

#### 策略原理
1. 计算净值数据的效率比(ER),即净值变化量与总变化量之比。ER值越小,说明净值变化越平稳;ER值越大,说明净值变化越剧烈。
2. 将ER作为pine_ema函数的平滑因子alpha,动态计算净值的EMA均值和绝对偏差。
3. 将EMA均值加减绝对偏差,得到动态变化的上下轨。
4. 当前净值突破上轨时开仓做多,突破下轨时平仓。

#### 策略优势
1. 充分利用了净值时间序列数据包含的全部信息,不需要设置任何参数和优化,方法简洁自然。
2. 通过动态计算ER来调整EMA平滑因子,可以自适应净值变化的复杂程度,灵活应对市场变化。
3. 与传统的固定参数EMA相比,动态EMA可以有效减少交易次数和持仓时间,降低交易成本和风险。
4. 可以有效控制回撤。与买入持有相比,该策略可以将最大回撤降低2-3倍,或者在相同回撤下将收益提高2-3倍。
5. 可以方便地应用于多个策略的组合,实现策略自动开关的目的。

#### 策略风险
1. 该策略基于净值时间序列数据,对于价格走势发生根本逆转的情况,触发平仓的速度可能较慢,从而影响收益。
2. 虽然该策略可以自适应地调整参数,但是对于极端行情的适应性还有待进一步考察。
3. 该策略目前主要针对做多的情形,对于做空的情形还需要进一步完善。
4. 在实际应用中,该策略对于选择标的的品质要求较高,需要选择长期走势向上的标的。

#### 策略优化方向
1. 可以考虑将ER的计算方法进一步优化,引入更多反映净值变化特征的指标,提高ER的稳健性和有效性。
2. 可以进一步细化开平仓条件,如考虑加入移动止盈止损、百分比止盈止损等,提高策略的盈利能力和抗风险能力。
3. 针对不同的标的和市场环境,可以对策略进行参数优化和适应性调整,提高策略的普适性。
4. 可以将该策略和其他策略(如趋势跟踪、均值回归等)进行组合,发挥不同策略的优势,提高组合的稳健性和收益性。

#### 总结
该策略通过动态计算效率比(ER)作为指数移动平均(EMA)的平滑因子,自适应地调整上下轨,实现买卖信号的触发。该策略充分利用了净值时间序列数据包含的信息,不需要过多参数设置和优化,方法简洁自然,可以灵活应对市场变化,有效控制回撤。但是该策略对于极端行情的适应性还有待进一步考察,在实际应用中需要注意标的的选择。未来可以从计算方法、开平仓条件、参数优化、策略组合等方面对该策略进行进一步优化和完善,提高策略的稳健性和盈利能力。

|| 

#### Overview
This strategy is based on the time series data of net asset value of stocks or other financial assets. By dynamically calculating the efficiency ratio (ER) as the smoothing factor of the exponential moving average (EMA), it adaptively adjusts the upper and lower bands to trigger buy and sell signals. The main idea of this strategy is to use all the information contained in the net asset value data itself, by calculating the complexity of net asset value changes (ER) to dynamically adjust the EMA smoothing factor, and then obtain the dynamically changing upper and lower bands. When the price breaks through the upper band, it opens a long position, and when it breaks through the lower band, it closes the position.

#### Strategy Principle
1. Calculate the efficiency ratio (ER) of the net asset value data, which is the ratio of the net asset value change to the total change. The smaller the ER value, the more stable the net asset value changes; the larger the ER value, the more dramatic the net asset value changes.
2. Use ER as the smoothing factor alpha of the pine_ema function to dynamically calculate the EMA mean and absolute deviation of the net asset value.
3. Add and subtract the absolute deviation from the EMA mean to obtain the dynamically changing upper and lower bands.
4. When the current net asset value breaks through the upper band, open a long position, and when it breaks through the lower band, close the position.

#### Strategy Advantages
1. It makes full use of all the information contained in the time series data of net asset value, without the need to set any parameters and optimize, the method is simple and natural.
2. By dynamically calculating ER to adjust the EMA smoothing factor, it can adapt to the complexity of net asset value changes and flexibly respond to market changes.
3. Compared with the traditional fixed-parameter EMA, the dynamic EMA can effectively reduce the number of trades and holding time, reducing transaction costs and risks.
4. It can effectively control drawdowns. Compared with buy and hold, this strategy can reduce the maximum drawdown by 2-3 times, or increase the return by 2-3 times under the same drawdown.
5. It can be easily applied to the combination of multiple strategies to achieve the purpose of automatic on/off of strategies.

#### Strategy Risks
1. This strategy is based on the time series data of net asset value. For situations where the price trend reverses fundamentally, the speed of triggering the closing of positions may be slower, thus affecting the return.
2. Although this strategy can adaptively adjust parameters, its adaptability to extreme market conditions needs further examination.
3. This strategy currently mainly focuses on long positions, and needs to be further improved for short positions.
4. In practical applications, this strategy has higher requirements for the quality of the selected targets, and requires the selection of targets with long-term upward trends.

#### Strategy Optimization Directions
1. Consider further optimizing the calculation method of ER, introducing more indicators that reflect the characteristics of net asset value changes, and improving the robustness and effectiveness of ER.
2. Further refine the opening and closing conditions, such as considering adding trailing stop loss, percentage stop loss, etc., to improve the profitability and risk resistance of the strategy.
3. For different targets and market environments, optimize the parameters and adaptively adjust the strategy to improve the versatility of the strategy.
4. Combine this strategy with other strategies (such as trend tracking, mean reversion, etc.) to leverage the advantages of different strategies and improve the robustness and profitability of the portfolio.

#### Summary
This strategy dynamically calculates the efficiency ratio (ER) as the smoothing factor of the exponential moving average (EMA), adaptively adjusts the upper and lower bands, and triggers buy and sell signals. This strategy makes full use of the information contained in the time series data of net asset value, without the need for too many parameter settings and optimizations, the method is simple and natural, and can flexibly respond to market changes and effectively control drawdowns. However, the adaptability of this strategy to extreme market conditions needs further examination, and attention should be paid to the selection of targets in practical applications. In the future, we can further optimize and improve the strategy from the aspects of calculation methods, opening and closing conditions, parameter optimization, strategy combination, etc., to improve the robustness and profitability of the strategy.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_1_close|0|src: close|high|low|open|hl2|hlc3|hlcc4|ohlc4|


> Source (PineScript)

``` pinescript
/*backtest
start: 2023-03-26 00:00:00
end: 2024-03-31 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=4
strategy('Equity control', 'EC')
// study('Exponential bands', 'EB', overlay = true)


er(src) =>
    var start = src
    var total = 0.0

    total += abs(src - nz(src[1], src))
    net    = abs(src - start          )
    
    net / total

pine_ema(src, alpha) =>
    mean = 0.0
    dev  = 0.0

    mean := na(mean[1]) ? src : (1 - alpha) * mean[1] + alpha *     src
    dev  := na(dev [1]) ? 0   : (1 - alpha) * dev [1] + alpha * abs(src - mean)

    [mean, dev]


src = input(close)


a           = er      (src   )
[mean, dev] = pine_ema(src, a)

dev_lower = mean - dev
dev_upper = mean + dev


// plot(dev_lower, 'lower deviation', color.silver, 2, plot.style_stepline)
// plot(mean     , 'basis'          , color.purple, 1, plot.style_stepline)
// plot(dev_upper, 'upper deviation', color.silver, 2, plot.style_stepline)


if src > dev_upper
    strategy.entry('event', true, comment = 'on')
if src < dev_lower
    strategy.close('event', comment = 'off')


plot(strategy.equity)

//bigDope
```

> Detail

https://www.fmz.com/strategy/446752

> Last Modified

2024-04-01 10:48:52
