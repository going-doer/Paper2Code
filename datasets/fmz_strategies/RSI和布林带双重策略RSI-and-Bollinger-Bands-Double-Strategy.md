
> Name

RSI和布林带双重策略RSI-and-Bollinger-Bands-Double-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/124fb999ee4214b2801.png)

[trans]
#### 概述
该策略结合了相对强弱指数(RSI)和布林带(Bollinger Bands)两个技术指标,当价格低于布林带下轨时产生买入信号,当价格高于布林带上轨时产生卖出信号。该策略只有在RSI指标和布林带指标同时处于超卖或超买状态时才会触发交易信号。

#### 策略原理
1. 根据设定的RSI参数计算RSI值。 
2. 使用布林带公式计算布林带中轨、上轨和下轨。
3. 判断当前收盘价是否突破布林带上轨或下轨。
4. 判断当前RSI值是否高于超买阈值或低于超卖阈值。
5. 当布林带和RSI指标同时满足买入或卖出条件时,产生相应的交易信号。

#### 策略优势
1. 结合了趋势和动量两种技术指标,能够更全面地判断市场状态。
2. 同时使用两个指标作为过滤条件,有效降低了假信号的出现概率。
3. 代码逻辑清晰,参数设置灵活,适合不同市场环境和交易风格。

#### 策略风险
1. 在震荡市中,该策略可能会产生较多的亏损交易。
2. 参数设置不当可能导致策略表现欠佳,需要根据实际情况进行优化。
3. 该策略未设置止损,可能面临较大的回撤风险。

#### 策略优化方向
1. 可以根据市场特点和个人偏好,对RSI和布林带的参数进行优化。
2. 引入其他技术指标如MACD、均线等,提高信号的可靠性。
3. 设置合理的止损和止盈,控制单笔交易风险。
4. 对于震荡市,可以考虑增加判断条件或减小仓位,降低频繁交易带来的成本。

#### 总结
RSI和布林带双重策略通过结合趋势和动量指标,能够比较全面地判断市场状态,并给出相应的交易信号。但该策略在震荡市中表现可能欠佳,且未设置风险控制措施,因此实盘运用时需要谨慎。通过优化参数、引入其他指标和设置合理的止损止盈等方式,可以进一步提升该策略的稳定性和盈利能力。

|| 

#### Overview
This strategy combines the Relative Strength Index (RSI) and Bollinger Bands technical indicators. It generates buy signals when the price falls below the lower Bollinger Band and sell signals when the price rises above the upper Bollinger Band. The strategy only triggers trading signals when both the RSI and Bollinger Bands indicators are simultaneously in an oversold or overbought state.

#### Strategy Logic
1. Calculate the RSI value based on the set RSI parameters.
2. Use the Bollinger Bands formula to calculate the middle, upper, and lower Bollinger Bands.
3. Determine if the current closing price breaks through the upper or lower Bollinger Band.
4. Determine if the current RSI value is above the overbought threshold or below the oversold threshold.
5. Generate corresponding buy or sell signals when both the Bollinger Bands and RSI indicators meet the respective conditions.

#### Strategy Advantages
1. Combines trend and momentum indicators for a more comprehensive assessment of market conditions.
2. Using two indicators as filters effectively reduces the probability of false signals.
3. Clear code logic and flexible parameter settings, suitable for different market environments and trading styles.

#### Strategy Risks
1. In choppy markets, this strategy may generate more losing trades.
2. Improper parameter settings may lead to poor strategy performance and require optimization based on actual conditions.
3. The strategy does not include a stop-loss, potentially exposing it to significant drawdown risk.

#### Strategy Optimization Directions
1. Optimize RSI and Bollinger Bands parameters based on market characteristics and personal preferences.
2. Introduce additional technical indicators such as MACD, moving averages, etc., to improve signal reliability.
3. Set reasonable stop-loss and take-profit levels to control single-trade risk.
4. For choppy markets, consider adding more conditions or reducing position size to lower costs associated with frequent trading.

#### Summary
The RSI and Bollinger Bands Double Strategy combines trend and momentum indicators to provide a relatively comprehensive assessment of market conditions and generate corresponding trading signals. However, the strategy may underperform in choppy markets and lacks risk control measures, so caution is needed when applying it to live trading. By optimizing parameters, introducing other indicators, and setting reasonable stop-loss and take-profit levels, the stability and profitability of this strategy can be further improved.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_1|14|RSI Period Length|
|v_input_2|20|Bollinger Period Length|
|v_input_3|2|Bollinger Bands Standard Deviation|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-03-01 00:00:00
end: 2024-03-31 23:59:59
period: 4h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=4
strategy("Bollinger + RSI, Double Strategy (by ChartArt) v1.1", shorttitle="CA_-_RSI_Bol_Strat_1.1", overlay=true)

// ChartArt's RSI + Bollinger Bands, Double Strategy - Update
//
// Version 1.1
// Idea by ChartArt on January 18, 2015.
//
// This strategy uses the RSI indicator 
// together with the Bollinger Bands 
// to sell when the price is above the
// upper Bollinger Band (and to buy when
// this value is below the lower band).
//
// This simple strategy only triggers when
// both the RSI and the Bollinger Bands
// indicators are at the same time in
// a overbought or oversold condition.
//
// In this version 1.1 the strategy was
// both simplified for the user and
// made more successful in backtesting. 
//
// List of my work: 
// https://www.tradingview.com/u/ChartArt/
// 
//  __             __  ___       __  ___ 
// /  ` |__|  /\  |__)  |   /\  |__)  |  
// \__, |  | /~~\ |  \  |  /~~\ |  \  |  
// 
// 

///////////// RSI
RSIlength = input(14,title="RSI Period Length") 
RSIoverSold = 30
RSIoverBought = 70
price = close
vrsi = rsi(price, RSIlength)


///////////// Bollinger Bands
BBlength = input(20, minval=1,title="Bollinger Period Length")
BBmult = input(2.0, minval=0.001, maxval=50,title="Bollinger Bands Standard Deviation")
BBbasis = sma(price, BBlength)
BBdev = BBmult * stdev(price, BBlength)
BBupper = BBbasis + BBdev
BBlower = BBbasis - BBdev
source = close
buyEntry = crossover(source, BBlower)
sellEntry = crossunder(source, BBupper)
plot(BBbasis, color=color.blue,title="Bollinger Bands SMA Basis Line")
p1 = plot(BBupper, color=color.red,title="Bollinger Bands Upper Line")
p2 = plot(BBlower, color=color.green,title="Bollinger Bands Lower Line")
fill(p1, p2)

// Entry conditions
crossover_rsi = crossover(vrsi, RSIoverSold) and crossover(source, BBlower)
crossunder_rsi = crossunder(vrsi, RSIoverBought) and crossunder(source, BBupper)

///////////// RSI + Bollinger Bands Strategy
if (not na(vrsi))
    if (crossover_rsi)
        strategy.entry("RSI_BB_L", strategy.long, comment="RSI_BB_L")
    else
        strategy.cancel(id="RSI_BB_L")
        
    if (crossunder_rsi)
        strategy.entry("RSI_BB_S", strategy.short, comment="RSI_BB_S")
    else
        strategy.cancel(id="RSI_BB_S")

```

> Detail

https://www.fmz.com/strategy/446986

> Last Modified

2024-04-03 17:54:52
