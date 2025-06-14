
> Name

超级趋势与布林带组合策略-Supertrend-and-Bollinger-Bands-Combination-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/d92bacf6999be07086.png)
[trans]
#### 概述
该策略结合了超级趋势指标和布林带指标,旨在捕捉市场的趋势性机会。超级趋势指标用于判断当前市场的趋势方向,而布林带指标用于衡量市场的波动率。当收盘价突破超级趋势线且位于布林带下轨时产生做多信号,当收盘价跌破超级趋势线且位于布林带上轨时产生做空信号。该策略的优势在于能够在趋势明确时及时入场,同时避免在震荡市中过早入场。

#### 策略原理
1. 计算真实波幅(ATR)和超级趋势指标,用于判断当前市场的趋势方向。
2. 计算布林带上下轨,用于衡量市场的波动率。 
3. 当收盘价突破超级趋势线且位于布林带下轨时,产生做多信号;当收盘价跌破超级趋势线且位于布林带上轨时,产生做空信号。
4. 当持有多头仓位时,如果收盘价跌破超级趋势线则平仓;当持有空头仓位时,如果收盘价突破超级趋势线则平仓。

#### 策略优势
1. 结合趋势和波动率两个维度的信息,能够更全面地把握市场机会。
2. 在趋势明确时能够及时入场,有助于捕捉趋势行情的收益。
3. 在震荡市中,布林带与超级趋势结合能够有效过滤掉假突破信号,降低震荡行情下的亏损风险。
4. 代码逻辑清晰,参数较少,易于理解和实现。

#### 策略风险
1. 在单边趋势行情下,由于频繁出现突破信号,可能会导致交易频率过高,增加交易成本。
2. 对突破点的捕捉依赖于超级趋势指标,而该指标对参数较为敏感,不同参数下指标走势差异较大,可能影响策略效果。
3. 布林带宽度会随着市场波动率的变化而变化,在高波动率环境下可能会扩大止损。

#### 策略优化方向 
1. 可以考虑引入更多有效的过滤条件,如交易量、市场情绪等,以进一步提高信号的可靠性。
2. 对于超级趋势指标的参数,可以进行优化测试,选择最佳参数以提高策略稳定性。
3. 在交易执行方面,可以引入更为细致的仓位管理和风险控制措施,如设置移动止损、动态调整仓位等,以降低单次交易的风险敞口。

#### 总结
超级趋势布林带组合策略是一个趋势追踪型策略,通过结合趋势和波动率两个市场要素,能够比较有效地捕捉趋势性机会。但该策略也存在一定的局限性,如对参数敏感、在高波动率环境下风险加大等。因此,在实际应用中还需要根据市场特点和自身风险偏好,对策略进行适当的优化和改进。

|| 

#### Overview
This strategy combines the Supertrend indicator and the Bollinger Bands indicator to capture trending opportunities in the market. The Supertrend indicator is used to determine the current market trend direction, while the Bollinger Bands indicator is used to measure market volatility. A long signal is generated when the closing price breaks above the Supertrend line and is below the lower Bollinger Band, and a short signal is generated when the closing price breaks below the Supertrend line and is above the upper Bollinger Band. The advantage of this strategy is that it can enter the market in a timely manner when the trend is clear, while avoiding premature entry in a choppy market.

#### Strategy Principles
1. Calculate the Average True Range (ATR) and Supertrend indicator to determine the current market trend direction.
2. Calculate the upper and lower Bollinger Bands to measure market volatility.
3. Generate a long signal when the closing price breaks above the Supertrend line and is below the lower Bollinger Band; generate a short signal when the closing price breaks below the Supertrend line and is above the upper Bollinger Band.
4. When holding a long position, if the closing price breaks below the Supertrend line, close the position; when holding a short position, if the closing price breaks above the Supertrend line, close the position.

#### Strategy Advantages
1. Combining information from both the trend and volatility dimensions can more comprehensively grasp market opportunities.
2. Able to enter the market in a timely manner when the trend is clear, which helps capture the gains of trending markets.
3. In a choppy market, the combination of Bollinger Bands and Supertrend can effectively filter out false breakout signals and reduce the risk of losses.
4. The code logic is clear, with few parameters, and is easy to understand and implement.

#### Strategy Risks
1. In a unilateral trending market, due to frequent breakout signals, it may lead to excessive trading frequency and increase transaction costs.
2. The capture of breakout points relies on the Supertrend indicator, which is sensitive to parameters, and the indicator trends vary greatly under different parameters, which may affect the strategy's effectiveness.
3. The width of the Bollinger Bands will change with changes in market volatility, and may widen stop-losses in high-volatility environments.

#### Strategy Optimization Directions
1. Consider introducing more effective filtering conditions, such as trading volume, market sentiment, etc., to further improve the reliability of signals.
2. For the parameters of the Supertrend indicator, optimization tests can be conducted to select the optimal parameters to improve strategy stability.
3. In terms of trade execution, more detailed position management and risk control measures can be introduced, such as setting trailing stops, dynamically adjusting positions, etc., to reduce the risk exposure of a single trade.

#### Summary
The Supertrend Bollinger Band combination strategy is a trend-following strategy that can effectively capture trending opportunities by combining two market factors: trend and volatility. However, this strategy also has certain limitations, such as being sensitive to parameters and increased risk in high-volatility environments. Therefore, in actual application, it is necessary to appropriately optimize and improve the strategy according to market characteristics and one's own risk preferences.

[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_1|3|Supertrend Factor|
|v_input_2|10|ATR Length|
|v_input_3|20|Bollinger Bands Length|
|v_input_4|2|Bollinger Bands Deviation|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-03-21 00:00:00
end: 2024-03-28 00:00:00
period: 5m
basePeriod: 1m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © sabhiv27

//@version=4
strategy("Supertrend & Bollinger Bands Strategy", shorttitle="ST_BB_Strategy", overlay=true)

// Input options
factor = input(3, title="Supertrend Factor")
length = input(10, title="ATR Length")
bollinger_length = input(20, title="Bollinger Bands Length")
bollinger_deviation = input(2, title="Bollinger Bands Deviation")

// Calculate True Range for Supertrend
truerange = rma(tr, length)

// Calculate Supertrend
var float up_trend = na
var float dn_trend = na
var float trend = na
up_signal = hl2 - (factor * truerange)
dn_signal = hl2 + (factor * truerange)
up_trend := close[1] > up_trend[1] ? max(up_signal, up_trend[1]) : up_signal
dn_trend := close[1] < dn_trend[1] ? min(dn_signal, dn_trend[1]) : dn_signal
trend := close > dn_trend ? 1 : close < up_trend ? -1 : nz(trend[1], 1)

// Calculate Bollinger Bands
basis = sma(close, bollinger_length)
dev = stdev(close, bollinger_length)
upper_band = basis + bollinger_deviation * dev
lower_band = basis - bollinger_deviation * dev

// Entry conditions
long_condition = crossover(close, up_trend) and close < lower_band
short_condition = crossunder(close, dn_trend) and close > upper_band

// Exit conditions
exit_long_condition = crossover(close, dn_trend)
exit_short_condition = crossunder(close, up_trend)

// Plot Supertrend
plot(trend == 1 ? up_trend : dn_trend, color=trend == 1 ? color.green : color.red, linewidth=2)

// Plot Bollinger Bands
plot(upper_band, color=color.blue)
plot(lower_band, color=color.blue)

// Generate buy and sell signals
strategy.entry("Long", strategy.long, when=long_condition)
strategy.entry("Short", strategy.short, when=short_condition)
strategy.close("Long", when=exit_long_condition)
strategy.close("Short", when=exit_short_condition)
```

> Detail

https://www.fmz.com/strategy/446546

> Last Modified

2024-03-29 15:18:22
