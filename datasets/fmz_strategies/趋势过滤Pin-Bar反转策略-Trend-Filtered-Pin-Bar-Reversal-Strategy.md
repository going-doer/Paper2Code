
> Name

趋势过滤Pin-Bar反转策略-Trend-Filtered-Pin-Bar-Reversal-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1c2f7d3bb94b33811a3.png)
[trans]
#### 概述
该策略主要通过识别特定的K线形态--Pin Bar来判断潜在的市场反转点。Pin Bar是一种由长影线和小实体组成的K线形态,表明市场在该点位出现了较大的波动,但最终价格回撤,显示出该点位可能是一个支撑或阻力位。该策略采用50周期简单移动平均线(SMA)来判断当前趋势方向,并使用20周期SMA作为交易量过滤条件,只有在Pin Bar出现时交易量大于该均线才认为是有效信号。此外,该策略还计算了相对强弱指标(RSI),但并未直接用于入场和出场条件中,而是作为进一步过滤信号的可选条件。

#### 策略原理
1. 首先判断Pin Bar的上下影线和实体的相对大小,要求上影线或下影线至少为整个K线高低点范围的60%,而实体部分不能超过30%。
2. 通过比较收盘价和开盘价的大小来判断Pin Bar是看涨还是看跌。
3. 使用50周期SMA判断当前趋势,当收盘价在SMA上方时认为是上升趋势,反之则为下降趋势。
4. 以20周期交易量SMA作为交易量过滤条件的阈值,只有Pin Bar实现时的交易量大于该值才认为是有效信号。
5. 画出识别到的看涨和看跌Pin Bar。
6. 看涨Pin Bar出现时开多头仓位,看跌Pin Bar出现时开空头仓位。
7. 止损设置为Pin Bar实体大小的2倍,止盈设置为3倍。多头止损位于Pin Bar最低点下方,止盈位于最高点上方;空头反之。

#### 优势分析
1. Pin Bar是一种非常直观有效的价格反转形态,能够比较准确地捕捉到市场情绪的突变。
2. 趋势过滤条件能够保证Pin Bar信号和当前趋势方向一致,大大提高了信号的胜率。
3. 交易量条件能够过滤掉流动性不足的市场噪音,确保Pin Bar信号有足够的市场参与度。
4. 止损和止盈位置的设置基于Pin Bar本身的特征,风险收益比合理。
5. 代码逻辑和规则清晰,易于理解和实现。

#### 风险分析
1. Pin Bar信号的可靠性在震荡市中会大打折扣,此时趋势过滤作用不大。
2. 遇到特大利空或利多事件时,Pin Bar也可能失效。
3. 交易频率偏低,回测样本可能不足。
4. 对于特定品种和周期,默认参数可能还需要进一步优化。
5. 单一信号系统整体上风险较大。

#### 优化方向
1. 可以考虑同时引入其他反转形态如内部Bar等来丰富信号来源。
2. 止损和止盈的位置可以用ATR等波动率指标动态调整以适应不同的市场状态。
3. 可以用百分比追踪止损的方式来放大利润。
4. 可以增加更多基本面数据如经济日历和重大事件等来过滤可能失效的信号。
5. 考虑引入资金管理模块,控制每次交易的风险敞口。

#### 总结
该Pin Bar反转策略采用了简单有效的思路,通过趋势过滤、交易量过滤等方式提高了信号识别的准确性。虽然目前还有一些可以改进的地方,但整体思路是可行的,值得进一步优化测试。Pin Bar本身作为一种经典的价格形态,也可以和其他指标或信号结合使用,以期获得更稳健的交易系统。

|| 

#### Overview
This strategy primarily aims to identify potential market reversal points by recognizing a specific candlestick pattern called the Pin Bar. A Pin Bar is characterized by a long shadow and a small body, indicating significant market volatility at that price level, but ultimately the price retraces, suggesting that the level may act as a support or resistance. The strategy utilizes a 50-period Simple Moving Average (SMA) to determine the current trend direction and a 20-period SMA of volume as a filter, requiring the volume to be above this average for a Pin Bar signal to be considered valid. Additionally, the Relative Strength Index (RSI) is calculated but not directly used in entry/exit conditions, serving instead as an optional further filtering condition.

#### Strategy Principles
1. First, determine the relative size of the Pin Bar's upper and lower shadows and body, requiring the upper or lower shadow to be at least 60% of the entire candlestick's high-low range, while the body should not exceed 30%.
2. Compare the closing and opening prices to determine if the Pin Bar is bullish or bearish.
3. Use the 50-period SMA to identify the current trend, considering it an uptrend when the closing price is above the SMA and a downtrend when below.
4. Set the 20-period volume SMA as the threshold for the volume filter, only considering a Pin Bar signal valid if the volume at its occurrence is greater than this value.
5. Plot the identified bullish and bearish Pin Bars.
6. Enter a long position when a bullish Pin Bar appears and a short position when a bearish Pin Bar appears.
7. Set the stop loss to twice the size of the Pin Bar's body and the take profit to three times. For long positions, place the stop loss below the Pin Bar's low and the take profit above its high; vice versa for short positions.

#### Advantage Analysis
1. The Pin Bar is a highly intuitive and effective price reversal pattern, capable of accurately capturing sudden changes in market sentiment.
2. The trend filter ensures that Pin Bar signals align with the current trend direction, significantly improving the signal's win rate.
3. The volume condition filters out market noise with insufficient liquidity, ensuring that Pin Bar signals have adequate market participation.
4. The stop loss and take profit positions are set based on the Pin Bar's characteristics, providing a reasonable risk-reward ratio.
5. The code logic and rules are clear and easy to understand and implement.

#### Risk Analysis
1. The reliability of Pin Bar signals may be greatly diminished in choppy markets, where the trend filter is less effective.
2. Pin Bars may fail in the face of exceptionally strong bearish or bullish events.
3. The trading frequency is relatively low, potentially leading to insufficient backtest samples.
4. Default parameters may require further optimization for specific instruments and timeframes.
5. As a single-signal system, the overall risk is relatively high.

#### Optimization Directions
1. Consider introducing other reversal patterns such as Inside Bars to enrich signal sources.
2. Use volatility indicators like ATR to dynamically adjust stop loss and take profit positions to adapt to different market conditions.
3. Implement a percentage trailing stop to maximize profits.
4. Incorporate more fundamental data, such as economic calendars and significant events, to filter out potentially invalid signals.
5. Introduce a money management module to control the risk exposure of each trade.

#### Summary
This Pin Bar reversal strategy employs a simple and effective approach, using trend filtering and volume filtering to improve signal recognition accuracy. Although there is room for improvement, the overall concept is viable and worthy of further optimization and testing. As a classic price pattern, the Pin Bar can also be combined with other indicators or signals to achieve a more robust trading system.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-05-01 00:00:00
end: 2024-05-31 23:59:59
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Filtered Pin Bar Strategy with Relaxed Volume", overlay=true)

// Define the size of the pin bar's wick and body
wickSize = 0.6
bodySize = 0.3

// Calculate the size of the wicks and body
upperWick = high - math.max(open, close)
lowerWick = math.min(open, close) - low
body = math.abs(close - open)

// Define a simple moving average to determine the trend
smaLength = 50
sma = ta.sma(close, smaLength)

// Define a more relaxed volume threshold
volumeThreshold = ta.sma(volume, 20) * 1.0

// Define RSI parameters
rsiLength = 14
rsiOverbought = 70
rsiOversold = 30
rsi = ta.rsi(close, rsiLength)

// Define the conditions for a bullish pin bar
bullishPinBar = (lowerWick > (wickSize * (high - low))) and
     (body < (bodySize * (high - low))) and
     (close > open) and
     (close > sma) and
     (volume > volumeThreshold)

// Define the conditions for a bearish pin bar
bearishPinBar = (upperWick > (wickSize * (high - low))) and
     (body < (bodySize * (high - low))) and
     (close < open) and
     (close < sma) and
     (volume > volumeThreshold)

// Plot the bullish and bearish pin bars on the chart
plotshape(series=bullishPinBar, title="Bullish Pin Bar", location=location.belowbar, color=color.green, style=shape.labelup, text="PB")
plotshape(series=bearishPinBar, title="Bearish Pin Bar", location=location.abovebar, color=color.red, style=shape.labeldown, text="PB")

// Entry and exit rules
if (bullishPinBar)
    strategy.entry("Bullish Pin Bar", strategy.long)
if (bearishPinBar)
    strategy.entry("Bearish Pin Bar", strategy.short)

// Optional: Set stop loss and take profit
stopLoss = 2 * body
takeProfit = 3 * body
strategy.exit("Exit Long", from_entry="Bullish Pin Bar", stop=low - stopLoss, limit=high + takeProfit)
strategy.exit("Exit Short", from_entry="Bearish Pin Bar", stop=high + stopLoss, limit=low - takeProfit)

```

> Detail

https://www.fmz.com/strategy/453669

> Last Modified

2024-06-07 16:48:23
