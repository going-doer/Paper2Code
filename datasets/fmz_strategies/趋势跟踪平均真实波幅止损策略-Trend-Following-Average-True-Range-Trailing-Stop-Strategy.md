
> Name

趋势跟踪平均真实波幅止损策略-Trend-Following-Average-True-Range-Trailing-Stop-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1332d500ca01a4c59de.png)

[trans]
#### 概述
该策略使用平均真实波幅(ATR)作为跟踪止损(TS)的依据,通过动态调整止损位置来实现追踪趋势的目的。当价格向有利方向移动时,止损位置也会随之调整,从而锁定已获得的利润;当价格向不利方向移动时,止损位置保持不变,一旦价格触及止损价格,则平仓止损。该策略的关键在于止损位置的动态调整,既可以保护已获利润,又能让利润随趋势延续而不断扩大。

#### 策略原理
1. 计算ATR,作为跟踪止损的依据。ATR反映了市场波动性,用于衡量价格变动的平均幅度。
2. 根据ATR和KeyValue参数计算止损距离nLoss。KeyValue为用户自定义的倍数,nLoss为KeyValue与ATR的乘积,表示止损距离是ATR的几倍。
3. 计算动态跟踪止损位置xATRTrailingStop。多头持仓时,将其设为"前一根K线的最高价与(收盘价-nLoss)两者的较大值";空头持仓时,将其设为"前一根K线的最低价与(收盘价+nLoss)两者的较小值"。
4. 产生开仓信号。当收盘价上穿xATRTrailingStop时,做多;当收盘价下穿xATRTrailingStop时,做空。

#### 优势分析
1. 止损位置随价格波动而动态调整,既能锁定利润,又能让利润随趋势延续而扩大。
2. 止损位置基于ATR计算,能够客观反映市场波动性,与主观设置的固定止损相比更加灵活有效。
3. 通过KeyValue参数放大ATR,可以根据自己的风险偏好设置合适的止损距离,较大的KeyValue会带来较宽的止损空间和较少的止损频率。

#### 风险分析
1. 趋势型策略在震荡市中表现欠佳,单边趋势不明显时会频繁止损,导致资金快速流失。
2. 入场时机依赖收盘价与动态止损线的交叉信号,在震荡行情中可能出现连续的小止损。
3. 跟踪止损策略无法避免巨大利空或利多带来的缺口跳空,止损位置调整速度跟不上价格变动速度,导致实际亏损远大于预期可控亏损。

#### 优化方向
1. 可以在策略基础上增加趋势判断指标,如均线系统、动量指标等,在趋势明确时才进场,避免在震荡市频繁交易。
2. 可以考虑引入止盈策略,如根据凯利公式计算持仓头寸、设置固定获利点数回撤止盈等,降低趋势末期潜在利润回吐的可能。
3. 针对跳空缺口,可以设置最大止损限制,如固定金额或固定百分比,一旦达到该限额,无论动态止损价位在何处,都立即止损。

#### 总结
ATR跟踪止损策略能够根据价格波动幅度动态调整止损位置,在趋势行情中可以取得不错的效果。但是,该策略也存在无法应对震荡市、止损过于频繁以及难以避免跳空缺口等风险。针对上述缺陷,可以从趋势判断、止盈策略、最大止损限制等方面对策略进行优化和改进。通过这些调整,有望增强策略的适应性和盈利能力。

|| 

#### Overview
This strategy uses the Average True Range (ATR) as the basis for a Trailing Stop (TS), dynamically adjusting the stop-loss position to follow the trend. When the price moves in a favorable direction, the stop-loss position is adjusted accordingly to lock in profits; when the price moves in an adverse direction, the stop-loss position remains unchanged, and once the price reaches the stop-loss price, the position is closed. The key to this strategy lies in the dynamic adjustment of the stop-loss position, which can both protect profits and allow profits to expand as the trend continues.

#### Strategy Principle
1. Calculate the ATR as the basis for the trailing stop. ATR reflects market volatility and is used to measure the average magnitude of price changes.
2. Calculate the stop-loss distance nLoss based on the ATR and the KeyValue parameter. KeyValue is a user-defined multiplier, and nLoss is the product of KeyValue and ATR, indicating that the stop-loss distance is several times the ATR.
3. Calculate the dynamic trailing stop position xATRTrailingStop. For a long position, it is set to the greater of "the highest price of the previous candle and (close - nLoss)"; for a short position, it is set to the lesser of "the lowest price of the previous candle and (close + nLoss)".
4. Generate entry signals. When the closing price crosses above xATRTrailingStop, go long; when the closing price crosses below xATRTrailingStop, go short.

#### Advantage Analysis
1. The stop-loss position is dynamically adjusted with price fluctuations, allowing profits to be locked in while also allowing profits to expand as the trend continues.
2. The stop-loss position is based on the ATR calculation, which can objectively reflect market volatility and is more flexible and effective compared to subjectively set fixed stop-losses.
3. By amplifying the ATR with the KeyValue parameter, you can set an appropriate stop-loss distance based on your risk preference. A larger KeyValue will result in a wider stop-loss space and fewer stop-loss occurrences.

#### Risk Analysis
1. Trend-following strategies perform poorly in choppy markets, and when the unilateral trend is not clear, frequent stop-losses can lead to rapid loss of funds.
2. The entry timing relies on the cross signal between the closing price and the dynamic stop-loss line, which may result in consecutive small stop-losses in a choppy market.
3. Trailing stop strategies cannot avoid gaps caused by significant bearish or bullish news, and the adjustment speed of the stop-loss position cannot keep up with the price change speed, resulting in actual losses far greater than expected controllable losses.

#### Optimization Direction
1. Trend judgment indicators, such as moving average systems and momentum indicators, can be added to the strategy to only enter the market when the trend is clear, avoiding frequent trading in choppy markets.
2. A take-profit strategy can be considered, such as calculating position sizes based on the Kelly formula, setting fixed profit points for retracement stop-profits, etc., to reduce the possibility of potential profit givebacks at the end of a trend.
3. For gap openings, a maximum stop-loss limit can be set, such as a fixed amount or a fixed percentage. Once this limit is reached, the position is immediately stopped out regardless of where the dynamic stop-loss price is.

#### Summary
The ATR trailing stop strategy can dynamically adjust the stop-loss position based on the magnitude of price fluctuations and can achieve good results in trending markets. However, this strategy also has risks such as inability to cope with choppy markets, excessive stop-loss frequency, and difficulty in avoiding gap openings. To address these shortcomings, the strategy can be optimized and improved in terms of trend judgment, take-profit strategies, and maximum stop-loss limits. With these adjustments, the strategy's adaptability and profitability can hopefully be enhanced.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-04-01 00:00:00
end: 2024-04-30 23:59:59
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Long TAP", overlay=true)

// Constants
keyValueDefault = 3.0
keyValueStep = 0.5
atrPeriodDefault = 10

// Inputs
keyValue = input.float(keyValueDefault, title="Key Value")
atrPeriod = input.int(atrPeriodDefault, title="ATR Period")

// Calculations
xATR = ta.atr(atrPeriod)
nLoss = keyValue * xATR

// Trailing Stop Calculation
var float xATRTrailingStop = 0.0
xATRTrailingStop := ta.highest(math.max(nz(xATRTrailingStop[1], 0), close - nLoss), 1)
xATRTrailingStop := ta.lowest(math.min(nz(xATRTrailingStop, 0), close + nLoss), 1)

// Position Calculation
var int pos = 0
pos := nz(pos[1], 0)
if (close[1] < nz(xATRTrailingStop, 0) and close > nz(xATRTrailingStop, 0))
    pos := 1
else if (close[1] > nz(xATRTrailingStop, 0) and close < nz(xATRTrailingStop, 0))
    pos := -1

// Plotting Trailing Stop
var color xcolor = na
if (pos == -1)
    xcolor := color.red
else if (pos == 1)
    xcolor := color.green
plot(xATRTrailingStop, color=xcolor, title="Trailing Stop")

// Buy/Sell Signals
buySignal = ta.crossover(close, xATRTrailingStop)
sellSignal = ta.crossunder(close, xATRTrailingStop)

// Strategy
if (buySignal)
    strategy.entry("Long", strategy.long)
    label.new(bar_index, xATRTrailingStop, text="Buy Signal", color=color.green, style=label.style_label_up, yloc=yloc.belowbar)
if (sellSignal)
    strategy.entry("Short", strategy.short)
    label.new(bar_index, xATRTrailingStop, text="Sell Signal", color=color.red, style=label.style_label_down, yloc=yloc.abovebar)

// Alerts
alertcondition(buySignal, title='UT BOT Buy', message='UT BOT Buy')
alertcondition(sellSignal, title='UT BOT Sell', message='UT BOT Sell')

```

> Detail

https://www.fmz.com/strategy/452364

> Last Modified

2024-05-24 18:12:01
