
> Name

基于成交量热图与实时价格的买卖策略-Buy-and-Sell-Volume-Heatmap-with-Real-Time-Price-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/14176c4b51b96da49a3.png)

[trans]
#### 概述
该策略结合了成交量热图和实时价格,通过分析一定时间内的价格和成交量分布,生成买卖信号。策略首先根据当前价格和设定的价格范围百分比,计算出若干个价格水平。然后统计每个价格水平在过去一段时间内的买卖成交量,并计算出累计的买卖成交量。根据累计的买卖成交量,确定标签的颜色。同时,策略还会绘制实时价格曲线。此外,该策略还结合了EMA、VWAP等指标,根据它们与价格、成交量的关系,产生买卖信号。当满足买入条件且之前没有产生过信号时,会产生买入信号;当满足卖出条件或连续两根阴线且之前没有产生过信号时,会产生卖出信号。

#### 策略原理
1. 根据当前价格和设定的价格范围百分比,计算出若干个价格水平。
2. 统计每个价格水平在过去一段时间内的买卖成交量,并计算出累计的买卖成交量。
3. 根据累计的买卖成交量,确定标签的颜色,并显示标签或绘制图形。
4. 绘制实时价格曲线。
5. 计算EMA、VWAP等指标。
6. 根据价格与EMA、VWAP等指标的关系以及成交量条件,判断是否满足买入条件。若满足且之前没有产生过信号,则产生买入信号。
7. 根据价格与EMA等指标的关系以及成交量条件,判断是否满足卖出条件。若满足且之前没有产生过信号,则产生卖出信号。若连续两根阴线且之前没有产生过信号,也产生卖出信号。
8. 记录当前买卖条件状态,并更新信号产生状态。

#### 优势分析
1. 结合了成交量热图和实时价格,能够直观地展示价格和成交量的分布情况,为交易决策提供参考。
2. 引入了EMA、VWAP等指标,丰富了策略的条件判断,提高了策略的可靠性。
3. 同时考虑了价格、指标和成交量等多方面因素,使得买卖信号更加全面和稳健。
4. 设置了信号产生的限制条件,避免了连续产生重复信号的情况,减少了误导性信号。

#### 风险分析
1. 策略的表现可能受到价格范围百分比、回溯期等参数设置的影响,需要根据具体情况进行调整和优化。
2. EMA、VWAP等指标本身也有一定的滞后性和局限性,在某些市场环境下可能失效。
3. 该策略主要适用于趋势性较强的市场,在震荡市可能产生较多的虚假信号。
4. 策略的风险控制措施相对简单,缺乏止损和仓位管理等风险管理手段。

#### 优化方向
1. 引入更多的技术指标和市场情绪指标,如RSI、MACD、布林带等,丰富策略的判断依据。
2. 优化买卖信号的产生条件,提高信号的准确性和可靠性。可以考虑引入多个时间框架的分析,确认趋势方向。
3. 加入止损和仓位管理等风险控制措施,设置合理的止损位和仓位规模,控制单笔交易的风险敞口。
4. 对策略进行参数优化和回测,找出最优的参数组合和市场适用范围。
5. 考虑将该策略与其他策略进行组合,发挥不同策略的优势,提高整体的稳定性和收益性。

#### 总结
该策略通过结合成交量热图、实时价格和多个技术指标,生成买卖信号,具有一定的参考价值。策略的优势在于能够直观展示价格和成交量的分布,并综合考虑多方面因素产生信号。但策略也存在一些局限性和风险,如参数设置的影响、指标的滞后性、趋势性市场的依赖等。因此,在实际应用中,需要进一步优化和完善策略,如引入更多指标、优化信号条件、加强风险控制等,以提升策略的稳健性和盈利能力。

|| 

#### Overview
This strategy combines a volume heatmap and real-time price to generate buy and sell signals by analyzing the distribution of price and volume over a certain period. The strategy first calculates several price levels based on the current price and a set price range percentage. It then counts the buy and sell volumes at each price level over a past period and calculates the cumulative buy and sell volumes. The color of the labels is determined based on the cumulative buy and sell volumes. Additionally, the strategy plots the real-time price curve. Moreover, the strategy incorporates indicators such as EMA and VWAP to generate buy and sell signals based on their relationship with price and volume. A buy signal is generated when the buy conditions are met, and no previous signal has occurred. A sell signal is generated when the sell conditions are met, or there are two consecutive red candles, and no previous signal has occurred.

#### Strategy Principle
1. Calculate several price levels based on the current price and a set price range percentage.
2. Count the buy and sell volumes at each price level over a past period and calculate the cumulative buy and sell volumes.
3. Determine the color of the labels based on the cumulative buy and sell volumes and display the labels or plot shapes.
4. Plot the real-time price curve.
5. Calculate indicators such as EMA and VWAP.
6. Determine whether the buy conditions are met based on the relationship between price and indicators like EMA and VWAP, as well as volume conditions. If met and no previous signal has occurred, generate a buy signal.
7. Determine whether the sell conditions are met based on the relationship between price and indicators like EMA, as well as volume conditions. If met and no previous signal has occurred, generate a sell signal. If there are two consecutive red candles and no previous signal has occurred, also generate a sell signal.
8. Record the current buy and sell condition states and update the signal occurrence status.

#### Advantage Analysis
1. The combination of volume heatmap and real-time price provides an intuitive display of the price and volume distribution, serving as a reference for trading decisions.
2. The incorporation of indicators such as EMA and VWAP enriches the strategy's condition judgment and improves its reliability.
3. The strategy considers multiple factors, including price, indicators, and volume, making the buy and sell signals more comprehensive and robust.
4. The strategy sets limitations on signal generation to avoid generating repeated signals continuously, reducing misleading signals.

#### Risk Analysis
1. The strategy's performance may be affected by parameter settings such as the price range percentage and lookback period, requiring adjustments and optimizations based on specific situations.
2. Indicators like EMA and VWAP have inherent lagging and limitations, which may become ineffective in certain market environments.
3. The strategy is mainly suitable for trending markets and may generate more false signals in choppy markets.
4. The risk control measures of the strategy are relatively simple, lacking risk management tools such as stop-loss and position sizing.

#### Optimization Direction
1. Introduce more technical indicators and market sentiment indicators, such as RSI, MACD, Bollinger Bands, etc., to enrich the strategy's judgment basis.
2. Optimize the conditions for generating buy and sell signals to improve signal accuracy and reliability. Consider incorporating multi-timeframe analysis to confirm trend directions.
3. Incorporate risk control measures such as stop-loss and position sizing, setting reasonable stop-loss levels and position sizes to control the risk exposure of individual trades.
4. Perform parameter optimization and backtesting on the strategy to find the optimal parameter combinations and market applicability.
5. Consider combining this strategy with other strategies to leverage the strengths of different strategies and improve overall stability and profitability.

#### Summary
This strategy generates buy and sell signals by combining a volume heatmap, real-time price, and multiple technical indicators, providing a certain reference value. The strategy's advantage lies in its ability to intuitively display the price and volume distribution and comprehensively consider multiple factors to generate signals. However, the strategy also has some limitations and risks, such as the impact of parameter settings, the lagging nature of indicators, and the reliance on trending markets. Therefore, in practical applications, further optimization and improvement of the strategy are needed, such as introducing more indicators, optimizing signal conditions, enhancing risk control, etc., to improve the strategy's robustness and profitability.
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
strategy("Buy and Sell Volume Heatmap with Real-Time Price Strategy", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=10)

// Settings for Volume Heatmap
lookbackPeriod = input.int(100, title="Lookback Period")
baseGreenColor = input.color(color.green, title="Buy Volume Color")
baseRedColor = input.color(color.red, title="Sell Volume Color")
priceLevels = input.int(10, title="Number of Price Levels")
priceRangePct = input.float(0.01, title="Price Range Percentage")
labelSize = input.string("small", title="Label Size", options=["tiny", "small", "normal", "large"])
showLabels = input.bool(true, title="Show Volume Labels")

// Initialize arrays to store price levels, buy volumes, and sell volumes
var float[] priceLevelsArr = array.new_float(priceLevels)
var float[] buyVolumes = array.new_float(priceLevels)
var float[] sellVolumes = array.new_float(priceLevels)

// Calculate price levels around the current price
for i = 0 to priceLevels - 1
    priceLevel = close * (1 + (i - priceLevels / 2) * priceRangePct)  // Adjust multiplier for desired spacing
    array.set(priceLevelsArr, i, priceLevel)

// Calculate buy and sell volumes for each price level
for i = 0 to priceLevels - 1
    level = array.get(priceLevelsArr, i)
    buyVol = 0.0
    sellVol = 0.0
    for j = 1 to lookbackPeriod
        if close[j] > open[j]
            if close[j] >= level and low[j] <= level
                buyVol := buyVol + volume[j]
        else
            if close[j] <= level and high[j] >= level
                sellVol := sellVol + volume[j]
    array.set(buyVolumes, i, buyVol)
    array.set(sellVolumes, i, sellVol)

// Determine the maximum volumes for normalization
maxBuyVolume = array.max(buyVolumes)
maxSellVolume = array.max(sellVolumes)

// Initialize cumulative buy and sell volumes for the current bar
cumulativeBuyVol = 0.0
cumulativeSellVol = 0.0

// Calculate colors based on the volumes and accumulate volumes for the current bar
for i = 0 to priceLevels - 1
    buyVol = array.get(buyVolumes, i)
    sellVol = array.get(sellVolumes, i)
    cumulativeBuyVol := cumulativeBuyVol + buyVol
    cumulativeSellVol := cumulativeSellVol + sellVol

// Determine the label color based on which volume is higher
labelColor = cumulativeBuyVol > cumulativeSellVol ? baseGreenColor : baseRedColor

// Initialize variables for plotshape
var float shapePosition = na
var color shapeColor = na

if cumulativeBuyVol > 0 or cumulativeSellVol > 0
    if showLabels
        labelText = "Buy: " + str.tostring(cumulativeBuyVol) + "\nSell: " + str.tostring(cumulativeSellVol)
        label.new(x=bar_index, y=high + (high - low) * 0.02, text=labelText, color=color.new(labelColor, 0), textcolor=color.white, style=label.style_label_down, size=labelSize)
    else
        shapePosition := high + (high - low) * 0.02
        shapeColor := labelColor

// Plot the shape outside the local scope
plotshape(series=showLabels ? na : shapePosition, location=location.absolute, style=shape.circle, size=size.tiny, color=shapeColor)

// Plot the real-time price on the chart
plot(close, title="Real-Time Price", color=color.blue, linewidth=2, style=plot.style_line)

// Mpullback Indicator Settings
a = ta.ema(close, 9)
b = ta.ema(close, 20)
e = ta.vwap(close)
volume_ma = ta.sma(volume, 20)

// Calculate conditions for buy and sell signals
buy_condition = close > a and close > e and volume > volume_ma and close > open and low > a and low > e // Ensure close, low are higher than open, EMA, and VWAP
sell_condition = close < a and close < b and close < e and volume > volume_ma

// Store the previous buy and sell conditions
var bool prev_buy_condition = na
var bool prev_sell_condition = na

// Track if a buy or sell signal has occurred
var bool signal_occurred = false

// Generate buy and sell signals based on conditions
buy_signal = buy_condition and not prev_buy_condition and not signal_occurred
sell_signal = sell_condition and not prev_sell_condition and not signal_occurred

// Determine bearish condition (close lower than the bottom 30% of the candle's range)
bearish = close < low + (high - low) * 0.3

// Add sell signal when there are two consecutive red candles and no signal has occurred
two_consecutive_red_candles = close[1] < open[1] and close < open
sell_signal := sell_signal or (two_consecutive_red_candles and not signal_occurred)

// Remember the current conditions for the next bar
prev_buy_condition := buy_condition
prev_sell_condition := sell_condition

// Update signal occurred status
signal_occurred := buy_signal or sell_signal

// Plot buy and sell signals
plotshape(buy_signal, title="Buy", style=shape.labelup, location=location.belowbar, color=color.green, text="Buy", textcolor=color.white)
plotshape(sell_signal, title="Sell", style=shape.labeldown, location=location.abovebar, color=color.red, text="Sell", textcolor=color.white)

// Strategy entry and exit
if buy_signal
    strategy.entry("Buy", strategy.long)

if sell_signal
    strategy.entry("Sell", strategy.short)

```

> Detail

https://www.fmz.com/strategy/452354

> Last Modified

2024-05-24 17:16:58
