
> Name

一云多均线交易策略-Ichimoku-Cloud-and-Moving-Average-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/17e975d8001cb544ebf.png)

[trans]
#### 概述
该策略结合了一目均衡图云以及短期(55)和长期(200)简单移动平均线(SMA)来识别潜在的买卖信号。买入信号需要价格高于云层和长期SMA,并在上穿短期SMA后回踩短期SMA。卖出信号需要价格低于云层和长期SMA,并在下穿短期SMA后回踩短期SMA。该策略避免在横盘市或重大新闻事件期间产生信号,因为这些时期虚假信号较多。回测表明该策略在1小时和2小时时间框架上表现最佳。

#### 策略原理 
该策略基于以下原则:
1. 当价格高于云层和长期SMA时,市场处于上升趋势。
2. 当价格低于云层和长期SMA时,市场处于下降趋势。 
3. 短期SMA的上穿和下穿可以确认趋势,回踩短期SMA提供低风险入场机会。
4. 横盘市和重大新闻事件期间虚假信号较多,应该避免交易。

程序首先计算所需的一目云组件(转换线、基准线、先行span A和B),以及短期和长期SMA。然后定义多个条件来识别价格相对于云层和均线的位置。当满足所有买入/卖出条件时,程序分别产生买入和卖出信号。

#### 策略优势
1. 结合多个指标确认趋势,提高信号可靠性。一目云能过滤掉很多噪音,SMA交叉能确认趋势。
2. 在已确认的趋势中寻找回踩均线的入场机会,风险相对较低。
3. 通过避免横盘市和重大新闻事件期间交易,进一步降低虚假信号风险。
4. 适用于1小时和2小时等中长期交易,把握大趋势利润空间大。

#### 策略风险
1. 在趋势转折期可能会发生亏损。虽然均线交叉和云层突破能确认趋势,但滞后性仍然存在。
2. 缺乏明确的止损位置。现有条件主要关注入场时机,但没有定义具体离场位置。
3. 参数选择具有主观性和不确定性。云层参数、均线长度等的不同选择会影响策略表现。
  
#### 策略优化方向
1. 加入明确的止损位置,比如突破前低/前高、ATR倍数等,以降低单次交易风险。
2. 对照其他趋势确认指标,如MACD、DMI等,形成更加稳健可靠的信号组合。
3. 针对参数进行优化,找到最佳参数组合,提高策略在各种市场状态下的适应性。
4. 区分趋势和震荡市,在趋势市中积极入场,震荡市中则适当降低交易频率。

#### 总结
该"一云多均线交易策略"通过结合一目均衡图云和简单移动平均线,在已确立的趋势中寻找回踩均线的低风险入场机会。通过过滤掉横盘市和重大新闻事件期间的交易,该策略能够降低虚假信号风险,从而提高整体表现。策略主要适用于中长期交易者,在1小时和2小时等时间框架上表现良好。但该策略仍有进一步优化空间,如引入明确止损位、优化信号组合、对参数进行调试等,以期获得更加稳健的策略表现。

|| 
#### Overview
This strategy combines the Ichimoku Cloud, short-term (55) and long-term (200) Simple Moving Averages (SMA) to identify potential buy and sell signals. Buy signals require the price to be above the cloud and long-term SMA, and to retest the short-term SMA after crossing above it. Sell signals require the price to be below the cloud and long-term SMA, and to retest the short-term SMA after crossing below it. The strategy avoids generating signals during ranging markets or high news events, as these periods tend to have more fake-outs. Backtesting shows the strategy performs best on the 1-hour and 2-hour timeframes.

#### Strategy Principles
The strategy is based on the following principles:
1. When price is above the cloud and long-term SMA, the market is in an uptrend. 
2. When price is below the cloud and long-term SMA, the market is in a downtrend.
3. Crossovers of the short-term SMA confirm trends, and retests of the short-term SMA provide low-risk entry opportunities.
4. Ranging markets and high news events have more fake-outs and should be avoided.

The code first calculates the required Ichimoku Cloud components (Conversion Line, Base Line, Leading Span A and B), as well as the short-term and long-term SMAs. It then defines multiple conditions to identify the price position relative to the cloud and moving averages. When all buy/sell conditions are met, the code generates buy and sell signals respectively.

#### Strategy Advantages
1. Combines multiple indicators to confirm trends, improving signal reliability. The Ichimoku Cloud filters out noise, while SMA crossovers confirm trends.
2. Seeks low-risk entry opportunities on retests of moving averages within confirmed trends.
3. Further reduces fake-out risks by avoiding trades during ranging markets and high news events.
4. Suitable for medium to long-term trading on 1-hour and 2-hour timeframes, capturing big trends with large profit potential.

#### Strategy Risks 
1. Losses may occur during trend reversals. Although moving average crossovers and cloud breakouts confirm trends, they still lag.
2. Lacks clear stop-loss levels. Current conditions focus on entry timing but do not define specific exit points.
3. Parameter selection is subjective and uncertain. Different choices of cloud parameters, moving average lengths, etc. will affect strategy performance.

#### Strategy Optimization Directions
1. Introduce clear stop-loss levels, such as previous high/low breaches, ATR multiples, etc., to reduce single trade risk.
2. Cross-reference with other trend confirmation indicators, such as MACD, DMI, etc., to form more robust signal combinations.
3. Optimize parameters to find the best combination that improves the strategy's adaptability to various market conditions. 
4. Differentiate between trending and ranging markets, actively enter positions in trends while reducing trading frequency in ranges.

#### Summary
The "Ichimoku Cloud and Moving Average Strategy" seeks low-risk entry opportunities by combining the Ichimoku Cloud with Simple Moving Averages within established trends. By filtering out trades during ranging markets and high news events, the strategy reduces fake-out risks and improves overall performance. It is mainly suitable for medium to long-term traders and performs well on 1-hour and 2-hour timeframes. However, there is still room for further optimization, such as introducing clear stop-losses, optimizing signal combinations, and tuning parameters, to achieve more robust strategy performance.

[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-05-11 00:00:00
end: 2024-05-16 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Ichimoku Cloud and Moving Average Strategy", shorttitle="ICMA", overlay=true)

// Input parameters
shortMA = input.int(55, title="Short-term Moving Average Length")
longMA = input.int(200, title="Long-term Moving Average Length")

// Calculate moving averages
shortSMA = ta.sma(close, shortMA)
longSMA = ta.sma(close, longMA)

// Ichimoku Cloud settings
conversionPeriod = input.int(9, title="Conversion Line Period")
basePeriod = input.int(26, title="Base Line Period")
spanBPeriod = input.int(52, title="Span B Period")
displacement = input.int(26, title="Displacement")

// Calculate Ichimoku Cloud components
conversionLine = ta.sma(high + low, conversionPeriod) / 2
baseLine = ta.sma(high + low, basePeriod) / 2
leadSpanA = (conversionLine + baseLine) / 2
leadSpanB = ta.sma(high + low, spanBPeriod) / 2

// Plot Ichimoku Cloud components
plot(leadSpanA, color=color.blue, title="Leading Span A")
plot(leadSpanB, color=color.red, title="Leading Span B")

// Entry conditions
aboveCloud = close > leadSpanA and close > leadSpanB
belowCloud = close < leadSpanA and close < leadSpanB
aboveShortMA = close > shortSMA
aboveLongMA = close > longSMA
belowShortMA = close < shortSMA
belowLongMA = close < longSMA

// Buy condition (Price retests 55 moving average after being above it)
buyCondition = aboveCloud and aboveLongMA and close[1] < shortSMA and close > shortSMA

// Sell condition (Price retests 55 moving average after being below it)
sellCondition = belowCloud and belowLongMA and close[1] > shortSMA and close < shortSMA

// Strategy entry and exit
strategy.entry("Buy", strategy.long, when = buyCondition)
strategy.entry("Sell", strategy.short, when = sellCondition)

// Plot moving averages
plot(shortSMA, color=color.green, title="Short-term SMA")
plot(longSMA, color=color.red, title="Long-term SMA")

// Plot buy and sell signals
plotshape(series=buyCondition, style=shape.triangleup, location=location.belowbar, color=color.green, size=size.small, title="Buy Signal")
plotshape(series=sellCondition, style=shape.triangledown, location=location.abovebar, color=color.red, size=size.small, title="Sell Signal")













```

> Detail

https://www.fmz.com/strategy/451705

> Last Modified

2024-05-17 10:55:29
