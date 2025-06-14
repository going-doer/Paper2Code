
> Name

横线突破捕捉趋势策略Trend-Capturing-Strategy-with-Horizontal-Line-Breakout

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/15d57b211c4cef934de.png)

[trans]
#### 概述
该策略使用水平线作为支撑位和阻力位,当价格突破水平线时会产生交易信号。主要思路是:首先根据某些条件画出水平线,当价格向上突破水平线时做多,当价格向下突破水平线时平仓。同时水平线的产生也有相应的条件,如前一根K线的最低价大于当前收盘价。

#### 策略原理
1. 根据前一根K线的最低价大于当前收盘价的条件,在前一根K线的最低价位置画一条长度为20的水平线,并且始终只保留最新的10条线。
2. 如果当前价格向上突破了最新的一条水平线,则平掉之前所有的多单,重新做多。
3. 在画新的水平线时,如果此时有持仓,则先平仓,再做多。

#### 策略优势
1. 策略逻辑简单清晰,容易理解和实现。
2. 通过水平线支撑阻力位的突破来产生信号,能较好地捕捉趋势。
3. 新的水平线产生时会先平掉之前的仓位,然后再开新仓,这样可以降低之前开仓可能带来的风险。
4. 可以通过调整水平线的长度和数量来优化策略。

#### 策略风险
1. 对于震荡行情,频繁的水平线突破可能会导致过度交易,从而产生较大的滑点和手续费。
2. 策略对于水平线的定义比较简单,缺乏其他指标的验证,可能会产生一些错误信号。
3. 只做多不做空,无法充分利用下跌行情。
   
#### 策略优化方向 
1. 可以结合其他指标来确认水平线的有效性,比如成交量的变化等,以减少错误信号。
2. 对于震荡行情,可以通过增加水平线突破的幅度来减少交易频率。
3. 加入做空机制,在下跌趋势中也能获利。
4. 可以考虑动态调整水平线的长度和数量,以适应不同的行情。

#### 总结
该策略以水平线作为重要的支撑阻力位,通过突破水平线产生交易信号。优点是逻辑简单,容易实现,能较好地捕捉趋势。但是缺点是可能会过度交易,产生错误信号,并且只能做多不能做空。后续可以从结合其他指标、减少交易频率、加入做空机制和动态调整参数等方面进行优化和改进。

|| 

#### Overview
This strategy uses horizontal lines as support and resistance levels, and generates trading signals when the price breaks through these lines. The main idea is: first, draw horizontal lines based on certain conditions, go long when the price breaks above the line, and close positions when the price breaks below the line. At the same time, there are corresponding conditions for generating horizontal lines, such as the low price of the previous candle being greater than the current closing price.

#### Strategy Principle
1. Based on the condition that the low price of the previous candle is greater than the current closing price, draw a horizontal line with a length of 20 at the low price of the previous candle, and always keep only the latest 10 lines.
2. If the current price breaks above the latest horizontal line, close all previous long positions and open new long positions.
3. When drawing a new horizontal line, if there is a position, close it first and then open a new long position.

#### Strategy Advantages
1. The strategy logic is simple and clear, easy to understand and implement.
2. By using the breakthrough of horizontal line support and resistance levels to generate signals, it can capture trends well.
3. When a new horizontal line is generated, it will first close the previous positions and then open new ones, which can reduce the risk that the previous positions may bring.
4. The strategy can be optimized by adjusting the length and number of horizontal lines.

#### Strategy Risks
1. For oscillating markets, frequent horizontal line breakthroughs may lead to overtrading, resulting in large slippage and transaction fees.
2. The strategy's definition of horizontal lines is relatively simple and lacks verification from other indicators, which may generate some false signals.
3. It only goes long and does not go short, and cannot fully utilize downward trends.

#### Strategy Optimization Directions
1. Other indicators can be combined to confirm the validity of horizontal lines, such as changes in trading volume, to reduce false signals.
2. For oscillating markets, the frequency of trading can be reduced by increasing the magnitude of horizontal line breakthroughs.
3. Add a short-selling mechanism to profit from downward trends.
4. Consider dynamically adjusting the length and number of horizontal lines to adapt to different market conditions.

#### Summary
This strategy uses horizontal lines as important support and resistance levels, and generates trading signals through line breakthroughs. The advantage is that the logic is simple and easy to implement, and it can capture trends well. However, the disadvantage is that it may overtrade, generate false signals, and can only go long but not short. In the future, it can be optimized and improved by combining other indicators, reducing trading frequency, adding short-selling mechanisms, and dynamically adjusting parameters.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_int_1|20|(?Yatay Çizgi Ayarları)Çizgi uzunluğu?|
|v_input_int_2|10|Son Kaç Çizgi?|


> Source (PineScript)

``` pinescript
/*backtest
start: 2023-04-20 00:00:00
end: 2024-04-25 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © Traderxprox

//@version=5
strategy("Alarm Trader_ALL", overlay=true)

// Yatay çizgi oluşum
yatayc = low[1] > close[0]

if yatayc
    if strategy.opentrades > 0
        strategy.close("AL", comment = "Fiyat:" + str.tostring(low[1], "#.##") + "\n" + timeframe.period +"\n Yatay Direnç Oluştu")
    else
        strategy.entry("AL", strategy.long, comment = "Fiyat:" + str.tostring(low[1], "#.##") + "\n" + timeframe.period +"\n Yatay Direnç Oluştu")


//YATAY ÇİZGİ
int cizgilen = input.int(20, "Çizgi uzunluğu?", group = "Yatay Çizgi Ayarları")
var array<line> lines = array.new<line>()
int numberOfLines = input.int(10, "Son Kaç Çizgi?", 0, group = "Yatay Çizgi Ayarları")
kural22 = low[1] > close[0]
// if kural22
//     newLine = line.new(bar_index-2, low[1], bar_index+cizgilen, low[1] ,color=color.red, width=1, style=line.style_solid)
//     // Push the `newLine` into the `lines` array.
//     lines.push(newLine)
//     // Delete the oldest line when the size of the array exceeds the specified `numberOfLines`.
//     if array.size(lines) > numberOfLines
//         line.delete(lines.shift())
    
// Alarm kırılım için koşul

var float lastLinePrice = na
if not na(close) and array.size(lines) > 0 
    lastLinePrice := line.get_price(array.get(lines, array.size(lines) - 1), bar_index)
if open < lastLinePrice and close > lastLinePrice
    if strategy.opentrades > 0
        strategy.close("AL", comment = "Fiyat:" + str.tostring(lastLinePrice, "#.##") + "\n" + timeframe.period +" \n Yatay çizgi yukarı kırılımı")
    else
        strategy.entry("AL", strategy.long, comment = "Fiyat:" + str.tostring(lastLinePrice, "#.##") + "\n" + timeframe.period +" \n Yatay çizgi yukarı kırılımı")


```

> Detail

https://www.fmz.com/strategy/449518

> Last Modified

2024-04-26 15:22:06
