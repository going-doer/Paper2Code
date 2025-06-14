
> Name

N-Bars-Breakout-Strategy-N-Bars-突破策略

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/a474deee97b1563108.png)

[trans]
#### 概述
N Bars 突破策略是一种基于价格突破的量化交易策略。该策略的主要思路是,当收盘价突破过去N个交易日的最高价时开多仓,当收盘价跌破过去N个交易日的最低价时平多仓。该策略通过比较当前价格与过去N个交易日的最高最低价,捕捉强势突破行情,达到趋势跟踪的效果。

#### 策略原理
1. 计算过去N个交易日的最高价(highest)和最低价(lowest)。
2. 如果当前收盘价高于highest,则开多仓(long)。
3. 如果当前收盘价低于lowest,则平多仓(short)。
4. 可以选择使用收盘价(close)或者高低价(high/low)作为信号源(source)。
5. 根据信号源的不同,使用ta.highest和ta.lowest计算最高价和最低价。
6. 使用ta.crossover和ta.crossunder判断价格突破情况。

#### 策略优势
1. 逻辑简单清晰,易于实现和优化。
2. 可以有效捕捉强势突破行情,趋势跟踪能力强。
3. 参数调优空间大,可以根据不同品种和周期进行优化。
4. 适用性广,对多数品种和周期都有不错的表现。
5. 可以灵活选择信号源,提高策略适应性。

#### 策略风险
1. 对震荡行情和小幅波动行情表现较差,频繁开平仓导致较高的交易成本。
2. 参数选择不当可能导致过拟合风险。
3. 趋势转折时可能出现较大回撤。
4. 单一信号源可能面临信号失真的风险。

#### 策略优化方向  
1. 增加趋势过滤条件,如ma趋势方向、adx等,减少震荡行情下的交易。
2. 优化参数选择,如N值、信号源等,提高策略稳定性和盈利能力。
3. 增加止损和移动止损逻辑,控制单笔交易风险。
4. 结合多个信号源,提高信号可靠性,如同时考虑收盘价和高低价突破。
5. 根据不同品种和周期,分别优化参数和逻辑。

#### 总结
N Bars 突破策略是一个简单实用的量化交易策略,通过捕捉价格突破行情,实现了较好的趋势跟踪效果。该策略逻辑清晰,优化空间大,适用性广,是一个值得进一步研究和优化的量化策略。通过合理的参数优化和逻辑改进,可以进一步提高该策略的稳定性和盈利能力,更好地适应不同市场环境。

|| 

#### Overview
The N Bars Breakout Strategy is a quantitative trading strategy based on price breakouts. The main idea of this strategy is to open a long position when the closing price breaks above the highest high of the past N bars, and close the long position when the closing price breaks below the lowest low of the past N bars. By comparing the current price with the highest and lowest prices of the past N bars, this strategy aims to capture strong breakout moves and achieve the effect of trend following.

#### Strategy Principle
1. Calculate the highest high (highest) and lowest low (lowest) of the past N bars.
2. If the current closing price is higher than the highest, open a long position (long).
3. If the current closing price is lower than the lowest, close the long position (short).
4. You can choose to use the closing price (close) or high/low price (high/low) as the signal source.
5. Depending on the signal source, use ta.highest and ta.lowest to calculate the highest and lowest prices.
6. Use ta.crossover and ta.crossunder to determine price breakouts.

#### Strategy Advantages
1. Simple and clear logic, easy to implement and optimize.
2. Can effectively capture strong breakout moves, with strong trend-following capability.
3. Large room for parameter optimization, can be optimized for different instruments and timeframes.
4. Wide applicability, performs well for most instruments and timeframes.
5. Flexible choice of signal source, improving strategy adaptability.

#### Strategy Risks
1. Performs poorly in choppy and small-fluctuation markets, with frequent opening and closing of positions leading to high transaction costs.
2. Improper parameter selection may lead to overfitting risk.
3. May experience large drawdowns during trend reversals.
4. Single signal source may face the risk of signal distortion.

#### Strategy Optimization Directions
1. Add trend filtering conditions, such as MA trend direction, ADX, etc., to reduce trading in choppy markets.
2. Optimize parameter selection, such as N value, signal source, etc., to improve strategy stability and profitability.
3. Add stop-loss and trailing stop-loss logic to control single-trade risk.
4. Combine multiple signal sources to improve signal reliability, such as considering both closing price and high/low price breakouts.
5. Optimize parameters and logic separately for different instruments and timeframes.

#### Summary
The N Bars Breakout Strategy is a simple and practical quantitative trading strategy that achieves good trend-following effects by capturing price breakouts. The strategy has clear logic, large optimization space, and wide applicability, making it a quantitative strategy worth further research and optimization. Through reasonable parameter optimization and logic improvement, the stability and profitability of this strategy can be further enhanced to better adapt to different market environments.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_int_1|5|N Bars|
|v_input_string_1|Close|Source|


> Source (PineScript)

``` pinescript
/*backtest
start: 2023-04-06 00:00:00
end: 2024-04-11 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Breakout", overlay=true, precision=6, pyramiding=0, initial_capital=1000, default_qty_type=strategy.percent_of_equity, default_qty_value=25.0, commission_value=0.05)

n = input.int(5, "N Bars", minval=1)
src_input = input.string("Close", "Source", ["Close", "High/Low"])

bull_src = switch src_input
	"Close" => close
	"High/Low" => high
	=>
		runtime.error("Invalid source input")
		na

bear_src = switch src_input
	"Close" => close
	"High/Low" => low
	=>
		runtime.error("Invalid source input")
		na

highest = ta.highest(bull_src[1], n)
lowest = ta.lowest(bear_src[1], n)

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------
// Plots
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------

bool long = ta.crossover(bull_src, highest)
bool short = ta.crossunder(bear_src, lowest)

//Plots
lowest_plot  = plot(lowest,  color=color.red, title="Lowest")
highest_plot  = plot(highest,  color=color.green, title="Highest")
bull_src_plot = plot(bull_src, color=color.blue, title="Bull")
bear_src_plot = plot(bear_src, color=color.orange, title="Bear")

// this message is an alert that can be sent to a webhook, which allows for simple automation if you have a server that listens to alerts and trades programmatically.
enter_long_alert = '{"side": "Long", "order": "Enter", "price": ' + str.tostring(open) + ', "timestamp": ' + str.tostring(timenow) + '}'
exit_long_alert = '{"side": "Long", "order": "Exit", "price": ' + str.tostring(open) + ', "timestamp": ' + str.tostring(timenow) + '}'

if long
    strategy.entry(id="Long", direction=strategy.long, limit=open, alert_message=enter_long_alert)

if short
    strategy.close(id="Long", comment="Close Long", alert_message=exit_long_alert)
```

> Detail

https://www.fmz.com/strategy/448067

> Last Modified

2024-04-12 16:57:15
