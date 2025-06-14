
> Name

波林格带突破与波动率过滤策略Bollinger-Bands-Breakout-with-Volatility-Filter-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/13dbe9dae210a699f57.png)
[trans]

## 策略概述

波林格带突破与波动率过滤策略是一个基于波林格带指标的交易策略。它利用波林格带来判断价格相对于移动平均线的位置和波动性,从而决定开仓和平仓。该策略的一个独特之处在于它采用了波动率过滤器,通过检测连续K线的涨跌幅来避免在市场波动较大时入场交易。此外,该策略还设置了止盈和止损条件,以保护利润和控制风险。

## 策略原理

该策略的核心是计算波林格带指标。波林格带由三条线组成:中轨是简单移动平均线,上轨和下轨分别在中轨的基础上加减一定的标准差。标准差的大小由参数mult控制。

策略的开仓条件基于收盘价相对于波林格带的位置。如果交易方向设置为做多(tradeDirection>=0),且收盘价跌破下轨一定比例(lower_breakout_pct),则开多仓;如果交易方向设置为做空(tradeDirection<=0),且收盘价突破上轨一定比例(upper_breakout_pct),则开空仓。这里的突破比例参数允许价格略微突破波林格带才开仓,以确认趋势。  

另一方面,如果连续两根K线的涨跌幅都超过了预设的波动率阈值(Volatility),则判断当前市场波动较大,策略不会开新仓。这个波动率过滤器可以在一定程度上避免剧烈波动的市场环境。

平仓方面,如果多头仓位的收盘价触及上轨附近(upper-area*long_win_pct),或空头仓位的收盘价触及下轨附近(lower+area*short_win_pct),策略就会平掉相应仓位以获利了结。此外,如果持仓的浮动亏损超过了预设的最大回撤比例(max_drawdown_percent),策略也会平仓止损。

## 策略优势

1. 波林格带是一个成熟且广泛使用的技术指标,它融合了移动平均线和价格波动性的信息。利用波林格带制定交易策略,可以捕捉到趋势和波动的变化。

2. 该策略同时包含了开多和开空的逻辑,可以在多空双向市场中灵活把握机会。波林格带突破点的设置让策略的入场点更有确认性。

3. 波动率过滤器避免了在剧烈波动的市场下开仓,一定程度上降低了频繁交易和杠杆方面的风险。

4. 策略采用了止盈和止损机制,可以主动控制仓位,在价格回撤至关键位置时平仓。这有利于保护利润,控制回撤。

## 策略风险

1. 波林格带本质上是一个落后指标,对市场的反应存在一定的滞后性。在趋势转折或走势变化的关键时刻,策略可能错失最佳的入场时机。

2. 策略的参数设置对于不同的市场状况并不一定都适用。比如波动率过滤器的阈值设定,在趋势型和震荡型行情中可能需要有所区别。固定参数可能导致策略在某些行情中无法开仓或者开仓过于频繁。  

3. 虽然有止损措施,但是当市场出现跳空缺口时,策略可能无法按预设的价位成交,导致更大的损失。

4. 策略在开仓后并没有设置移动止损或者跟踪止损,这可能导致部分利润回吐。

## 优化方向

1. 可以考虑引入更多技术指标或者市场状态判断,比如 ATR、趋势指标、波动率指标等,作为策略的过滤条件,提高开仓的质量和时机把握。

2. 对于波动率过滤器,可以尝试采用动态的阈值,根据不同品种或者不同时间周期自适应调整,提高过滤效果。

3. 在止损止盈方面,可以引入移动止损或者跟踪止盈的机制,让策略在趋势延续时持仓,而不是过早平仓。同时可以考虑设置不同的止盈止损比例,优化风险收益比。

4. 可以进一步优化仓位管理,根据趋势强度、波动率、风险度等指标动态调整开仓比例,控制回撤。此外,还可以通过加仓、减仓等操作,更好地利用资金。

## 总结

波林格带突破与波动率过滤策略利用了波林格带对价格位置和波动性的刻画,构建了一个双向交易策略。该策略的独特之处在于波动率过滤器避免了剧烈波动市场的交易,同时设置了较为简单的止盈止损条件。整体来看,该策略较为完备地包含了开平仓逻辑和风险控制,但是在应对市场变化、参数适用性、止损效果等方面还有进一步优化的空间。如果能够引入更多技术指标、动态参数和仓位管理优化,该策略的稳健性和收益性或许能够得到提高。

|| 

## Strategy Overview

The Bollinger Bands Breakout with Volatility Filter Strategy is a trading strategy based on the Bollinger Bands indicator. It utilizes Bollinger Bands to determine the position and volatility of price relative to the moving average, thus deciding on entry and exit points. A unique aspect of this strategy is that it employs a volatility filter, which avoids entering trades during high market volatility by detecting the rate of change of consecutive candlesticks. Additionally, the strategy sets conditions for taking profits and stopping losses to protect profits and control risks.

## Strategy Principles

At the core of the strategy is the calculation of the Bollinger Bands indicator. Bollinger Bands consist of three lines: the middle line is a simple moving average, while the upper and lower bands are set at a certain standard deviation above and below the middle line, respectively. The size of the standard deviation is controlled by the parameter mult.

The entry conditions of the strategy are based on the position of the closing price relative to the Bollinger Bands. If the trade direction is set to long (tradeDirection>=0) and the closing price breaks below the lower band by a certain percentage (lower_breakout_pct), a long position is opened; if the trade direction is set to short (tradeDirection<=0) and the closing price breaks above the upper band by a certain percentage (upper_breakout_pct), a short position is opened. The breakout percentage parameters allow the price to slightly break through the Bollinger Bands before entering a position to confirm the trend.

On the other hand, if the rate of change of two consecutive candlesticks both exceed the preset volatility threshold (Volatility), the current market volatility is considered high, and the strategy will not open new positions. This volatility filter can avoid trading in extremely volatile market conditions to a certain extent.

In terms of exiting positions, if the closing price of a long position reaches near the upper band (upper-area*long_win_pct), or the closing price of a short position reaches near the lower band (lower+area*short_win_pct), the strategy will close the corresponding position to take profits. Additionally, if the unrealized loss of a position exceeds the preset maximum drawdown percentage (max_drawdown_percent), the strategy will also close the position to stop losses.

## Strategy Advantages

1. Bollinger Bands are a mature and widely used technical indicator that incorporates information about moving averages and price volatility. Utilizing Bollinger Bands to formulate trading strategies can capture changes in trends and volatility.

2. The strategy includes both long and short entry logic, allowing flexible capture of opportunities in both bullish and bearish markets. The setting of Bollinger Band breakout points makes the entry points of the strategy more confirmatory.

3. The volatility filter avoids opening positions in extremely volatile markets, reducing risks associated with frequent trading and leverage to a certain extent.

4. The strategy employs take-profit and stop-loss mechanisms to actively manage positions and close them when prices retrace to key levels. This helps protect profits and control drawdowns.

## Strategy Risks

1. Bollinger Bands are essentially a lagging indicator and have a certain delay in reacting to the market. At critical moments of trend reversal or changes in market conditions, the strategy may miss the best entry timing.

2. The parameter settings of the strategy may not be universally applicable to different market conditions. For example, the threshold setting of the volatility filter may need to be differentiated in trending and oscillating markets. Fixed parameters may cause the strategy to be unable to open positions or open too frequently in certain market conditions.

3. Although there are stop-loss measures, when the market gaps, the strategy may not be able to execute at the preset price, leading to greater losses.

4. The strategy does not set trailing stop-losses or breakeven stops after opening positions, which may lead to some profit givebacks.

## Optimization Directions

1. Consider introducing more technical indicators or market state judgments, such as ATR, trend indicators, volatility indicators, etc., as filtering conditions for the strategy to improve the quality and timing of entries.

2. For the volatility filter, try adopting dynamic thresholds that adapt to different instruments or time frames to improve filtering effectiveness.

3. In terms of stop-loss and take-profit, introduce trailing stop or breakeven stop mechanisms to allow the strategy to hold positions when the trend continues, rather than closing positions prematurely. At the same time, consider setting different stop-loss and take-profit ratios to optimize the risk-reward ratio.

4. Further optimize position management by dynamically adjusting the entry size based on trend strength, volatility, risk level, and other indicators to control drawdowns. In addition, better utilize capital through operations such as adding and reducing positions.

## Summary

The Bollinger Bands Breakout with Volatility Filter Strategy leverages the characterization of price position and volatility by Bollinger Bands to construct a two-way trading strategy. The unique aspect of this strategy is the volatility filter that avoids trading in extremely volatile markets, while also setting relatively simple take-profit and stop-loss conditions. Overall, the strategy includes fairly comprehensive entry and exit logic and risk control, but there is room for further optimization in terms of adapting to market changes, parameter applicability, and stop-loss effectiveness. If more technical indicators, dynamic parameters, and position management optimizations can be introduced, the robustness and profitability of the strategy may be improved.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_1|5.5|Maximum Acceptable Drawdown|
|v_input_2|50|Short Entry Breakout Percentage|
|v_input_3|25|Long Entry Breakout Percentage|
|v_input_4|true|Trade Direction|
|v_input_5|0.5|Volatility|
|v_input_6|-0.15|Long Settlement Rate Near Boll Upper Limit|
|v_input_7|0.4|Short Settlement Rate Near Boll Lower Limit|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-02-29 00:00:00
end: 2024-03-07 00:00:00
period: 1m
basePeriod: 1m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("[Oppen Chow] Super BBS 1.0", default_qty_type = strategy.percent_of_equity, default_qty_value =100, initial_capital=500, commission_type=strategy.commission.percent, commission_value=0.08, pyramiding=2 )

// Input parameters
length = 20
mult = 2
max_drawdown_percent = input(5.5, "Maximum Acceptable Drawdown") / 100
upper_breakout_pct = input(50, "Short Entry Breakout Percentage") / 100
lower_breakout_pct = input(25, "Long Entry Breakout Percentage") / 100
tradeDirection = input(1, title="Trade Direction")
Volatility = input(0.5, title="Volatility") / 100
long_win_pct = input(-0.15, title = "Long Settlement Rate Near Boll Upper Limit")
short_win_pct = input(0.4, title = "Short Settlement Rate Near Boll Lower Limit")

// Bollinger Bands calculation
basis = ta.sma(close, length)
dev = mult * ta.stdev(close, length)
upper = basis + dev
lower = basis - dev
area = upper - lower

// Calculate the rate of change for two consecutive candlesticks
var float change1 = na
var float change2 = na
change1 := change2
change2 := ((close - open) / open) 

// Check for two or more consecutive candlesticks with a change rate greater than 0.5%
var bool highVolatility = false
highVolatility := change2 > Volatility

// Trading logic
var float highestPriceSinceOpen = na
var float lowestPriceSinceOpen = na
var int profitableDrawbackCount = 1


// Entry logic - In the absence of high volatility
if not highVolatility and strategy.position_size == 0
    if (tradeDirection >= 0) and (close < lower - area * lower_breakout_pct)
        strategy.entry("Long", strategy.long)
        highestPriceSinceOpen := close
        profitableDrawbackCount := 0
    if (tradeDirection <= 0) and (close > upper + area * upper_breakout_pct)
        strategy.entry("Short", strategy.short)
        lowestPriceSinceOpen := close
        profitableDrawbackCount := 0


if strategy.position_size > 0 and close > upper - area * long_win_pct
    strategy.close("Long", comment = "Take Profit")
if strategy.position_size < 0 and close < lower + area * short_win_pct
    strategy.close("Short", comment = "Take Profit")

// Stop loss logic - Based on drawdown percentage
if strategy.position_size > 0
    if (strategy.position_avg_price - close)/strategy.position_avg_price >= max_drawdown_percent
        strategy.close("Long", comment = "Drawdown Stop Loss")
else if strategy.position_size < 0
    if (close - strategy.position_avg_price)/strategy.position_avg_price >= max_drawdown_percent
        strategy.close("Short", comment = "Drawdown Stop Loss")

```

> Detail

https://www.fmz.com/strategy/444009

> Last Modified

2024-03-08 15:28:05
