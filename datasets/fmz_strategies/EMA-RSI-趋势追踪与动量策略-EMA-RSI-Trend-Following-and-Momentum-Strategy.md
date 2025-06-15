
> Name

EMA-RSI-趋势追踪与动量策略-EMA-RSI-Trend-Following-and-Momentum-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/b746e2bf9f503ec849.png)

[trans]
#### 概述
Bybit EMA RSI 趋势追踪与动量策略是一个结合了指数移动平均线(EMA)和相对强弱指数(RSI)的量化交易策略。该策略利用两条不同周期的EMA来判断市场趋势,同时使用RSI指标来确认趋势的有效性。当快速EMA上穿慢速EMA且RSI低于特定下限时,策略产生做多信号;反之,当快速EMA下穿慢速EMA且RSI高于特定上限时,策略产生做空信号。该策略还根据Bybit账户等级设置不同的手续费比例,并内置止盈止损功能,可以有效控制风险。

#### 策略原理
1. 计算快速EMA和慢速EMA,周期分别为90和300。
2. 计算RSI指标,周期为5。
3. 当快速EMA上穿慢速EMA且RSI低于45时,产生做多信号;当快速EMA下穿慢速EMA且RSI高于85时,产生做空信号。
4. 根据Bybit账户等级设置不同的手续费比例,从VIP 0的0.075%到VIP 4的0.035%不等。
5. 计算包含手续费的开仓价格。
6. 根据设定的止盈止损百分比(5%和3%)计算止盈价和止损价。
7. 在图表上绘制开仓价、止盈线和止损线。
8. 根据交易信号执行开仓操作。

#### 策略优势
1. 结合趋势追踪和动量指标,能够较好地捕捉市场趋势。
2. 内置止盈止损功能,可以有效控制风险。
3. 根据Bybit账户等级设置不同的手续费比例,适应不同用户的交易条件。
4. 在图表上绘制开仓价、止盈线和止损线,提供直观的交易信号确认。

#### 策略风险
1. EMA和RSI参数的选择可能不适用于所有市场环境,需要根据实际情况进行优化。
2. 在震荡市场中,该策略可能产生频繁的交易信号,导致高昂的交易成本。
3. 止盈止损的设置可能过于保守或激进,需要根据个人风险偏好进行调整。

#### 策略优化方向
1. 对EMA和RSI的参数进行优化,以适应不同的市场环境。可以通过回测和参数扫描来寻找最佳参数组合。
2. 引入其他技术指标,如布林带、MACD等,以提高交易信号的准确性。
3. 优化止盈止损的设置,例如采用移动止损或动态止损方法,以更好地保护利润和控制风险。
4. 考虑市场波动性和交易量等因素,对交易信号进行过滤,减少频繁交易带来的成本。

#### 总结
Bybit EMA RSI 趋势追踪与动量策略是一个结合了趋势追踪和动量指标的量化交易策略,通过EMA和RSI的配合使用,可以较好地捕捉市场趋势。该策略内置止盈止损功能和根据Bybit账户等级设置手续费的功能,可以有效控制风险并适应不同用户的交易条件。然而,该策略仍有优化空间,如参数优化、引入其他技术指标、优化止盈止损设置等。通过不断优化和改进,该策略有望在实际交易中取得更好的效果。

|| 

#### Overview
The Bybit EMA RSI Trend-Following and Momentum Strategy is a quantitative trading strategy that combines Exponential Moving Averages (EMA) and the Relative Strength Index (RSI). The strategy uses two EMAs with different periods to determine market trends and the RSI indicator to confirm the validity of the trends. When the fast EMA crosses above the slow EMA and the RSI is below a specific lower threshold, the strategy generates a long signal. Conversely, when the fast EMA crosses below the slow EMA and the RSI is above a specific upper threshold, the strategy generates a short signal. The strategy also sets different commission percentages based on the Bybit account level and includes built-in take profit and stop loss functions to effectively manage risk.

#### Strategy Principles
1. Calculate the fast EMA and slow EMA with periods of 90 and 300, respectively.
2. Calculate the RSI indicator with a period of 5.
3. Generate a long signal when the fast EMA crosses above the slow EMA and the RSI is below 45; generate a short signal when the fast EMA crosses below the slow EMA and the RSI is above 85.
4. Set different commission percentages based on the Bybit account level, ranging from 0.075% for VIP 0 to 0.035% for VIP 4.
5. Calculate the entry prices with commission included.
6. Calculate the take profit and stop loss prices based on the set percentages (5% and 3%).
7. Plot the entry prices, take profit lines, and stop loss lines on the chart.
8. Execute entry orders based on the trading signals.

#### Strategy Advantages
1. Combines trend-following and momentum indicators to effectively capture market trends.
2. Includes built-in take profit and stop loss functions to manage risk effectively.
3. Sets different commission percentages based on the Bybit account level, adapting to different users' trading conditions.
4. Plots entry prices, take profit lines, and stop loss lines on the chart, providing visual confirmation of trading signals.

#### Strategy Risks
1. The selection of EMA and RSI parameters may not be suitable for all market conditions and may require optimization based on actual situations.
2. In choppy markets, the strategy may generate frequent trading signals, leading to high trading costs.
3. The take profit and stop loss settings may be too conservative or aggressive and may need adjustment based on personal risk preferences.

#### Strategy Optimization Directions
1. Optimize the EMA and RSI parameters to adapt to different market conditions. This can be done through backtesting and parameter scanning to find the optimal parameter combinations.
2. Introduce other technical indicators, such as Bollinger Bands, MACD, etc., to improve the accuracy of trading signals.
3. Optimize the take profit and stop loss settings, for example, by using trailing stops or dynamic stop loss methods to better protect profits and manage risk.
4. Consider factors such as market volatility and trading volume to filter trading signals and reduce the costs associated with frequent trading.

#### Summary
The Bybit EMA RSI Trend-Following and Momentum Strategy is a quantitative trading strategy that combines trend-following and momentum indicators. By using EMAs and RSI together, it can effectively capture market trends. The strategy includes built-in take profit and stop loss functions and sets commission percentages based on the Bybit account level, effectively managing risk and adapting to different users' trading conditions. However, there is still room for optimization in the strategy, such as parameter optimization, introducing other technical indicators, and optimizing take profit and stop loss settings. With continuous optimization and improvement, the strategy is expected to achieve better results in actual trading.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_1|90|Fast EMA Length|
|v_input_2|300|Slow EMA Length|
|v_input_3|5|RSI Length|
|v_input_4|85|RSI Upper Threshold|
|v_input_5|45|RSI Lower Threshold|
|v_input_6|5|Take Profit %|
|v_input_7|3|Stop Loss %|
|v_input_string_1|0|Bybit Account Level: VIP 0|VIP 1|VIP 2|VIP 3|VIP 4|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-03-21 00:00:00
end: 2024-03-28 00:00:00
period: 1m
basePeriod: 1m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// @BryanAaron

//@version=5
strategy("Bybit EMA RSI Strategy", overlay=true)

// Input parameters
fastLength = input(90, title="Fast EMA Length")
slowLength = input(300, title="Slow EMA Length")
rsiLength = input(5, title="RSI Length")
rsiUpperThreshold = input(85, title="RSI Upper Threshold")
rsiLowerThreshold = input(45, title="RSI Lower Threshold")
takeProfitPerc = input(5, title="Take Profit %")
stopLossPerc = input(3, title="Stop Loss %")
bybitAccountLevel = input.string("VIP 0", title="Bybit Account Level", options=["VIP 0", "VIP 1", "VIP 2", "VIP 3", "VIP 4"])

// Calculate moving averages
fastMA = ta.ema(close, fastLength)
slowMA = ta.ema(close, slowLength)

// Calculate RSI
rsi = ta.rsi(close, rsiLength)

// Trading conditions
longCondition = (fastMA > slowMA) and (rsi < rsiLowerThreshold)
shortCondition = (fastMA < slowMA) and (rsi > rsiUpperThreshold)

// Set commission based on Bybit account level
commissionPerc = switch bybitAccountLevel
    "VIP 0" => 0.075
    "VIP 1" => 0.065
    "VIP 2" => 0.055
    "VIP 3" => 0.045
    "VIP 4" => 0.035
    => 0.075

// Calculate entry prices with commission
var float longEntryPrice = na
var float shortEntryPrice = na

longEntryPriceWithCommission = close * (1 + commissionPerc / 100)
shortEntryPriceWithCommission = close * (1 - commissionPerc / 100)

// Calculate take profit and stop loss prices
takeProfitPrice(entryPrice) => entryPrice * (1 + takeProfitPerc / 100)
stopLossPrice(entryPrice) => entryPrice * (1 - stopLossPerc / 100)

// Plot entry prices
plotchar(longCondition, title="Long Entry Price", char="LE", location=location.belowbar, color=color.green)
plotchar(shortCondition, title="Short Entry Price", char="SE", location=location.abovebar, color=color.red)

// Draw position on the chart
longColor = color.green
shortColor = color.red
profitColor = color.new(color.green, 80)
lossColor = color.new(color.red, 80)

plotshape(longCondition and strategy.position_size > 0, title="Long Position", text="Long", location=location.belowbar, style=shape.labelup, size=size.small, color=longColor, textcolor=color.white)
plotshape(shortCondition and strategy.position_size < 0, title="Short Position", text="Short", location=location.abovebar, style=shape.labeldown, size=size.small, color=shortColor, textcolor=color.white)

if (strategy.position_size > 0)
    line.new(bar_index, longEntryPrice, bar_index + 1, longEntryPrice, color=longColor, width=2)
    
    longProfitLine = line.new(bar_index, takeProfitPrice(longEntryPrice), bar_index + 1, takeProfitPrice(longEntryPrice), color=profitColor, width=1)
    longLossLine = line.new(bar_index, stopLossPrice(longEntryPrice), bar_index + 1, stopLossPrice(longEntryPrice), color=lossColor, width=1)
    

else if (strategy.position_size < 0)
    line.new(bar_index, shortEntryPrice, bar_index + 1, shortEntryPrice, color=shortColor, width=2)
    
    shortProfitLine = line.new(bar_index, stopLossPrice(shortEntryPrice), bar_index + 1, stopLossPrice(shortEntryPrice), color=profitColor, width=1)
    shortLossLine = line.new(bar_index, takeProfitPrice(shortEntryPrice), bar_index + 1, takeProfitPrice(shortEntryPrice), color=lossColor, width=1)
    


// Entry
if (longCondition)
    strategy.entry("Long", strategy.long)
    longEntryPrice := longEntryPriceWithCommission
else if (shortCondition)
    strategy.entry("Short", strategy.short)
    shortEntryPrice := shortEntryPriceWithCommission
```

> Detail

https://www.fmz.com/strategy/446556

> Last Modified

2024-03-29 16:30:42
