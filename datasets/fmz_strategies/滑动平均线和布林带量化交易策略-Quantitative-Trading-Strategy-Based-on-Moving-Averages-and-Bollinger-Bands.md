
> Name

滑动平均线和布林带量化交易策略-Quantitative-Trading-Strategy-Based-on-Moving-Averages-and-Bollinger-Bands

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/11f835a4683fe6de374.png)

[trans]
#### 概述
该策略主要利用滑动平均线和布林带来捕捉市场的趋势和波动。策略中使用了三种不同的移动平均线:简单移动平均线(SMA)、加权移动平均线(WMA)和指数移动平均线(EMA)。同时利用布林带来设定价格通道,上下轨分别作为开仓和平仓的信号。当价格突破布林带上轨时开空仓,突破下轨时开多仓。同时设置了更宽的布林带作为止损位,当价格突破止损布林带时平仓。总的来说,该策略试图在趋势产生时及时建仓,在风险加大时果断止损,以期获得稳定的收益。

#### 策略原理 
1. 计算三种不同周期的移动平均线:慢速SMA、快速EMA和中速WMA,分别反映市场的长期、短期和中期趋势。
2. 根据价格标准差计算两组布林带:开仓布林带(上下轨距离较近)和止损布林带(上下轨距离较宽)。开仓布林带用于开仓,止损布林带用于平仓止损。
3. 当快速EMA上穿开仓布林带上轨时,开空头仓位;当快速EMA下穿开仓布林带下轨时,开多头仓位。这意味着价格偏离均值较多,趋势可能产生。
4. 一旦开仓后,如果价格进一步上穿止损布林带上轨,则平掉所有多头仓位;如果价格进一步下穿止损布林带下轨,则平掉所有空头仓位。这是为了控制损失,一旦趋势反转即果断止损。
5. 以上过程不断循环,使得策略可以根据市场趋势灵活调整仓位,并及时止损,以期实现稳健的收益。

#### 策略优势
1. 考虑了三种不同速度的移动平均线,全面捕捉各种级别的市场趋势。
2. 引入布林带作为开平仓条件,可以根据市场波动率动态调整,灵活应对行情。
3. 设置止损布林带,控制回撤,并在市场剧烈波动时果断平仓,避免损失扩大。
4. 逻辑清晰,规则简单,易于实现和优化。
5. 适用范围广,对多种市场、多个时间周期都可能有效。

#### 策略风险
1. 在震荡市中,频繁开平仓可能导致大量交易成本,从而侵蚀利润。
2. 趋势转折初期,策略可能仍然按原趋势方向交易,造成一定损失。
3. 对于极端行情,如价格快速跳空,止损布林带可能无法很好地控制风险。
4. 参数选择不当(如移动平均线周期、布林带宽度等)可能使策略失效。
5. 如果市场持续震荡,策略可能长期无法捕捉到明显的趋势机会。

#### 策略优化方向
1. 适当增大移动平均线周期和布林带宽度参数,以减少在震荡市中的交易频率和成本。
2. 引入更多技术指标或市场情绪指标作为过滤,以提高开仓信号的精度,避免在趋势初期可能出现的亏损交易。
3. 对极端行情设置特殊规则,如跳空时暂停开新仓等,以控制风险。
4. 对参数进行优化,找到最适合当前市场的参数组合,提高策略的稳健性。
5. 增加仓位管理和资金管理规则,如根据趋势强度或盈利情况调整仓位,设置总体止损线等,进一步控制策略风险。

#### 总结
Marina Parfenova学校项目机器人是一个基于滑动平均线和布林带的量化交易策略。它试图通过捕捉市场趋势获利,同时通过布林带止损线控制回撤。策略逻辑简单明了,适用范围广泛,可以根据市场特点灵活调整参数。但在实际应用中仍需注意震荡市、极端行情、参数优化等问题,并进一步细化资金管理和仓位管理规则。总的来说,该策略可以作为一个基础的量化交易框架,在此基础上可以不断优化和改进,以期获得更稳健的交易效果。

|| 

#### Overview
This strategy mainly utilizes moving averages and Bollinger Bands to capture market trends and volatility. It employs three different types of moving averages: Simple Moving Average (SMA), Weighted Moving Average (WMA), and Exponential Moving Average (EMA). At the same time, it uses Bollinger Bands to set price channels, with the upper and lower bands serving as signals for opening and closing positions. When the price breaks through the upper Bollinger Band, it opens a short position; when it breaks through the lower band, it opens a long position. It also sets wider Bollinger Bands as stop-loss levels, closing positions when the price breaches these bands. Overall, this strategy attempts to establish positions promptly when trends emerge and decisively cut losses when risks escalate, aiming to achieve stable returns.

#### Strategy Principles
1. Calculate three moving averages with different periods: slow SMA, fast EMA, and medium WMA, reflecting the long-term, short-term, and medium-term market trends respectively.
2. Compute two sets of Bollinger Bands based on price standard deviation: entry Bollinger Bands (with narrower distance between upper and lower bands) and stop-loss Bollinger Bands (with wider distance). Entry bands are used for opening positions, while stop-loss bands are used for closing positions.
3. When the fast EMA crosses above the upper entry Bollinger Band, open a short position; when the fast EMA crosses below the lower entry Bollinger Band, open a long position. This implies that the price has deviated significantly from the mean and a trend may emerge.
4. Once a position is open, if the price further crosses above the upper stop-loss Bollinger Band, close all long positions; if the price further crosses below the lower stop-loss Bollinger Band, close all short positions. This is to control losses and decisively stop loss once the trend reverses.
5. The above process is repeated continuously, allowing the strategy to flexibly adjust positions according to market trends and stop losses in a timely manner, in order to achieve robust returns.

#### Strategy Advantages
1. It considers three moving averages of different speeds, comprehensively capturing market trends at various levels.
2. It introduces Bollinger Bands as conditions for opening and closing positions, which can be dynamically adjusted according to market volatility, flexibly adapting to market conditions.
3. It sets stop-loss Bollinger Bands to control drawdowns and decisively closes positions when the market fluctuates dramatically, avoiding amplified losses.
4. The logic is clear and the rules are simple, easy to implement and optimize.
5. It has a wide range of applications and may be effective for various markets and time periods.

#### Strategy Risks
1. In a sideways market, frequent opening and closing of positions may lead to substantial transaction costs, eroding profits.
2. At the initial stage of a trend reversal, the strategy may still trade in the direction of the original trend, resulting in certain losses.
3. For extreme market conditions, such as rapid price gaps, the stop-loss Bollinger Bands may not effectively control risks.
4. Improper parameter selection (such as moving average periods, Bollinger Band widths, etc.) may invalidate the strategy.
5. If the market continues to fluctuate, the strategy may not be able to capture significant trend opportunities for an extended period.

#### Strategy Optimization Directions
1. Appropriately increase the parameters of moving average periods and Bollinger Band widths to reduce trading frequency and costs in a sideways market.
2. Introduce more technical indicators or market sentiment indicators as filters to improve the accuracy of entry signals and avoid losing trades that may occur at the beginning of a trend.
3. Set special rules for extreme market conditions, such as suspending new positions when gaps occur, to control risks.
4. Optimize parameters to find the most suitable combination for the current market, enhancing the robustness of the strategy.
5. Add position management and capital management rules, such as adjusting positions based on trend strength or profitability, setting overall stop-loss lines, etc., to further control strategy risks.

#### Summary
The Marina Parfenova School Project Bot is a quantitative trading strategy based on moving averages and Bollinger Bands. It attempts to profit by capturing market trends while controlling drawdowns through Bollinger Band stop-loss lines. The strategy logic is simple and straightforward, with a wide range of applications, and parameters can be flexibly adjusted according to market characteristics. However, in practical application, attention still needs to be paid to issues such as sideways markets, extreme conditions, parameter optimization, etc., and further refinement of capital and position management rules is necessary. Overall, this strategy can serve as a basic quantitative trading framework, which can be continuously optimized and improved upon to achieve more robust trading results.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_int_1|50|Период медленной скользящей средней|
|v_input_int_2|4|Период быстрой скользящей средней|
|v_input_int_3|30|Период среднеквадратического отклонения|
|v_input_float_1|1.2|Коэффициент ширины коридора покупки|
|v_input_float_2|1.6|Коэффициент ширины коридора продажи|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-03-01 00:00:00
end: 2024-03-31 23:59:59
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy ("Marina Parfenova School Project Bot", overlay = true)

sma(price, n) =>
    result = 0.0
    for i = 0 to n - 1
        result := result + price [i] / n    
    result

wma(price, n) =>
    result = 0.0
    sum_weight = 0.0
    weight = 0.0
    for i = 0 to n - 1
        weight := n - 1
        result := result + price [i]*weight
        sum_weight := sum_weight + weight
    result/sum_weight

ema(price, n) =>
    result = 0.0
    alpha = 2/(n + 1)
    prevResult = price 
    if (na(result[1]) == false)
        prevResult := result[1]
    result := alpha * price + (1 - alpha) * prevResult

/// Настройки
n_slow = input.int(50, "Период медленной скользящей средней", step=5)
n_fast = input.int(4, "Период быстрой скользящей средней")
n_deviation = input.int(30, "Период среднеквадратического отклонения", step=5)
k_deviation_open = input.float(1.2, "Коэффициент ширины коридора покупки", step=0.1)
k_deviation_close = input.float(1.6, "Коэффициент ширины коридора продажи", step=0.1)

// ----- Линии индикаторов -----

// Медленная скользящая 
sma = sma(close, n_slow)
plot(sma, color=#d3d3d3)

// Линии Боллинджера, обозначающие коридор цены
bollinger_open = k_deviation_open * ta.stdev(close, n_deviation)
open_short_line = sma + bollinger_open
plot(open_short_line, color=#ec8383)
open_long_line = sma - bollinger_open
plot(open_long_line, color=#6dd86d)

bollinger_close = k_deviation_close * ta.stdev(close, n_deviation)
close_short_line = sma + bollinger_close
plot(close_short_line, color=#e3e3e3)
close_long_line = sma - bollinger_close
plot(close_long_line, color=#e3e3e3)

// Быстрая скользящая
ema = ema(close, n_fast)
plot(ema, color = color.aqua, linewidth = 2)

// ----- Сигналы для запуска стратегии -----

// если ema пересекает линию open_short сверху вниз - сигнал на создание ордера в short
if(ema[1] >= open_short_line[1] and ema < open_short_line)
    strategy.entry("short", strategy.short)

// если ema пересекает линию open_long снизу вверх - сигнал на создание ордера в long
if(ema[1] <= open_long_line[1] and ema > open_long_line)
    strategy.entry("long", strategy.long)

// если свеча пересекает верхнюю линию коридора продажи - закрываем все long-ордера 
if (high >= close_short_line)
    strategy.close("long")

// если свеча пересекает нижнюю линию коридора продажи - закрываем все short-ордера
if (low <= close_long_line)
    strategy.close("short")
```

> Detail

https://www.fmz.com/strategy/449506

> Last Modified

2024-04-26 11:45:05
