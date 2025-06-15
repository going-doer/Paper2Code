
> Name

MA99接触与动态止损策略-MA99-Touch-and-Dynamic-Stop-Loss-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/673f558bdac0e91900.png)

[trans]
#### 概述
该策略基于99周期简单移动平均线(MA99)来判断交易信号。当价格触及MA99时即可开仓,无需两根K线确认。而止损则采用动态止损,即当价格突破MA99并在下一根K线中得到确认时,即平仓止损。该策略旨在捕捉价格在MA99附近的波动,同时通过动态止损来控制风险。

#### 策略原理
1. 计算99周期简单移动平均线MA99。
2. 判断当前价格是否触及MA99,即最低价低于等于MA99且最高价高于等于MA99。
3. 若价格触及MA99且收盘价高于MA99,则做多;若价格触及MA99且收盘价低于MA99,则做空。
4. 对于多头仓位,若收盘价跌破MA99且下一根K线再次确认,则平仓;对于空头仓位,若收盘价突破MA99且下一根K线再次确认,则平仓。
5. 每次开仓时,将当前MA99设为止损价位;每次平仓后,重置止损价位。

#### 策略优势
1. 简单易用:该策略基于单一指标MA99,规则清晰明了,易于理解和实施。
2. 动态止损:与固定止损相比,动态止损可以更好地适应市场变化,及时控制风险。
3. 趋势跟踪:MA99代表中长期趋势,在价格触及MA99时开仓,能够跟随主要趋势方向交易。
4. 减少噪音:相比于使用更短周期的均线,99周期均线能够有效过滤短期波动噪音。

#### 策略风险
1. 参数优化:该策略仅使用了99这一参数,可能并非最优参数,需要通过回测和优化来确定最佳参数。
2. 震荡市:在震荡市场中,价格在MA99附近频繁波动,可能导致频繁交易和损失。
3. 趋势转折:当趋势转折、价格突破MA99后,该策略可能会继续持有错误方向的仓位而遭受损失。
4. 滑点成本:频繁交易可能会带来较高的滑点和交易成本,影响策略收益。

#### 策略优化方向
1. 引入趋势过滤:在判断开仓信号时,可以结合其他趋势指标如MACD、ADX等,以确认趋势强度和方向,提高开仓质量。
2. 优化参数:对MA周期、止损条件等参数进行优化,找到最佳参数组合,提高策略稳健性。
3. 加入仓位管理:根据市场趋势强度、波动率等因素动态调整仓位大小,控制回撤风险。
4. 考虑交易成本:在回测和实盘中,应考虑交易滑点、手续费等成本因素,以评估策略实际表现。

#### 总结
MA99接触与动态止损策略通过判断价格与MA99的关系来开仓,并采用动态止损来控制风险。该策略简单易用,能够跟随中长期趋势,但在震荡市中可能面临频繁交易的问题。通过引入其他指标过滤、优化参数、仓位管理和考虑成本等措施,可以进一步提升该策略的表现和稳健性。

|| 

#### Overview
This strategy is based on the 99-period Simple Moving Average (MA99) to determine trading signals. When the price touches the MA99, a position can be opened without requiring confirmation from two candles. The stop-loss uses a dynamic approach, meaning that when the price breaks through the MA99 and is confirmed in the next candle, the position is closed for stop-loss. This strategy aims to capture price fluctuations around the MA99 while controlling risk through dynamic stop-loss.

#### Strategy Principles
1. Calculate the 99-period Simple Moving Average MA99.
2. Determine if the current price touches the MA99, i.e., the lowest price is less than or equal to MA99, and the highest price is greater than or equal to MA99.
3. If the price touches MA99 and the closing price is above MA99, go long; if the price touches MA99 and the closing price is below MA99, go short.
4. For long positions, if the closing price falls below MA99 and is confirmed again in the next candle, close the position; for short positions, if the closing price breaks above MA99 and is confirmed again in the next candle, close the position.
5. Each time a position is opened, set the current MA99 as the stop-loss price; reset the stop-loss price after each position is closed.

#### Strategy Advantages
1. Simple and easy to use: This strategy is based on a single indicator, MA99, with clear and straightforward rules that are easy to understand and implement.
2. Dynamic stop-loss: Compared to fixed stop-loss, dynamic stop-loss can better adapt to market changes and control risk in a timely manner.
3. Trend following: MA99 represents the medium to long-term trend. Opening positions when the price touches MA99 allows for trading in the direction of the main trend.
4. Noise reduction: Compared to using shorter-period moving averages, the 99-period moving average can effectively filter out short-term fluctuation noise.

#### Strategy Risks
1. Parameter optimization: This strategy only uses the parameter of 99, which may not be the optimal parameter. It requires backtesting and optimization to determine the best parameters.
2. Choppy markets: In choppy markets, prices may frequently fluctuate around MA99, potentially leading to frequent trades and losses.
3. Trend reversal: When the trend reverses and the price breaks through MA99, this strategy may continue to hold positions in the wrong direction, resulting in losses.
4. Slippage costs: Frequent trading may incur higher slippage and transaction costs, affecting strategy profitability.

#### Strategy Optimization Directions
1. Introduce trend filters: When determining entry signals, other trend indicators such as MACD, ADX, etc., can be incorporated to confirm trend strength and direction, improving entry quality.
2. Optimize parameters: Optimize parameters such as the MA period and stop-loss conditions to find the best parameter combination and improve strategy robustness.
3. Incorporate position sizing: Dynamically adjust position size based on factors such as market trend strength and volatility to control drawdown risk.
4. Consider trading costs: When backtesting and live trading, consider cost factors such as trading slippage and commissions to evaluate the strategy's actual performance.

#### Summary
The MA99 Touch and Dynamic Stop-Loss Strategy opens positions based on the relationship between price and MA99 and uses dynamic stop-loss to control risk. This strategy is simple and easy to use, capable of following medium to long-term trends, but may face the problem of frequent trading in choppy markets. By introducing other indicators for filtering, optimizing parameters, managing positions, and considering costs, the performance and robustness of this strategy can be further improved.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-04-23 00:00:00
end: 2024-04-28 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/


//@version=5
strategy("MA99 Temas ve Dinamik Stop-Loss Stratejisi", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=10)

// MA99 hesaplayalım
ma99 = ta.sma(close, 99)
plot(ma99, color=color.blue, title="MA99")

// Fiyatın MA99'a temas edip etmediğini kontrol edelim
priceTouchedMA99 = (low <= ma99 and high >= ma99)

// Long ve short koşullarını tanımlayalım
longCondition = priceTouchedMA99 and close > ma99
shortCondition = priceTouchedMA99 and close < ma99

var float longStopLoss = na
var float shortStopLoss = na

var int longStopTriggered = 0
var int shortStopTriggered = 0

// Alım veya satım sinyallerine göre işlemleri başlatalım ve stop-loss ayarlayalım
if (longCondition)
    strategy.entry("Long Entry", strategy.long)
    longStopLoss := ma99
    longStopTriggered := 0

if (shortCondition)
    strategy.entry("Short Entry", strategy.short)
    shortStopLoss := ma99
    shortStopTriggered := 0

// Stop-loss koşullarını ve iki mum kuralını kontrol edelim
if (not na(longStopLoss))
    if (close < longStopLoss)
        longStopTriggered := 1
    else
        longStopTriggered := 0

    if (longStopTriggered[1] == 1 and close < longStopLoss)  // Bir önceki mumda tetiklendi ve hala altında
        strategy.close("Long Entry", comment="Stop Loss Long")
        longStopLoss := na
        longStopTriggered := 0

if (not na(shortStopLoss))
    if (close > shortStopLoss)
        shortStopTriggered := 1
    else
        shortStopTriggered := 0

    if (shortStopTriggered[1] == 1 and close > shortStopLoss)  // Bir önceki mumda tetiklendi ve hala üstünde
        strategy.close("Short Entry", comment="Stop Loss Short")
        shortStopLoss := na
        shortStopTriggered := 0
```

> Detail

https://www.fmz.com/strategy/449843

> Last Modified

2024-04-29 16:59:41
