
> Name

TURTLE-ATR布林带突破策略-TURTLE-ATR-Bollinger-Bands-Breakout-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1379c2aadf995b85673.png)

[trans]
#### 概述
这是一个基于海龟交易法则的趋势跟踪策略。该策略使用ATR(平均真实波幅)来确定趋势方向和交易头寸规模。当价格突破过去一段时间的最高价或最低价时,策略会开仓做多或做空。持仓头寸会一直持有,直到价格突破过去一段时间的最低价或最高价,策略才会平仓。该策略的目的是捕捉强劲的趋势行情,同时严格控制风险。

#### 策略原理
该策略的核心是利用ATR指标来确定趋势方向和交易头寸规模。ATR指标可以衡量市场波动性,从而帮助我们确定合适的止损位和头寸规模。策略的主要步骤如下:
1. 计算ATR指标值。
2. 确定多头和空头的突破价格水平。多头突破价格为过去一段时间的最高价,空头突破价格为过去一段时间的最低价。
3. 如果价格突破多头突破价格,开仓做多;如果价格突破空头突破价格,开仓做空。
4. 根据ATR指标值和账户余额,计算每次交易的头寸规模。
5. 持有头寸,直到价格突破过去一段时间的最低价(多头持仓)或最高价(空头持仓),平仓了结。

通过这种方式,该策略能够捕捉强劲的趋势行情,同时严格控制风险。ATR指标的使用可以帮助我们动态调整头寸规模,从而更好地适应市场波动。

#### 策略优势
1. 趋势跟踪:该策略的目标是捕捉强劲的趋势行情,从而获取可观的利润。
2. 风险控制:通过使用ATR指标来确定头寸规模和止损位,该策略可以有效控制风险。
3. 适应性强:ATR指标可以动态调整,从而使策略能够适应不同的市场环境。
4. 简单易用:该策略的逻辑清晰,易于理解和实现。

#### 策略风险
1. 趋势反转:当市场趋势突然反转时,该策略可能会遭受较大损失。
2. 震荡市场:在震荡市场中,该策略可能会频繁开仓平仓,导致高昂的交易成本。
3. 参数敏感性:该策略的表现可能对参数设置较为敏感,不恰当的参数可能导致策略表现不佳。

为了应对这些风险,可以考虑以下解决方法:
1. 引入趋势确认机制,避免在趋势反转时过早开仓。
2. 在震荡市场中减小头寸规模,或暂停交易。
3. 对参数进行优化,寻找最佳的参数组合。

#### 策略优化方向
1. 引入更多指标:除了ATR指标外,可以考虑引入其他趋势确认指标,如移动平均线,以提高趋势判断的准确性。
2. 动态调整参数:根据不同的市场环境,动态调整策略参数,如ATR周期、突破价格周期等,以适应市场变化。
3. 加入止盈机制:在获得一定利润后,可以考虑部分平仓,以锁定利润,降低风险。
4. 多空头寸管理:可以考虑分别对多头和空头头寸进行管理,如采用不同的止损止盈标准,以提高策略的灵活性。

通过以上优化,可以进一步提高该策略的稳定性和盈利能力。

#### 总结
TURTLE-ATR布林带突破策略是一个基于海龟交易法则的趋势跟踪策略。该策略利用ATR指标来确定趋势方向和交易头寸规模,通过突破过去一段时间的最高价或最低价来开仓,并持有头寸直到趋势反转。该策略的优势在于能够捕捉强劲的趋势行情,同时严格控制风险。但是,该策略也面临着趋势反转、震荡市场和参数敏感性等风险。为了进一步提高策略的表现,可以考虑引入更多指标、动态调整参数、加入止盈机制和优化头寸管理等方面进行优化。总的来说,TURTLE-ATR布林带突破策略是一个简单易用、适应性强的趋势跟踪策略,值得进一步研究和应用。

|| 

#### Overview
This is a trend-following strategy based on the Turtle Trading rules. The strategy uses ATR (Average True Range) to determine the trend direction and trading position size. When the price breaks out of the highest or lowest price over a certain period, the strategy will open a long or short position. The position will be held until the price breaks out of the lowest or highest price over a certain period, at which point the strategy will close the position. The goal of this strategy is to capture strong trending markets while strictly controlling risk.

#### Strategy Principle
The core of this strategy is to use the ATR indicator to determine the trend direction and trading position size. The ATR indicator can measure market volatility, helping us determine appropriate stop-loss levels and position sizes. The main steps of the strategy are as follows:
1. Calculate the ATR indicator value.
2. Determine the breakout price levels for long and short positions. The long breakout price is the highest price over a certain period, and the short breakout price is the lowest price over a certain period.
3. If the price breaks out above the long breakout price, open a long position; if the price breaks out below the short breakout price, open a short position.
4. Calculate the position size for each trade based on the ATR indicator value and account balance.
5. Hold the position until the price breaks out of the lowest price (for long positions) or highest price (for short positions) over a certain period, then close the position.

By following this approach, the strategy can capture strong trending markets while strictly controlling risk. The use of the ATR indicator helps us dynamically adjust position sizes to better adapt to market volatility.

#### Strategy Advantages
1. Trend Following: The strategy aims to capture strong trending markets, thus generating substantial profits.
2. Risk Control: By using the ATR indicator to determine position sizes and stop-loss levels, the strategy can effectively control risk.
3. Adaptability: The ATR indicator can be dynamically adjusted, allowing the strategy to adapt to different market environments.
4. Simplicity: The strategy's logic is clear and easy to understand and implement.

#### Strategy Risks
1. Trend Reversal: When market trends suddenly reverse, the strategy may suffer significant losses.
2. Choppy Markets: In choppy markets, the strategy may frequently open and close positions, leading to high trading costs.
3. Parameter Sensitivity: The performance of the strategy may be sensitive to parameter settings, and inappropriate parameters may lead to poor performance.

To address these risks, the following solutions can be considered:
1. Introduce a trend confirmation mechanism to avoid opening positions prematurely when trends reverse.
2. Reduce position sizes or suspend trading in choppy markets.
3. Optimize parameters to find the best combination.

#### Strategy Optimization Directions
1. Introduce More Indicators: In addition to the ATR indicator, consider introducing other trend confirmation indicators, such as moving averages, to improve the accuracy of trend identification.
2. Dynamic Parameter Adjustment: Dynamically adjust strategy parameters based on different market environments, such as ATR period and breakout price period, to adapt to market changes.
3. Incorporate a Take Profit Mechanism: After achieving a certain profit, consider partially closing positions to lock in profits and reduce risk.
4. Long/Short Position Management: Consider managing long and short positions separately, such as using different stop-loss and take-profit standards, to improve the strategy's flexibility.

Through the above optimizations, the stability and profitability of the strategy can be further improved.

#### Summary
The TURTLE-ATR Bollinger Bands Breakout Strategy is a trend-following strategy based on the Turtle Trading rules. The strategy utilizes the ATR indicator to determine trend direction and trading position size, opening positions when the price breaks out of the highest or lowest price over a certain period and holding positions until the trend reverses. The strategy's advantages lie in its ability to capture strong trending markets while strictly controlling risk. However, the strategy also faces risks such as trend reversals, choppy markets, and parameter sensitivity. To further improve the strategy's performance, optimizations can be considered in areas such as introducing more indicators, dynamically adjusting parameters, incorporating take-profit mechanisms, and optimizing position management. Overall, the TURTLE-ATR Bollinger Bands Breakout Strategy is a simple, adaptable trend-following strategy that is worth further research and application.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-05-28 00:00:00
end: 2024-06-02 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © luisfeijoo22

//@version=5
strategy("Estrategia de las tortugas_ES", overlay=true, pyramiding=3)

// Parámetros 
atrLength = input.int(20, "Longitud del ATR")
atrFactor = input.float(2, "Factor del ATR")
entryBreakout = input.int(20, "Breakout de entrada")
exitBreakout = input.int(10, "Breakout de salida")
longOnly = input.bool(false, "Solo largos")
shortOnly = input.bool(false, "Solo cortos")


// Cálculo del ATR
atr = ta.atr(atrLength)

// Cálculo de los niveles de breakout
longEntry = ta.highest(high, exitBreakout)[1]
longExit = ta.lowest(low, exitBreakout)[1]
shortEntry = ta.lowest(low, exitBreakout)[1]
shortExit = ta.highest(high, exitBreakout)[1]

// Cálculo del tamaño de la posición
nContracts = math.floor((strategy.equity * 0.01) / (atrFactor * atr))

// Filtra las fechas según el rango deseado
// in_range = time >= timestamp(year(start_date), month(start_date), dayofmonth(start_date)) 

// Condiciones de entrada y salida
longCondition = not longOnly and close > longEntry and time >= timestamp("2023-03-15")
if longCondition
    strategy.entry("Long", strategy.long, qty = nContracts)

shortCondition = not shortOnly and close < shortEntry and time >=  timestamp("2023-03-15")
if shortCondition
    strategy.entry("Short", strategy.short, qty = nContracts)
    
strategy.exit("Exit Long", "Long", stop = longExit)
strategy.exit("Exit Short", "Short", stop = shortExit)

// Visualización de los niveles de breakout
plot(longEntry, "Entrada larga", color.green, style = plot.style_line)
plot(longExit, "Salida larga", color.red, style = plot.style_line)
plot(shortEntry, "Entrada corta", color.green, style = plot.style_line)
plot(shortExit, "Salida corta", color.red, style = plot.style_line)


```

> Detail

https://www.fmz.com/strategy/453228

> Last Modified

2024-06-03 10:48:09
