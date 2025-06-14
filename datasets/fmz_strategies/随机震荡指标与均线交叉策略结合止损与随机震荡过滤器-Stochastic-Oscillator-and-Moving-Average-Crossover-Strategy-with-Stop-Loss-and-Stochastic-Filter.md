
> Name

随机震荡指标与均线交叉策略结合止损与随机震荡过滤器-Stochastic-Oscillator-and-Moving-Average-Crossover-Strategy-with-Stop-Loss-and-Stochastic-Filter

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/be1910e5150144a87d.png)

[trans]
#### 概述

该策略结合了随机震荡指标(Stochastic Oscillator)与移动平均线(Moving Average),通过观察随机指标的超买超卖情况以及移动平均线的趋势来产生交易信号。当随机指标在超买区且移动平均线向下时产生做空信号,在超卖区且移动平均线向上时产生做多信号。同时,该策略还引入了随机指标过滤器,当随机指标K线在50以下维持一定数量K线后,与D线交叉也可以产生相应的交易信号。该策略还设置了止损(Stop Loss)来控制风险。

#### 策略原理

1. 计算随机震荡指标,得到K线和D线。参数可调,包括随机指标周期、K值平滑、D值平滑、超买区和超卖区。  

2. 计算移动平均线,默认使用收盘价,周期可调。

3. 计算随机指标过滤器。当K线在50以下维持一定K线后,产生过滤信号。周期可调。

4. 产生多头信号的条件:随机指标在超卖区交叉向上 或 随机指标过滤器信号 且 移动平均线向上。

5. 产生空头信号的条件:随机指标在超买区交叉向下 或 随机指标过滤器信号 且 移动平均线向下。

6. 多头平仓条件:随机K线上穿移动平均线 且 均线转向下行。  

7. 空头平仓条件:随机K线下穿移动平均线 且 均线转向上行。

8. 仓位管理使用固定资金比例,默认10%。同时设置止损,默认为2%。

#### 优势分析

1. 结合超买超卖和趋势特征,可以在趋势中追涨杀跌。

2. 随机指标过滤器避免在震荡行情频繁交易。

3. 止损设置有助于控制回撤。

4. 代码结构清晰,参数可调,适合进一步优化。

#### 风险分析

1. 随机指标具有一定滞后性,可能错过最佳买卖点。

2. 在趋势转折点抓单准确性欠佳,止损频率可能较高。

3. 固定比例资金管理在连续亏损的情况下回撤较大。

#### 优化方向

1. 引入更多过滤条件,如价格行为、其他辅助指标等,提高信号准确度。

2. 对信号进行强弱划分,在强势信号出现时加大仓位。

3. 对趋势转折点的判断进行优化,以期抓住更多行情。

4. 对仓位管理进行优化,可以考虑浮动盈亏比仓位调整等。

5. 尝试不同参数组合,寻找最优参数。

#### 总结

该策略在随机震荡指标的基础上,结合移动平均线对趋势进行判断,同时运用随机指标本身的过滤功能,产生相对可靠的交易信号。策略整体思路清晰,适合在趋势行情中运用。但是由于随机指标滞后性的存在,在行情转折点的表现可能欠佳,整体适应性和鲁棒性有待进一步考察。后续可以从过滤条件、仓位管理、参数优化等方面对策略进行完善。

|| 
#### Overview
This strategy combines the Stochastic Oscillator with a Moving Average, generating trading signals by observing the overbought and oversold conditions of the stochastic indicator and the trend of the moving average. It produces a short signal when the stochastic indicator is in the overbought zone and the moving average is downward, and a long signal when in the oversold zone and the moving average is upward. Additionally, the strategy introduces a stochastic indicator filter, which can also generate corresponding trading signals when the stochastic K line crosses the D line after staying below 50 for a certain number of K lines. The strategy also sets a Stop Loss to control risk.

#### Strategy Principle
1. Calculate the Stochastic Oscillator to obtain the K line and D line. Parameters are adjustable, including the stochastic period, K smoothing, D smoothing, overbought zone, and oversold zone.

2. Calculate the Moving Average, using the closing price by default, with an adjustable period.

3. Calculate the Stochastic Indicator Filter. When the K line stays below 50 for a certain number of K lines, it generates a filter signal.

4. Conditions for generating a long signal: Stochastic indicator crosses upward in the oversold zone OR Stochastic indicator filter signal AND Moving average is upward.

5. Conditions for generating a short signal: Stochastic indicator crosses downward in the overbought zone OR Stochastic indicator filter signal AND Moving average is downward.

6. Long position closing condition: Stochastic K line crosses above the Moving Average AND the Average turns downward.

7. Short position closing condition: Stochastic K line crosses below the Moving Average AND the Average turns upward.

8. Position management uses a fixed percentage of funds, 10% by default. It also sets a Stop Loss, 2% by default.

#### Advantage Analysis
1. By combining overbought/oversold and trend characteristics, it can chase up and kill down in a trend.

2. The Stochastic Indicator Filter avoids frequent trading in oscillating markets.

3. The Stop Loss setting helps control drawdowns.

4. The code structure is clear, parameters are adjustable, and it is suitable for further optimization.

#### Risk Analysis
1. The Stochastic Oscillator has a certain lag, which may miss the best buying and selling points.

2. The accuracy of capturing orders at trend turning points is poor, and the frequency of stop-loss may be high.

3. Fixed-ratio fund management has a large drawdown in the case of consecutive losses.

#### Optimization Direction
1. Introduce more filtering conditions, such as price behavior, other auxiliary indicators, etc., to improve signal accuracy.

2. Divide signals into strong and weak, and increase positions when strong signals appear.

3. Optimize the judgment of trend turning points to capture more market movements.

4. Optimize position management, consider adjusting positions based on floating profit and loss ratios, etc.

5. Try different parameter combinations to find the optimal parameters.

#### Summary
Based on the Stochastic Oscillator, this strategy combines Moving Averages to judge trends, while also utilizing the filtering function of the Stochastic Indicator itself, generating relatively reliable trading signals. The overall idea of the strategy is clear and suitable for use in trending markets. However, due to the lag of the Stochastic Oscillator, its performance at market turning points may be poor, and its overall adaptability and robustness need further examination. In the future, the strategy can be improved from aspects such as filtering conditions, position management, and parameter optimization.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_int_1|14|Longitud Estocástico|
|v_input_int_2|3|Suavizado K|
|v_input_int_3|3|Suavizado D|
|v_input_int_4|20|Sobreventa|
|v_input_int_5|80|Sobrecompra|
|v_input_int_6|9|Longitud MA|
|v_input_1_close|0|Fuente MA: close|high|low|open|hl2|hlc3|hlcc4|ohlc4|
|v_input_int_7|2|Stop Loss (%)|
|v_input_int_8|12|Ruedas de Filtro Estocástico|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-03-01 00:00:00
end: 2024-03-31 23:59:59
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © Pablo_2uc

//@version=5
strategy("Estrategia Estocástico + MA con Stop Loss y Filtro Estocástico", overlay=true)

// Parámetros del Estocástico
length = input.int(14, title="Longitud Estocástico")
smoothK = input.int(3, title="Suavizado K")
smoothD = input.int(3, title="Suavizado D")
oversold = input.int(20, title="Sobreventa")
overbought = input.int(80, title="Sobrecompra")

// Parámetros de la Media Móvil
maLength = input.int(9, title="Longitud MA")
maSource = input(close, title="Fuente MA")

// Capital inicial
capital = 5000

// Tamaño de posición (10% del capital)
positionSize = capital * 0.10

// Stop Loss (2% del precio de entrada)
stopLossPercent = input.int(2, title="Stop Loss (%)") / 100

// Número de ruedas para el filtro estocástico
filterPeriods = input.int(12, title="Ruedas de Filtro Estocástico")

// Cálculo del Estocástico
k = ta.sma(ta.stoch(close, high, low, length), smoothK)
d = ta.sma(k, smoothD)

// Cálculo de la Media Móvil
ma = ta.sma(maSource, maLength)

// Filtro estocástico
stochasticFilter = ta.sma(k > 50 ? 1 : 0, filterPeriods)

// Condiciones de entrada en largo y corto
longCondition = (ta.crossunder(k, oversold) or ta.crossover(stochasticFilter, 1)) and ma > ma[1]
shortCondition = (ta.crossover(k, overbought) or ta.crossover(stochasticFilter, 1)) and ma < ma[1]

// Condiciones de salida
exitLongCondition = ta.crossover(k, ma) and ma < ma[1]
exitShortCondition = ta.crossunder(k, ma) and ma > ma[1]

// Estrategia
if (longCondition)
    strategy.entry("Long", strategy.long, qty=positionSize)
    strategy.exit("Exit Long", "Long", stop=close * (1 - stopLossPercent))
if (shortCondition)
    strategy.entry("Short", strategy.short, qty=positionSize)
    strategy.exit("Exit Short", "Short", stop=close * (1 + stopLossPercent))

// Cierre de posiciones
if (exitLongCondition)
    strategy.close("Long")
if (exitShortCondition)
    strategy.close("Short")
```

> Detail

https://www.fmz.com/strategy/449526

> Last Modified

2024-04-26 16:10:11
