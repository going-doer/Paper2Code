
> Name

TEMA双均线交叉策略-TEMA-Dual-Moving-Average-Crossover-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/d14b0a0c9737d4ca5e.png)
[trans]
#### 概述
TEMA双均线交叉策略是一种基于两条不同周期的三重指数移动平均线(TEMA)交叉信号产生交易的量化交易策略。该策略通过比较两条TEMA线的相对位置,当短期TEMA线上穿长期TEMA线时开仓做多,当短期TEMA线下穿长期TEMA线时开仓做空,当出现相反的交叉信号时平仓。该策略适用于在震荡市场中捕捉短期趋势。

#### 策略原理
TEMA双均线交叉策略的核心是构建两条不同周期的TEMA线。TEMA是对EMA(指数移动平均线)的一种改进,通过对EMA的EMA再做一次EMA来计算,相比EMA和SMA(简单移动平均线)具有更少的滞后性,更加贴近价格走势,对短期趋势更加敏感。

策略通过比较短期TEMA线和长期TEMA线的位置关系来产生交易信号:
1. 当短期TEMA线上穿长期TEMA线,且短期TEMA线位于长期TEMA线上方时,开仓做多。
2. 当短期TEMA线下穿长期TEMA线,且短期TEMA线位于长期TEMA线下方时,开仓做空。
3. 当持有多单时,如果短期TEMA线下穿长期TEMA线则平多单;当持有空单时,如果短期TEMA线上穿长期TEMA线则平空单。

通过两条不同周期的TEMA线的交叉信号来开仓和平仓,可以在震荡市场中捕捉短期价格趋势。

#### 策略优势
1. TEMA指标相比EMA和SMA具有更少的滞后性,信号更加灵敏,更能贴合价格的走势。
2. 通过两条不同周期的TEMA线交叉来产生开平仓信号,信号明确,能有效把握短期趋势行情。
3. 策略逻辑和代码实现简单清晰,容易理解和优化。
4. 适合在震荡市场中使用,可以获得较为稳定的收益。

#### 策略风险
1. 在单边趋势行情下,该策略可能会出现频繁交易,导致交易成本增加,影响收益。
2. TEMA指标相比EMA和SMA对价格更加敏感,在市场波动剧烈时可能会出现频繁的虚假信号。
3. 策略在参数选择上比较依赖历史数据,如果未来市场特征发生变化,可能影响策略表现。
4. 策略没有设置止损,在极端行情下可能承担较大风险。

#### 策略优化方向
1. 可以通过优化TEMA指标的参数来提高策略表现,例如使用参数优化方法找到最佳的两条TEMA线周期参数。
2. 在产生交易信号时,可以结合其他技术指标或市场情绪指标作为过滤条件,以提高信号的可靠性,减少虚假信号。
3. 可以根据市场波动特征,设置动态止损和移动止损来控制风险。
4. 可以考虑分析持仓周期和交易频率,根据市场特征和交易成本来优化开平仓时机和交易频率。
5. 可以将该策略和其他类型策略进行组合,发挥不同策略的优势,提高策略稳健性。

#### 总结
TEMA双均线交叉策略是一个简单易用的量化交易策略,通过两条不同周期的TEMA指标交叉信号来捕捉短期价格趋势。该策略逻辑清晰,适合在震荡市场中使用。但是该策略也存在一些风险,如频繁交易、虚假信号和极端行情风险等。可以通过优化参数、增加过滤条件、设置止损和组合不同策略等方法来改进策略表现,提高策略的稳健性和实用性。

|| 

#### Overview
The TEMA Dual Moving Average Crossover Strategy is a quantitative trading strategy that generates trading signals based on the crossover of two Triple Exponential Moving Averages (TEMA) with different periods. The strategy compares the relative positions of the two TEMA lines. It opens a long position when the short-term TEMA line crosses above the long-term TEMA line and opens a short position when the short-term TEMA line crosses below the long-term TEMA line. The positions are closed when the opposite crossover signals occur. This strategy is suitable for capturing short-term trends in a ranging market.

#### Strategy Principle
The core of the TEMA Dual Moving Average Crossover Strategy is to construct two TEMA lines with different periods. TEMA is an improvement over the Exponential Moving Average (EMA). It is calculated by applying EMA to the EMA of the EMA, resulting in less lag compared to EMA and Simple Moving Average (SMA). TEMA is more responsive to price movements and more sensitive to short-term trends.

The strategy generates trading signals by comparing the positions of the short-term and long-term TEMA lines:
1. When the short-term TEMA line crosses above the long-term TEMA line and the short-term TEMA line is above the long-term TEMA line, it opens a long position.
2. When the short-term TEMA line crosses below the long-term TEMA line and the short-term TEMA line is below the long-term TEMA line, it opens a short position.
3. When holding a long position, if the short-term TEMA line crosses below the long-term TEMA line, it closes the long position. When holding a short position, if the short-term TEMA line crosses above the long-term TEMA line, it closes the short position.

By using the crossover signals of two TEMA lines with different periods, it can capture short-term price trends in a ranging market.

#### Strategy Advantages
1. TEMA indicator has less lag compared to EMA and SMA, providing more responsive signals and better alignment with price movements.
2. By using the crossover signals of two TEMA lines with different periods to open and close positions, the signals are clear and effective in capturing short-term trends.
3. The strategy logic and code implementation are simple and clear, easy to understand and optimize.
4. Suitable for use in ranging markets, potentially generating stable returns.

#### Strategy Risks
1. In strong trending markets, the strategy may generate frequent trades, leading to increased transaction costs and affecting profitability.
2. TEMA indicator is more sensitive to price compared to EMA and SMA, potentially generating frequent false signals during high market volatility.
3. The strategy's performance depends on parameter selection based on historical data. If future market characteristics change, it may impact the strategy's performance.
4. The strategy does not include stop-loss, potentially incurring significant risks in extreme market conditions.

#### Strategy Optimization Directions
1. Optimize the parameters of the TEMA indicator to improve strategy performance, such as using parameter optimization methods to find the optimal periods for the two TEMA lines.
2. When generating trading signals, incorporate other technical indicators or market sentiment indicators as filters to improve signal reliability and reduce false signals.
3. Set dynamic stop-loss and trailing stop-loss based on market volatility characteristics to control risks.
4. Analyze holding periods and trading frequency, optimize entry and exit timing and trading frequency based on market characteristics and transaction costs.
5. Consider combining this strategy with other types of strategies to leverage the strengths of different strategies and improve overall robustness.

#### Summary
The TEMA Dual Moving Average Crossover Strategy is a simple and easy-to-use quantitative trading strategy that captures short-term price trends using crossover signals of two TEMA indicators with different periods. The strategy has a clear logic and is suitable for use in ranging markets. However, the strategy also has some risks, such as frequent trading, false signals, and extreme market risks. The strategy performance can be improved by optimizing parameters, adding filter conditions, setting stop-losses, and combining with other strategies to enhance its robustness and practicality.
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

//@version=5
strategy('2 TEMA Cross Strategy', shorttitle='2 TEMA Cross Strat', overlay=true, initial_capital=25000, currency=currency.USD)
//My backtesting showed best results on a 5 min chart
//Create 2 TEMA Input and pre-populate
len1 = input.int(9, minval=1, title='Length 1')
len2 = input.int(26, minval=2, title='Length 2')

//Calculate Tema values for each Input
//Tema 1
ema1 = ta.ema(close, len1)
ema11 = ta.ema(ema1, len1)
ema111 = ta.ema(ema11, len1)
tema1 = 3 * (ema1 - ema11) + ema111

//Tema 2
ema2 = ta.ema(close, len2)
ema22 = ta.ema(ema2, len2)
ema222 = ta.ema(ema22, len2)
tema2 = 3 * (ema2 - ema22) + ema222

//Plot the MAs
plot(tema1, color=color.new(color.black, 20))
plot(tema2, color=color.new(color.maroon, 20))

// Define long/short conditions
long = ta.crossover(tema1, tema2) and tema1 > tema2  
short = ta.crossunder(tema1, tema2) and tema1 < tema2
exitLong = ta.crossunder(tema1, tema2)
exitShort = ta.cross(tema1, tema2)

// Buys when buy condition met
strategy.entry('long', strategy.long, when=long)  
strategy.close('long', when=exitLong)

// Closes position when sell condition met
strategy.entry('short', strategy.short, when=short)  
strategy.close('short', when=exitShort)


```

> Detail

https://www.fmz.com/strategy/453234

> Last Modified

2024-06-03 10:59:42
