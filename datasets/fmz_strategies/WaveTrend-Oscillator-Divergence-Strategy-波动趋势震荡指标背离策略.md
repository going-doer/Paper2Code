
> Name

WaveTrend-Oscillator-Divergence-Strategy-波动趋势震荡指标背离策略

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/10cd391a3ffbeeb2711.png)
[trans]
#### 概述
这个策略结合了WaveTrend震荡指标(WT)和成交量加权平均价格(VWAP),通过识别价格和指标的背离来捕捉潜在的趋势反转机会。该策略使用ATR(Average True Range)来确定止损位置,并根据账户风险百分比动态调整头寸规模。该策略的主要优势在于其趋势跟踪能力和风险管理措施,但在震荡市场中可能会遭受亏损。优化方向包括增加额外的过滤条件和改进进出场规则。

#### 策略原理
1. 计算WaveTrend震荡指标(WT):通过比较当前价格与其通道和平均值之间的差异,生成动量振荡指标。
2. 计算成交量加权平均价格(VWAP):使用成交量作为权重计算移动平均价格。
3. 识别价格和WT指标的背离:当价格创新高/新低而指标未能创新高/新低时,表明可能发生趋势反转。
4. 进场条件:当识别到看涨背离时,开仓做多;当识别到看跌背离时,平仓。
5. 止损:基于ATR(Average True Range)设置动态止损位置。
6. 头寸规模:根据账户风险百分比和止损距离动态调整每笔交易的头寸规模。
7. 背景颜色:根据指标的超买/超卖水平改变背景颜色,提供额外的视觉提示。

#### 优势分析
1. 趋势跟踪:通过识别价格和指标背离,该策略能够捕捉潜在的趋势反转机会。
2. 风险管理:使用基于ATR的动态止损和根据风险百分比调整头寸规模,有助于控制潜在损失。
3. 视觉提示:背景颜色根据指标的超买/超卖状态变化,为交易者提供额外的视觉信号。
4. 灵活性:该策略的参数(如通道长度、平均长度、超买/超卖等级)可以根据不同的市场条件和交易风格进行调整。

#### 风险分析
1. 震荡市场:在没有明确趋势的市场条件下,该策略可能会遭受连续亏损。
2. 参数优化:该策略的表现在很大程度上取决于参数的选择,不当的参数设置可能导致次优结果。
3. 过度交易:频繁的进出场信号可能导致较高的交易成本,影响策略的整体表现。

#### 优化方向
1. 趋势过滤:在发生背离时,引入额外的趋势确认指标(如移动平均线),以过滤掉潜在的假信号。
2. 动态参数:根据市场波动性调整指标参数,在波动较低时使用较短的通道和平均长度,在波动较高时使用较长的参数。
3. 止盈:引入基于风险回报比或目标价格的动态止盈水平,以更好地管理已获利的头寸。
4. 多空过滤:根据市场的总体趋势方向(如长期移动平均线)过滤交易信号,只在趋势方向上进行交易。

#### 总结
WaveTrend Oscillator Divergence Strategy结合了波动趋势指标和成交量加权平均价格,以识别潜在的趋势反转机会。该策略的优势在于其趋势跟踪能力和风险管理措施,但在震荡市场中可能面临风险。通过引入额外的过滤条件、动态参数调整和改进的进出场规则,可以进一步优化该策略的表现。在实施该策略之前,全面的回测和前瞻性分析是至关重要的。

|| 

#### Overview
This strategy combines the WaveTrend Oscillator (WT) and the Volume Weighted Average Price (VWAP) to capture potential trend reversal opportunities by identifying divergences between price and the indicator. The strategy uses the Average True Range (ATR) to determine stop-loss levels and dynamically adjusts position sizing based on account risk percentage. The main strengths of the strategy lie in its trend-following capabilities and risk management measures, but it may suffer losses in choppy markets. Optimization directions include adding additional filters and improving entry and exit rules.

#### Strategy Principles
1. Calculate the WaveTrend Oscillator (WT): Generate a momentum oscillator by comparing the current price with its channel and average.
2. Calculate the Volume Weighted Average Price (VWAP): Compute a moving average price weighted by volume.
3. Identify divergences between price and the WT indicator: A potential trend reversal is indicated when price makes a new high/low while the indicator fails to do so.
4. Entry conditions: Open a long position when a bullish divergence is identified; close the position when a bearish divergence is identified.
5. Stop-loss: Set dynamic stop-loss levels based on the Average True Range (ATR).
6. Position sizing: Dynamically adjust the position size for each trade based on the account risk percentage and stop-loss distance.
7. Background color: Change the background color based on the overbought/oversold levels of the indicator, providing additional visual cues.

#### Advantages Analysis
1. Trend following: By identifying divergences between price and the indicator, the strategy can capture potential trend reversal opportunities.
2. Risk management: The use of ATR-based dynamic stop-losses and position sizing based on risk percentage helps control potential losses.
3. Visual cues: The background color changes based on the overbought/oversold state of the indicator, providing additional visual signals to traders.
4. Flexibility: The strategy's parameters (e.g., channel length, average length, overbought/oversold levels) can be adjusted to suit different market conditions and trading styles.

#### Risk Analysis
1. Choppy markets: In market conditions lacking clear trends, the strategy may suffer consecutive losses.
2. Parameter optimization: The strategy's performance largely depends on the choice of parameters, and suboptimal parameter settings may lead to subpar results.
3. Overtrading: Frequent entry and exit signals may result in high trading costs, affecting the overall performance of the strategy.

#### Optimization Directions
1. Trend filters: Introduce additional trend confirmation indicators (e.g., moving averages) when divergences occur to filter out potential false signals.
2. Dynamic parameters: Adjust indicator parameters based on market volatility, using shorter channel and average lengths during low volatility and longer parameters during high volatility.
3. Take-profit: Introduce dynamic take-profit levels based on risk-reward ratios or target prices to better manage profitable positions.
4. Long/short filters: Filter trading signals based on the overall market trend direction (e.g., long-term moving averages) to only trade in the direction of the trend.

#### Summary
The WaveTrend Oscillator Divergence Strategy combines the WaveTrend indicator and the Volume Weighted Average Price to identify potential trend reversal opportunities. The strategy's strengths lie in its trend-following capabilities and risk management measures, but it may face risks in choppy markets. The strategy can be further optimized by introducing additional filters, dynamic parameter adjustments, and improved entry and exit rules. Thorough backtesting and forward-looking analysis are crucial before implementing the strategy.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-05-22 00:00:00
end: 2024-05-27 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("PipShiesty Swagger", overlay=true)

// WaveTrend Oscillator (WT)
n1 = input.int(10, "Channel Length")
n2 = input.int(21, "Average Length")
obLevel1 = input.float(60.0, "Overbought Level 1")
obLevel2 = input.float(53.0, "Overbought Level 2")
osLevel1 = input.float(-60.0, "Oversold Level 1")
osLevel2 = input.float(-53.0, "Oversold Level 2")

ap = hlc3
esa = ta.ema(ap, n1)
d = ta.ema(math.abs(ap - esa), n1)
ci = (ap - esa) / (0.015 * d)
tci = ta.ema(ci, n2)

// VWAP
vwap = ta.vwma(close, n1)

// Signal Line
wt1 = tci
wt2 = ta.sma(wt1, 4)

// Bullish and Bearish Divergences
bullishDivergence = (ta.lowest(close, 5) > ta.lowest(close[1], 5)) and (wt1 < wt1[1]) and (close > close[1])
bearishDivergence = (ta.highest(close, 5) < ta.highest(close[1], 5)) and (wt1 > wt1[1]) and (close < close[1])

// Plot WaveTrend Oscillator
plot(wt1, title="WT1", color=color.blue)
plot(wt2, title="WT2", color=color.red)

// Plot Divergences
plotshape(series=bullishDivergence, location=location.belowbar, color=color.green, style=shape.labelup, title="Bullish Divergence")
plotshape(series=bearishDivergence, location=location.abovebar, color=color.red, style=shape.labeldown, title="Bearish Divergence")

// Risk Management Parameters
riskPercentage = input.float(1, title="Risk Percentage per Trade", minval=0.1, step=0.1) / 100
stopLossATR = input.float(1.5, title="Stop Loss ATR Multiplier", minval=0.5, step=0.1)

// ATR Calculation
atr = ta.atr(14)

// Position Size Calculation
calculatePositionSize(stopLoss) =>
    riskAmount = strategy.equity * riskPercentage
    positionSize = riskAmount / stopLoss
    positionSize

// Entry and Exit Logic with Stop Loss
if bullishDivergence
    stopLoss = low - atr * stopLossATR
    positionSize = calculatePositionSize(close - stopLoss)
    strategy.entry("Buy", strategy.long, qty=positionSize)
    strategy.exit("Sell", from_entry="Buy", stop=stopLoss)

if bearishDivergence
    strategy.close("Buy")

// Plot VWAP
plot(vwap, title="VWAP", color=color.orange)

// Background color to indicate Overbought/Oversold conditions
bgcolor(wt1 > obLevel1 ? color.new(color.red, 90) : na, title="Overbought Level 1")
bgcolor(wt1 < osLevel1 ? color.new(color.green, 90) : na, title="Oversold Level 1")
bgcolor(wt1 > obLevel2 ? color.new(color.red, 70) : na, title="Overbought Level 2")
bgcolor(wt1 < osLevel2 ? color.new(color.green, 70) : na, title="Oversold Level 2")

```

> Detail

https://www.fmz.com/strategy/452745

> Last Modified

2024-05-28 17:43:54
