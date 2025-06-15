
> Name

Simple-Combined-Strategy-Pivot-Point-SuperTrend-and-DEMA-简单组合策略枢轴点超级趋势和双重指数移动平均线

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/11572cce09be4e7ae7b.png)
[trans]
#### 概述
该策略结合了枢轴点超级趋势指标和双重指数移动平均线（DEMA）指标，通过分析价格在这两个指标之间的位置关系，判断交易信号。当价格突破枢轴点超级趋势指标并且高于DEMA指标时，产生做多信号；当价格跌破枢轴点超级趋势指标并且低于DEMA指标时，产生做空信号。该策略可以捕捉到市场的中长期趋势，同时也能够应对短期内的价格波动。

#### 策略原理
1. 计算枢轴点超级趋势指标：通过计算一定周期内的最高价和最低价的平均值作为中心点，然后根据平均真实波幅（ATR）计算上下轨，形成动态支撑和阻力位。
2. 计算DEMA指标：先计算收盘价的指数移动平均线（EMA），然后再对EMA进行一次指数移动平均，最后用两倍EMA减去DEMA得到最终的DEMA指标。
3. 产生交易信号：当收盘价突破枢轴点超级趋势上轨并且高于DEMA指标时，产生做多信号；当收盘价跌破枢轴点超级趋势下轨并且低于DEMA指标时，产生做空信号。
4. 设置止损和止盈：根据点值（Pip Value）和预设的止损点数（Stop Loss Pips）以及止盈点数（Take Profit Pips）计算出具体的止损价和止盈价。

#### 策略优势
1. 趋势跟踪能力强：枢轴点超级趋势指标可以有效地捕捉市场趋势，而DEMA指标可以消除价格噪音，提供更加平滑的趋势判断依据，两者结合可以准确把握市场主要趋势。
2. 适应性强：通过动态调整枢轴点超级趋势指标的上下轨，可以适应不同的市场波动情况，提高策略的适应性。
3. 风险控制能力强：设置了明确的止损和止盈位置，可以有效控制单笔交易的风险敞口，同时也能够及时锁定已有盈利。

#### 策略风险
1. 参数设置风险：策略的表现依赖于多个参数的设置，如枢轴点周期、ATR因子、DEMA长度等，不同参数组合可能导致策略表现差异较大，需要谨慎选择和优化。
2. 震荡市风险：在震荡市场环境下，频繁的交易信号可能导致过度交易，从而增加交易成本和滑点风险。
3. 趋势转折风险：当市场趋势发生转折时，策略可能会出现连续亏损的情况，需要结合其他分析手段及时调整策略。

#### 策略优化方向
1. 参数优化：通过对不同时间周期和交易品种进行参数优化测试，找到最佳的参数组合，提高策略的稳定性和盈利能力。
2. 信号过滤：在交易信号产生时，可以结合其他技术指标或者价格行为特征进行二次确认，提高信号的可靠性，减少虚假信号带来的损失。
3. 仓位管理：根据市场波动情况和账户风险承受能力，动态调整每笔交易的仓位大小，控制整体风险敞口。
4. 组合优化：将该策略与其他策略或者交易系统进行组合，通过分散风险和增强稳定性，提高策略的长期表现。

#### 总结
该策略通过枢轴点超级趋势指标和DEMA指标的结合，可以较好地捕捉市场趋势，同时也能够应对短期波动。策略具有趋势跟踪能力强、适应性强、风险控制能力强等优势，但同时也面临参数设置、震荡市和趋势转折等风险。通过参数优化、信号过滤、仓位管理和组合优化等手段，可以进一步提升策略的稳定性和盈利能力，更好地适应不同的市场环境。

|| 

#### Overview
This strategy combines the Pivot Point SuperTrend indicator and the Double Exponential Moving Average (DEMA) indicator to generate trading signals by analyzing the price position relative to these two indicators. When the price breaks above the Pivot Point SuperTrend indicator and is higher than the DEMA indicator, a long signal is generated; when the price breaks below the Pivot Point SuperTrend indicator and is lower than the DEMA indicator, a short signal is generated. This strategy can capture the medium to long-term market trends while also responding to short-term price fluctuations.

#### Strategy Principle
1. Calculate the Pivot Point SuperTrend indicator: The midpoint is calculated by taking the average of the highest and lowest prices over a certain period, and then the upper and lower bands are calculated based on the Average True Range (ATR), forming dynamic support and resistance levels.
2. Calculate the DEMA indicator: First, calculate the Exponential Moving Average (EMA) of the closing price, then calculate the EMA of the EMA, and finally subtract the DEMA from twice the EMA to get the final DEMA indicator.
3. Generate trading signals: When the closing price breaks above the upper band of the Pivot Point SuperTrend and is higher than the DEMA indicator, a long signal is generated; when the closing price breaks below the lower band of the Pivot Point SuperTrend and is lower than the DEMA indicator, a short signal is generated.
4. Set stop loss and take profit: Calculate the specific stop loss and take profit prices based on the pip value, preset stop loss pips, and take profit pips.

#### Strategy Advantages
1. Strong trend-following ability: The Pivot Point SuperTrend indicator can effectively capture market trends, while the DEMA indicator can eliminate price noise and provide a smoother basis for trend judgment. The combination of the two can accurately grasp the main market trends.
2. Strong adaptability: By dynamically adjusting the upper and lower bands of the Pivot Point SuperTrend indicator, the strategy can adapt to different market volatility situations, improving its adaptability.
3. Strong risk control ability: By setting clear stop loss and take profit positions, the risk exposure of a single transaction can be effectively controlled, while also timely locking in existing profits.

#### Strategy Risks
1. Parameter setting risk: The strategy's performance depends on the settings of multiple parameters, such as the pivot point period, ATR factor, DEMA length, etc. Different parameter combinations may lead to large differences in strategy performance, requiring careful selection and optimization.
2. Range-bound market risk: In a range-bound market environment, frequent trading signals may lead to overtrading, increasing transaction costs and slippage risks.
3. Trend reversal risk: When the market trend reverses, the strategy may experience consecutive losses, requiring timely adjustment of the strategy in combination with other analysis methods.

#### Strategy Optimization Directions
1. Parameter optimization: Conduct parameter optimization tests on different time periods and trading instruments to find the best parameter combination and improve the stability and profitability of the strategy.
2. Signal filtering: When trading signals are generated, they can be further confirmed in combination with other technical indicators or price behavior characteristics to improve the reliability of signals and reduce losses caused by false signals.
3. Position management: Dynamically adjust the position size of each trade according to market volatility and account risk tolerance to control overall risk exposure.
4. Portfolio optimization: Combine this strategy with other strategies or trading systems to diversify risk and enhance stability, improving the long-term performance of the strategy.

#### Summary
By combining the Pivot Point SuperTrend indicator and the DEMA indicator, this strategy can effectively capture market trends while also responding to short-term fluctuations. The strategy has advantages such as strong trend-following ability, strong adaptability, and strong risk control ability, but also faces risks such as parameter setting, range-bound markets, and trend reversals. Through parameter optimization, signal filtering, position management, and portfolio optimization, the stability and profitability of the strategy can be further improved to better adapt to different market environments.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-05-01 00:00:00
end: 2024-05-31 23:59:59
period: 4h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Simple Combined Strategy: Pivot Point SuperTrend and DEMA", overlay=true)

// Pivot Point SuperTrend settings
prd = input.int(2, title="Pivot Point Period", minval=1, maxval=50)
Factor = input.float(3.0, title="ATR Factor", minval=1, step=0.1)
Pd = input.int(10, title="ATR Period", minval=1)

// Double EMA settings
demaLength = input.int(200, title="DEMA Length", minval=1)
src = input(close, title="Source")

// Pip settings
pipValue = input.float(0.0001, title="Pip Value")
stopLossPips = input.int(15, title="Stop Loss (pips)")
takeProfitPips = input.int(35, title="Take Profit (pips)")

// Pivot Point SuperTrend Calculation
float ph = ta.pivothigh(prd, prd)
float pl = ta.pivotlow(prd, prd)
var float center = na
if not na(ph)
    center := na(center) ? ph : (center * 2 + ph) / 3
if not na(pl)
    center := na(center) ? pl : (center * 2 + pl) / 3

Up = center - (Factor * ta.atr(Pd))
Dn = center + (Factor * ta.atr(Pd))
var float TUp = na
var float TDown = na
var int Trend = na

if na(Trend)
    TUp := Up
    TDown := Dn
    Trend := close > Dn ? 1 : -1
else
    TUp := close[1] > TUp[1] ? math.max(Up, TUp[1]) : Up
    TDown := close[1] < TDown[1] ? math.min(Dn, TDown[1]) : Dn
    Trend := close > TDown[1] ? 1 : close < TUp[1] ? -1 : nz(Trend[1], 1)

Trailingsl = Trend == 1 ? TUp : TDown
linecolor = Trend == 1 ? color.lime : color.red
plot(Trailingsl, color=linecolor, linewidth=2, title="PP SuperTrend")

// Double EMA Calculation
e1 = ta.ema(src, demaLength)
e2 = ta.ema(e1, demaLength)
dema = 2 * e1 - e2
plot(dema, "DEMA", color=color.new(#43A047, 0))

// Strategy Logic
longCondition = close > Trailingsl and close > dema and strategy.position_size <= 0
shortCondition = close < Trailingsl and close < dema and strategy.position_size >= 0

// Plot signals
plotshape(series=longCondition, title="Long", location=location.belowbar, color=color.green, style=shape.labelup, text="Long")
plotshape(series=shortCondition, title="Short", location=location.abovebar, color=color.red, style=shape.labeldown, text="Short")

// Strategy Entry and Exit
if (longCondition)
    strategy.entry("Long", strategy.long, stop=close - (stopLossPips * pipValue), limit=close + (takeProfitPips * pipValue))
if (shortCondition)
    strategy.entry("Short", strategy.short, stop=close + (stopLossPips * pipValue), limit=close - (takeProfitPips * pipValue))

alertcondition(longCondition, title="Long Alert", message="Long Signal")
alertcondition(shortCondition, title="Short Alert", message="Short Signal")

```

> Detail

https://www.fmz.com/strategy/454332

> Last Modified

2024-06-17 14:49:14
