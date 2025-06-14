
> Name

基于QQE指标和RSI指标的多空信号策略-QQE-and-RSI-based-Long-Short-Signal-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1a2e17479bee128bc99.png)

[trans]
#### 概述
该策略基于QQE指标和RSI指标,通过计算RSI指标的平滑移动平均值和动态震荡幅度,构建多空信号区间。当RSI指标突破上轨时产生做多信号,突破下轨时产生做空信号。策略的主要思路是利用RSI指标的趋势特性和QQE指标的波动特性,捕捉市场的趋势变化和波动机会。

#### 策略原理
1. 计算RSI指标的平滑移动平均值RsiMa,作为判断趋势的基础。
2. 计算RSI指标的绝对偏离值AtrRsi,并计算其平滑移动平均值MaAtrRsi,作为判断波动的基础。
3. 根据QQE因子计算动态震荡幅度dar,并与RsiMa结合,构建多空信号区间longband和shortband。
4. 判断RSI指标与多空信号区间的关系,当RSI指标上穿longband时产生做多信号,下穿shortband时产生做空信号。
5. 根据多空信号进行交易,做多信号触发时开仓买入,做空信号触发时平仓。

#### 策略优势
1. 结合了RSI指标和QQE指标的特点,能够较好地捕捉市场趋势和波动机会。
2. 采用动态的震荡幅度来构建信号区间,能够自适应市场波动率的变化。
3. 平滑处理RSI指标和波动幅度,有效减少了噪音干扰和频繁交易。
4. 逻辑清晰,参数较少,适合进行进一步的优化和改进。

#### 策略风险
1. 对于震荡市场和波动率较小的市场,该策略的表现可能不够理想。
2. 缺乏明确的止损机制,在市场突然反转时可能面临较大回撤风险。
3. 参数设置对策略性能影响较大,需要根据不同市场和品种进行调优。

#### 策略优化方向
1. 引入明确的止损机制,如固定百分比止损、ATR止损等,以控制回撤风险。
2. 优化参数设置,可以通过遗传算法、网格搜索等方法寻找最优参数组合。
3. 考虑引入交易量、持仓量等其他指标,丰富交易信号,提高策略稳定性。
4. 对于震荡市场,可以考虑引入范围交易或者波段操作的逻辑,增强策略适应性。

#### 总结
该策略基于RSI指标和QQE指标构建多空信号,具有趋势捕捉和波动把握的特点。策略逻辑清晰,参数较少,适合进行进一步的优化和改进。但是策略也存在一定的风险,如回撤控制、参数设置等方面需要进一步完善。未来可以从止损机制、参数优化、信号丰富、不同市场适应性等方面对策略进行优化,以提高策略的稳健性和盈利能力。

|| 

#### Overview
This strategy is based on the QQE indicator and the RSI indicator. It calculates the smoothed moving average and dynamic oscillation range of the RSI indicator to construct long-short signal intervals. When the RSI indicator breaks through the upper rail, it generates a long signal, and when it breaks through the lower rail, it generates a short signal. The main idea of the strategy is to use the trend characteristics of the RSI indicator and the volatility characteristics of the QQE indicator to capture changes in market trends and volatility opportunities.

#### Strategy Principle
1. Calculate the smoothed moving average RsiMa of the RSI indicator as the basis for judging the trend.
2. Calculate the absolute deviation value AtrRsi of the RSI indicator and its smoothed moving average MaAtrRsi as the basis for judging volatility.
3. Calculate the dynamic oscillation range dar according to the QQE factor, and combine it with RsiMa to construct the long-short signal intervals longband and shortband.
4. Judge the relationship between the RSI indicator and the long-short signal intervals. When the RSI indicator crosses above longband, it generates a long signal, and when it crosses below shortband, it generates a short signal.
5. Conduct trades according to the long-short signals. When a long signal is triggered, open a long position, and when a short signal is triggered, close the position.

#### Strategy Advantages
1. It combines the characteristics of the RSI indicator and the QQE indicator, which can better capture market trends and volatility opportunities.
2. It uses dynamic oscillation range to construct signal intervals, which can adapt to changes in market volatility.
3. It smooths the RSI indicator and volatility range, effectively reducing noise interference and frequent trading.
4. The logic is clear, with fewer parameters, and is suitable for further optimization and improvement.

#### Strategy Risks
1. For volatile markets and markets with low volatility, the performance of this strategy may not be ideal.
2. It lacks a clear stop-loss mechanism and may face greater drawdown risk when the market suddenly reverses.
3. Parameter settings have a greater impact on strategy performance and need to be tuned according to different markets and varieties.

#### Strategy Optimization Directions
1. Introduce clear stop-loss mechanisms, such as fixed percentage stop-loss, ATR stop-loss, etc., to control drawdown risk.
2. Optimize parameter settings. The optimal parameter combination can be found through genetic algorithms, grid search and other methods.
3. Consider introducing other indicators such as trading volume and position volume to enrich trading signals and improve strategy stability.
4. For volatile markets, consider introducing range trading or swing trading logic to enhance the adaptability of the strategy.

#### Summary
This strategy constructs long-short signals based on the RSI indicator and the QQE indicator, and has the characteristics of trend capture and volatility grasp. The strategy logic is clear, with fewer parameters, and is suitable for further optimization and improvement. However, the strategy also has certain risks, such as drawdown control and parameter setting, which need to be further improved. In the future, the strategy can be optimized from aspects such as stop-loss mechanism, parameter optimization, signal enrichment, and adaptability to different markets, so as to improve the robustness and profitability of the strategy.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-05-21 00:00:00
end: 2024-05-26 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Binance","currency":"BTC_USDT"}]
*/

//@version=4
// modified by swigle
// thanks colinmck

strategy("QQE signals bot", overlay=true)


RSI_Period = input(14, title='RSI Length')
SF = input(5, title='RSI Smoothing')
QQE = input(4.236, title='Fast QQE Factor')
ThreshHold = input(10, title="Thresh-hold")

src = close
Wilders_Period = RSI_Period * 2 - 1

Rsi = rsi(src, RSI_Period)
RsiMa = ema(Rsi, SF)
AtrRsi = abs(RsiMa[1] - RsiMa)
MaAtrRsi = ema(AtrRsi, Wilders_Period)
dar = ema(MaAtrRsi, Wilders_Period) * QQE

longband = 0.0
shortband = 0.0
trend = 0

DeltaFastAtrRsi = dar
RSIndex = RsiMa
newshortband = RSIndex + DeltaFastAtrRsi
newlongband = RSIndex - DeltaFastAtrRsi
longband := RSIndex[1] > longband[1] and RSIndex > longband[1] ? max(longband[1], newlongband) : newlongband
shortband := RSIndex[1] < shortband[1] and RSIndex < shortband[1] ? min(shortband[1], newshortband) : newshortband
cross_1 = cross(longband[1], RSIndex)
trend := cross(RSIndex, shortband[1]) ? 1 : cross_1 ? -1 : nz(trend[1], 1)
FastAtrRsiTL = trend == 1 ? longband : shortband

// Find all the QQE Crosses

QQExlong = 0
QQExlong := nz(QQExlong[1])
QQExshort = 0
QQExshort := nz(QQExshort[1])
QQExlong := FastAtrRsiTL < RSIndex ? QQExlong + 1 : 0
QQExshort := FastAtrRsiTL > RSIndex ? QQExshort + 1 : 0

//Conditions

qqeLong = QQExlong == 1 ? FastAtrRsiTL[1] - 50 : na
qqeShort = QQExshort == 1 ? FastAtrRsiTL[1] - 50 : na

// Plotting

plotshape(qqeLong, title="QQE long", text="Long", textcolor=color.white, style=shape.labelup, location=location.belowbar, color=color.green, size=size.tiny)
plotshape(qqeShort, title="QQE short", text="Short", textcolor=color.white, style=shape.labeldown, location=location.abovebar, color=color.red, size=size.tiny)

// trade

//if qqeLong > 0
strategy.entry("buy long", strategy.long, 100, when=qqeLong)
    
if qqeShort > 0
    strategy.close("buy long")
    // strategy.exit("close_position", "buy long", loss=1000)
    // strategy.entry("sell", strategy.short, 1, when=strategy.position_size > 0)
    

```

> Detail

https://www.fmz.com/strategy/452613

> Last Modified

2024-05-27 15:17:45
