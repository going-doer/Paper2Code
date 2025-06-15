
> Name

趋势追踪优化双T3指标策略-Optimized-Dual-T3-Trend-Tracking-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/9e1ee76da1af0d6258.png)

[trans]
#### 概述
该策略是一个基于Tillson T3指标和双优化趋势追踪器(TOTT)的趋势跟踪系统。它通过结合动量振荡器Williams %R来优化交易信号的生成。该策略采用分离的买入和卖出参数设置,能够根据不同市场条件灵活调整敏感度,提高策略的适应性。

#### 策略原理
策略主要由三个核心组件构成:
1. Tillson T3指标 - 这是一个经过优化的指数移动平均线(EMA)变体,通过多重EMA加权计算产生更平滑的趋势线。
2. 双优化趋势追踪器(TOTT) - 基于价格行为和波动系数自适应调整的趋势跟踪工具,分别计算买入和卖出条件下的上下轨。
3. Williams %R指标 - 用于识别超买超卖状态的动量振荡器。

交易信号生成逻辑:
- 买入条件:当T3线突破TOTT上轨且Williams %R大于-20(超卖)时
- 卖出条件:当T3线跌破TOTT下轨且Williams %R大于-70时

#### 策略优势
1. 信号稳定性强 - 通过T3指标的多重平滑处理,有效降低假突破风险
2. 适应性好 - 买卖参数分离设计允许针对不同市场条件独立优化
3. 风险控制完善 - 集成Williams %R作为二次确认,提高交易可靠性
4. 可视化清晰 - 策略提供全面的图表可视化支持,便于分析判断

#### 策略风险
1. 趋势反转滞后 - T3指标的多重平滑可能导致信号延迟
2. 震荡市不适用 - 在横盘整理阶段可能产生过多交易信号
3. 参数敏感度高 - 需要针对不同市场环境频繁调整参数

风险控制建议:
- 引入止损机制
- 设置交易量限制
- 增加趋势确认过滤器

#### 策略优化方向
1. 动态参数优化 - 开发自适应参数调整机制
2. 增加市场环境识别 - 引入趋势强度指标
3. 完善风险管理 - 添加动态止损止盈
4. 增强信号过滤 - 整合更多技术指标确认

#### 总结
这是一个结构完整、逻辑清晰的趋势跟踪策略。通过T3指标和TOTT的结合,配合Williams %R的过滤,在趋势市场中表现出色。虽然存在一定的滞后性,但通过参数优化和风险管理的改进,该策略具有良好的实用价值和扩展空间。

|| 

#### Overview
This strategy is a trend following system based on the Tillson T3 indicator and Twin Optimized Trend Tracker (TOTT). It optimizes trade signal generation by incorporating the Williams %R momentum oscillator. The strategy employs separate buy and sell parameter settings, enabling flexible sensitivity adjustment for different market conditions.

#### Strategy Principles
The strategy consists of three core components:
1. Tillson T3 Indicator - An optimized variant of the Exponential Moving Average (EMA) that produces a smoother trend line through multiple weighted EMA calculations.
2. Twin Optimized Trend Tracker (TOTT) - An adaptive trend following tool that adjusts based on price action and volatility coefficient, calculating upper and lower bands for buy and sell conditions.
3. Williams %R Indicator - A momentum oscillator used to identify overbought and oversold conditions.

Signal generation logic:
- Buy condition: When T3 line crosses above TOTT upper band and Williams %R is above -20 (oversold)
- Sell condition: When T3 line crosses below TOTT lower band and Williams %R is above -70

#### Strategy Advantages
1. Strong signal stability - Effectively reduces false breakout risks through T3's multiple smoothing
2. Good adaptability - Separate buy/sell parameters allow independent optimization for different market conditions
3. Comprehensive risk control - Integrates Williams %R as secondary confirmation
4. Clear visualization - Provides comprehensive chart visualization support

#### Strategy Risks
1. Trend reversal lag - T3's multiple smoothing may cause signal delays
2. Unsuitable for ranging markets - May generate excessive signals during consolidation
3. High parameter sensitivity - Requires frequent adjustment for different market environments

Risk control suggestions:
- Implement stop-loss mechanisms
- Set trading volume limits
- Add trend confirmation filters

#### Optimization Directions
1. Dynamic parameter optimization - Develop adaptive parameter adjustment mechanisms
2. Enhanced market environment recognition - Introduce trend strength indicators
3. Improved risk management - Add dynamic stop-loss and take-profit
4. Enhanced signal filtering - Integrate additional technical indicators

#### Summary
This is a well-structured trend following strategy with clear logic. Through the combination of T3 indicator and TOTT, coupled with Williams %R filtering, it performs excellently in trending markets. While there is some inherent lag, the strategy shows good practical value and room for expansion through parameter optimization and risk management improvements.

[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2019-12-23 08:00:00
end: 2025-01-15 08:00:00
period: 1d
basePeriod: 1d
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT","balance":49999}]
*/

//@version=6
strategy("FON60DK by leventsah", overlay=true)

// Girdi AL
t3_length = input.int(5, title="Tillson Per AL", minval=1)
t3_opt = input.float(0.1, title="Tillson Opt AL", step=0.1, minval=0)
tott_length = input.int(5, title="TOTT Per AL", minval=1)
tott_opt = input.float(0.1, title="TOTT Opt AL", step=0.1, minval=0)
tott_coeff = input.float(0.006, title="TOTT Coeff AL", step=0.001, minval=0)

//GİRDİ SAT
t3_lengthSAT = input.int(5, title="Tillson Per SAT", minval=1)
t3_optSAT = input.float(0.1, title="Tillson Opt SAT", step=0.1, minval=0)
tott_lengthSAT = input.int(5, title="TOTT Per SAT", minval=1)
tott_opt_SAT = input.float(0.1, title="TOTT Opt SAT", step=0.1, minval=0)
tott_coeff_SAT = input.float(0.006, title="TOTT Coeff SAT", step=0.001, minval=0)

william_length = input.int(3, title="William %R Periyodu", minval=1)

// Tillson T3 AL
t3(src, length, opt) =>
    k = 2 / (length + 1)
    ema1 = ta.ema(src, length)
    ema2 = ta.ema(ema1, length)
    ema3 = ta.ema(ema2, length)
    ema4 = ta.ema(ema3, length)
    c1 = -opt * opt * opt
    c2 = 3 * opt * opt + 3 * opt * opt * opt
    c3 = -6 * opt * opt - 3 * opt - 3 * opt * opt * opt
    c4 = 1 + 3 * opt + opt * opt * opt + 3 * opt * opt
    t3_val = c1 * ema4 + c2 * ema3 + c3 * ema2 + c4 * ema1
    t3_val

t3_value = t3(close, t3_length, t3_opt)
t3_valueSAT = t3(close, t3_lengthSAT, t3_optSAT)


// TOTT hesaplaması (Twin Optimized Trend Tracker)
Var_Func(src, length) =>
    valpha = 2 / (length + 1)
    vud1 = math.max(src - src[1], 0)
    vdd1 = math.max(src[1] - src, 0)
    vUD = math.sum(vud1, 9)
    vDD = math.sum(vdd1, 9)
    vCMO = (vUD - vDD) / (vUD + vDD)
    var float VAR = na
    VAR := valpha * math.abs(vCMO) * src + (1 - valpha * math.abs(vCMO)) * nz(VAR[1], src)
    VAR

VAR = Var_Func(close, tott_length)
VAR_SAT = Var_Func(close, tott_lengthSAT)

//LONG 
MAvg = VAR
fark = MAvg * tott_opt * 0.01
longStop = MAvg - fark
longStopPrev = nz(longStop[1], longStop)
longStop := MAvg > longStopPrev ? math.max(longStop, longStopPrev) : longStop
shortStop = MAvg + fark
shortStopPrev = nz(shortStop[1], shortStop)
shortStop := MAvg < shortStopPrev ? math.min(shortStop, shortStopPrev) : shortStop
dir = 1
dir := nz(dir[1], dir)
dir := dir == -1 and MAvg > shortStopPrev ? 1 : dir == 1 and MAvg < longStopPrev ? -1 : dir
MT = dir == 1 ? longStop : shortStop
OTT = MAvg > MT ? MT * (200 + tott_opt) / 200 : MT * (200 - tott_opt) / 200
OTTup = OTT * (1 + tott_coeff)
OTTdn = OTT * (1 - tott_coeff)

//CLOSE
MAvgS = VAR_SAT
farkS = MAvgS * tott_opt_SAT * 0.01
longStopS = MAvgS - farkS
longStopPrevS = nz(longStopS[1], longStopS)
longStopS := MAvgS > longStopPrevS ? math.max(longStopS, longStopPrevS) : longStopS
shortStopS = MAvgS + farkS
shortStopPrevS = nz(shortStopS[1], shortStopS)
shortStopS := MAvgS < shortStopPrevS ? math.min(shortStopS, shortStopPrevS) : shortStopS
dirS = 1
dirS := nz(dirS[1], dirS)
dirS := dirS == -1 and MAvgS > shortStopPrevS ? 1 : dirS == 1 and MAvgS < longStopPrevS ? -1 : dirS
MTS = dirS == 1 ? longStopS : shortStopS
OTTS = MAvgS > MTS ? MTS * (200 + tott_opt_SAT) / 200 : MTS * (200 - tott_opt_SAT) / 200
OTTupS = OTTS * (1 + tott_coeff_SAT)
OTTdnS = OTTS * (1 - tott_coeff_SAT)

// Calculation of Williams %R
williamsR = -100 * (ta.highest(high, william_length) - close) / (ta.highest(high, william_length) - ta.lowest(low, william_length))

// Alım koşulu
longCondition = (t3_value > OTTup) and (williamsR > -20)

// Short koşulu (long pozisyonunu kapatmak için)
shortCondition = (t3_valueSAT < OTTdnS) and (williamsR > -70)

// Alım pozisyonu açma
if (longCondition)
    strategy.entry("Long", strategy.long)

// Short koşulu sağlandığında long pozisyonunu kapama
if (shortCondition)
    strategy.close("Long")


// Alım pozisyonu boyunca barları yeşil yapma
barcolor(strategy.position_size > 0 ? color.green : na)

// Grafikte göstergeleri çizme
plot(t3_value, color=color.blue, linewidth=1, title="Tillson AL")
plot(OTTup, color=color.green, linewidth=1, title="TOTT Up AL")
plot(OTTdn, color=color.red, linewidth=1, title="TOTT Down AL")

// Grafikte göstergeleri çizme
plot(t3_valueSAT, color=color.blue, linewidth=1, title="Tillson SAT")
plot(OTTupS, color=color.green, linewidth=1, title="TOTT Up SAT")
plot(OTTdnS, color=color.red, linewidth=1, title="TOTT Down SAT")

```

> Detail

https://www.fmz.com/strategy/478690

> Last Modified

2025-01-17 14:29:51
