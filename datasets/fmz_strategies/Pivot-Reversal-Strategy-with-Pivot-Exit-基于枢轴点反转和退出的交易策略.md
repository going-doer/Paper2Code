
> Name

Pivot-Reversal-Strategy-with-Pivot-Exit-基于枢轴点反转和退出的交易策略

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/175c47473db833c3289.png)

[trans]
#### 概述
该策略基于枢轴点(Pivot Points)来识别市场反转点,并以此为基础进行交易。当市场在左侧4根K线内出现枢轴点高点(Pivot High)时,策略开仓做多;当市场在左侧4根K线内出现枢轴点低点(Pivot Low)时,策略开仓做空。策略的止损设置在开仓价格的上下一个最小价格变动单位(syminfo.mintick)。策略的退出条件有两个:1)当出现下一个相反方向的枢轴点时平仓;2)当浮亏达到30%时平仓。

#### 策略原理
1. 使用ta.pivothigh()和ta.pivotlow()函数计算左侧4根K线、右侧2根K线范围内的枢轴点高点(swh)和低点(swl)。
2. 当枢轴点高点存在(swh_cond)时,更新最高价(hprice),同时满足当前最高价高于之前的最高价时,开启做多进场条件(le)。
3. 当做多进场条件(le)成立时,在枢轴点高点上方一个最小变动单位(syminfo.mintick)的位置开仓做多。
4. 与做多类似,当枢轴点低点存在(swl_cond)时,更新最低价(lprice),同时满足当前最低价低于之前的最低价时,开启做空进场条件(se)。
5. 当做空进场条件(se)成立时,在枢轴点低点下方一个最小变动单位(syminfo.mintick)的位置开仓做空。
6. 在exitAtNextPivot()函数中,若持有多头仓位,则在下一个枢轴点低点下方一个最小变动单位止损;若持有空头仓位,则在下一个枢轴点高点上方一个最小变动单位止损。
7. 在exitIfProfitLessThanThirtyPercent()函数中,计算多头和空头仓位的盈亏百分比,若亏损超过30%,则平仓。

#### 策略优势
1. 枢轴点能较好地反映市场的支撑和压力位置,是一种常用的技术分析指标,具有一定的市场认可度。
2. 在枢轴点突破时进场,能够捕捉到市场反转的机会。
3. 设置了两个退出条件,一个是基于下一个相反方向枢轴点的技术性退出,另一个是基于亏损百分比的风控性退出,能够一定程度上控制策略回撤。

#### 策略风险
1. 枢轴点指标本身也存在一定的滞后性和信号频繁的问题,在震荡市中可能表现不佳。
2. 固定的4根K线和2根K线计算参数可能并不适用于所有的市场状态,缺乏一定的自适应性和灵活性。
3. 止损位置距离进场价格较近(一个最小变动单位),在市场剧烈波动时可能会被抛离。
4. 亏损30%才止损的设置可能过于宽松,回撤幅度偏大。

#### 策略优化方向
1. 可以尝试使用其他类型的枢轴点指标,如因子型枢轴点、加权枢轴点等,以提高指标灵敏度和实时性。
2. 左右K线数可以作为输入参数,通过参数优化来寻找最佳值。
3. 止损位置可以优化为ATR或者百分比止损,前者能够随着市场波动率的变化自适应调整,后者可以限制风险在可控范围内。
4. 30%的亏损平仓条件可以收紧,从而降低策略回撤。此外,也可以增加盈利百分比平仓条件,做到浮盈与浮亏同时管理。
5. 可以在枢轴点突破基础上叠加其他过滤条件,如趋势过滤、波动率过滤等,提高信号质量。

#### 总结
该策略基于枢轴点指标构建了一个双向交易系统,通过在枢轴点高点做多、低点做空来捕捉市场反转机会。策略具有一定的理论基础和实践价值,但是由于枢轴点指标本身的局限性,策略在实际运行中可能面临一些风险和挑战。通过优化枢轴点指标类型、参数、过滤条件、止盈止损等,有望进一步提升该策略的稳健性和盈利能力。

|| 

#### Overview
This strategy uses Pivot Points to identify market reversal points and make trading decisions based on them. When a Pivot High is formed within the last 4 candles on the left, the strategy enters a long position; when a Pivot Low is formed within the last 4 candles on the left, the strategy enters a short position. The stop loss is set at one tick size (syminfo.mintick) above or below the entry price. There are two exit conditions: 1) exit when the next opposite pivot point appears; 2) exit when the floating loss reaches 30%.

#### Strategy Principles
1. Use ta.pivothigh() and ta.pivotlow() functions to calculate the Pivot High (swh) and Pivot Low (swl) within the range of 4 candles on the left and 2 candles on the right.
2. When a Pivot High exists (swh_cond), update the highest price (hprice), and when the current high is higher than the previous high, enable the long entry condition (le).
3. When the long entry condition (le) is met, enter a long position at one tick size (syminfo.mintick) above the Pivot High.
4. Similarly, when a Pivot Low exists (swl_cond), update the lowest price (lprice), and when the current low is lower than the previous low, enable the short entry condition (se).
5. When the short entry condition (se) is met, enter a short position at one tick size (syminfo.mintick) below the Pivot Low.
6. In the exitAtNextPivot() function, if holding a long position, set the stop loss at one tick size below the next Pivot Low; if holding a short position, set the stop loss at one tick size above the next Pivot High.
7. In the exitIfProfitLessThanThirtyPercent() function, calculate the profit and loss percentage for long and short positions, and close the position if the loss exceeds 30%.

#### Strategy Advantages
1. Pivot Points can well reflect the support and resistance levels in the market, and are a commonly used technical analysis indicator with certain market recognition.
2. Entering at the breakout of Pivot Points can capture market reversal opportunities.
3. Two exit conditions are set, one based on the next opposite pivot point for technical exit, and the other based on loss percentage for risk control exit, which can control the strategy drawdown to a certain extent.

#### Strategy Risks
1. The Pivot Point indicator itself has certain lag and frequent signal issues, and may not perform well in a fluctuating market.
2. The fixed calculation parameters of 4 candles and 2 candles may not be applicable to all market conditions, lacking certain adaptability and flexibility.
3. The stop loss is set close to the entry price (one tick size), which may be thrown away during violent market fluctuations.
4. The 30% loss stop setting may be too loose, with a large drawdown amplitude.

#### Strategy Optimization Directions
1. Try using other types of Pivot Point indicators, such as Factor Pivot Points, Weighted Pivot Points, etc., to improve the sensitivity and timeliness of the indicators.
2. The number of left and right candles can be used as input parameters, and the optimal values can be found through parameter optimization.
3. The stop loss position can be optimized to ATR or percentage stop loss. The former can adaptively adjust with changes in market volatility, while the latter can limit risks within a controllable range.
4. The 30% loss stop condition can be tightened to reduce strategy drawdown. In addition, a profit percentage stop condition can be added to manage both floating profits and losses.
5. Other filtering conditions, such as trend filtering and volatility filtering, can be superimposed on the basis of Pivot Point breakout to improve signal quality.

#### Summary
This strategy builds a bi-directional trading system based on the Pivot Point indicator, capturing market reversal opportunities by going long at Pivot Highs and short at Pivot Lows. The strategy has certain theoretical basis and practical value, but due to the limitations of the Pivot Point indicator itself, the strategy may face some risks and challenges in actual operation. By optimizing the Pivot Point indicator type, parameters, filtering conditions, stop loss and profit taking, etc., it is expected to further improve the robustness and profitability of the strategy.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_1|4|leftBars|
|v_input_2|2|rightBars|


> Source (PineScript)

``` pinescript
/*backtest
start: 2023-04-24 00:00:00
end: 2024-04-29 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Pivot Reversal Strategy with Pivot Exit", overlay=true)

leftBars = input(4)
rightBars = input(2)

var float dailyEquity = na

// Reset equity to $10,000 at the beginning of each day
isNewDay = ta.change(time("D")) != 0
if (isNewDay)
    dailyEquity := 10000

// Calculate pivot highs and lows
swh = ta.pivothigh(leftBars, rightBars)
swl = ta.pivotlow(leftBars, rightBars)

// Define long entry condition
swh_cond = not na(swh)
hprice = 0.0
hprice := swh_cond ? swh : hprice[1]
le = false
le := swh_cond ? true : (le[1] and high > hprice ? false : le[1])

// Enter long position if long entry condition is met
if (le)
    strategy.entry("PivRevLE", strategy.long, comment="EnterLong", stop=hprice + syminfo.mintick)

// Define short entry condition
swl_cond = not na(swl)
lprice = 0.0
lprice := swl_cond ? swl : lprice[1]
se = false
se := swl_cond ? true : (se[1] and low < lprice ? false : se[1])

// Enter short position if short entry condition is met
if (se)
    strategy.entry("PivRevSE", strategy.short, comment="EnterShort", stop=lprice - syminfo.mintick)

// Exit condition: Exit at the next pivot point
exitAtNextPivot() =>
    if strategy.opentrades > 0
        if strategy.position_size > 0
            // Exiting long position at next pivot low
            if not na(swl)
                strategy.exit("ExitLong", "PivRevLE", stop=swl - syminfo.mintick)
        else
            // Exiting short position at next pivot high
            if not na(swh)
                strategy.exit("ExitShort", "PivRevSE", stop=swh + syminfo.mintick)

// Call exitAtNextPivot function
exitAtNextPivot()

// Exit condition: Exit if profit is less than 30%
exitIfProfitLessThanThirtyPercent() =>
    if strategy.opentrades > 0
        if strategy.position_size > 0
            // Calculate profit percentage for long position
            profit_percentage_long = (close - strategy.position_avg_price) / strategy.position_avg_price * 100
            // Exiting long position if profit is less than 30%
            if profit_percentage_long < -30
                strategy.close("PivRevLE")
        else
            // Calculate profit percentage for short position
            profit_percentage_short = (strategy.position_avg_price - close) / strategy.position_avg_price * 100
            // Exiting short position if profit is less than 30%
            if profit_percentage_short < -30
                strategy.close("PivRevSE")

// Call exitIfProfitLessThanThirtyPercent function
exitIfProfitLessThanThirtyPercent()

```

> Detail

https://www.fmz.com/strategy/449944

> Last Modified

2024-04-30 15:57:54
