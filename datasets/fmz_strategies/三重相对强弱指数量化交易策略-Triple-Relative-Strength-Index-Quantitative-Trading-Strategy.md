
> Name

三重相对强弱指数量化交易策略-Triple-Relative-Strength-Index-Quantitative-Trading-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1298c4ef5259bb88875.png)
[trans]
#### 概述
该策略主要利用相对强弱指数(RSI)来判断市场超买超卖情况,结合价格在200日简单移动平均线(SMA)之上作为趋势过滤条件,以此来决定是否进场交易。该策略通过三重RSI指标共同构建开仓条件,只有当短期RSI小于35且连续三个周期呈下降趋势,同时第三周期RSI小于60,且当前收盘价在200日SMA之上时,才会做多。平仓条件为RSI上穿50。

#### 策略原理
1. 计算指定周期的RSI指标
2. 判断是否满足以下开仓条件:
   - 当前RSI小于35
   - 当前RSI小于前一周期RSI,前一周期RSI小于前二周期RSI,前二周期RSI小于前三周期RSI
   - 前三周期RSI小于60
   - 当前收盘价大于200日SMA
3. 若同时满足上述四个条件,则开仓做多
4. 持仓过程中,若RSI上穿50,则平仓
5. 重复步骤2-4,进行下一次交易

#### 策略优势
1. 通过RSI判断超买超卖,在超卖区域开仓,能够捕捉到市场反转机会
2. 通过三重RSI共同构建开仓信号,降低了假信号概率,提高了信号可靠性
3. 加入价格在200日均线之上作为趋势条件,避免在下跌趋势中交易
4. 平仓条件简单明了,能够及时兑现利润
5. 策略逻辑清晰,易于理解和实现

#### 策略风险
1. RSI指标存在信号滞后性,可能错过最佳开仓时机
2. 开仓条件相对严格,交易频率较低,可能错失部分行情
3. 对于震荡市可能表现不佳,陷入频繁开平仓
4. 策略只能捕捉到单边上涨行情,对于趋势反转后的下跌行情无法把握

#### 策略优化方向
1. 可以考虑加入移动止损或固定止损,控制单笔交易风险
2. 研究RSI与其他辅助指标的结合,提高开平仓信号的可靠性和及时性
3. 对开仓条件进行优化,在保证信号可靠性的同时提高交易频率
4. 引入仓位管理,根据市场趋势强度和波动率动态调整仓位
5. 考虑短线和中线结合,开发出适应不同市场状态的策略版本
   
#### 总结
该策略通过三重RSI构建开仓条件,结合价格在长期均线之上作为趋势过滤,以此捕捉超卖反转行情。策略逻辑简单明了,易于实现和优化。但是策略也存在信号滞后、交易频率低、只能捕捉单边行情等风险和不足,需要在实际应用中不断调试和改进。通过引入止损止盈、仓位管理、结合其他指标等方法,可以进一步提升策略的稳定性和收益性。

|| 

#### Overview
This strategy mainly uses the Relative Strength Index (RSI) to determine overbought and oversold conditions in the market, combined with the price above the 200-day Simple Moving Average (SMA) as a trend filter, to decide whether to enter a trade. The strategy constructs entry conditions through three RSI indicators. Only when the short-term RSI is below 35 and shows a downward trend for three consecutive periods, while the third-period RSI is below 60, and the current closing price is above the 200-day SMA, will it go long. The exit condition is when the RSI crosses above 50.

#### Strategy Principle
1. Calculate the RSI indicator for the specified period
2. Determine if the following entry conditions are met:
   - Current RSI is below 35
   - Current RSI is lower than the previous period RSI, previous period RSI is lower than the second previous period RSI, second previous period RSI is lower than the third previous period RSI
   - The third previous period RSI is below 60
   - Current closing price is above the 200-day SMA
3. If all four conditions above are met simultaneously, open a long position
4. During the holding period, if the RSI crosses above 50, close the position
5. Repeat steps 2-4 for the next trade

#### Strategy Advantages
1. By using RSI to determine overbought and oversold conditions and entering positions in the oversold area, it can capture market reversal opportunities
2. By constructing entry signals with three RSIs together, it reduces the probability of false signals and improves signal reliability
3. Adding the price above the 200-day moving average as a trend condition avoids trading in a downtrend
4. The exit condition is simple and clear, allowing for timely profit realization
5. The strategy logic is clear and easy to understand and implement

#### Strategy Risks
1. The RSI indicator has a signal lag, which may miss the best entry timing
2. The entry conditions are relatively strict, resulting in low trading frequency and potentially missing some market movements
3. It may not perform well in choppy markets, getting caught in frequent entries and exits
4. The strategy can only capture unilateral uptrends and cannot grasp downtrends after trend reversals

#### Strategy Optimization Directions
1. Consider adding a trailing stop or fixed stop loss to control single trade risk
2. Study the combination of RSI with other auxiliary indicators to improve the reliability and timeliness of entry and exit signals 
3. Optimize entry conditions to improve trading frequency while ensuring signal reliability
4. Introduce position management to dynamically adjust positions based on trend strength and volatility
5. Consider combining short-term and medium-term to develop strategy versions suitable for different market conditions
   
#### Summary
This strategy constructs entry conditions through a triple RSI, combined with the price above the long-term moving average as a trend filter, to capture oversold reversal setups. The strategy logic is simple and clear, easy to implement and optimize. However, the strategy also has risks and shortcomings such as signal lag, low trading frequency, and only being able to capture unilateral market moves. It needs continuous debugging and improvement in actual application. By introducing stop loss and profit taking, position management, combining with other indicators and other methods, the stability and profitability of the strategy can be further improved.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-05-15 00:00:00
end: 2024-05-14 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
//@author Honestcowboy
//
strategy("Triple RSI [Honestcowboy]" )

  
// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ >>
// ---------> User Inputs <----------- >>
// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ >>

rsiLengthInput = input.int(5, minval=1, title="RSI Length", group="RSI Settings")
rsiSourceInput = input.source(close, "Source", group="RSI Settings")

// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ >>
// ---------> VARIABLE CALCULATIONS <----------- >>
// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ >>

up = ta.rma(math.max(ta.change(rsiSourceInput), 0), rsiLengthInput)
down = ta.rma(-math.min(ta.change(rsiSourceInput), 0), rsiLengthInput)
rsi = down == 0 ? 100 : up == 0 ? 0 : 100 - (100 / (1 + up / down))

// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ >>
// ---------> CONDITIONALS <----------- >>
// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ >>

rule1   = rsi<35
rule2   = rsi<rsi[1] and rsi[1]<rsi[2] and rsi[2]<rsi[3]
rule3   = rsi[3]<60
rule4   = close>ta.sma(close, 200)

longCondition = rule1 and rule2 and rule3 and rule4
closeCondition = rsi>50

// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ >>
// ---------> GRAPHICAL DISPLAY <----------- >>
// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ >>

hline(30, title="Long Condition Line")
hline(50, title="Exit Condition Line")
plot(rsi)
plotshape(longCondition ? rsi-3 : na, title="Long Condition", style=shape.triangleup, color=color.lime, location=location.absolute)
plotshape(closeCondition and rsi[1]<50? rsi+3 : na, title="Exit Condition", style=shape.triangledown, color=#e60000, location=location.absolute)

// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ >>
// ---------> AUTOMATION AND BACKTESTING <----------- >>
// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ >>

if longCondition and strategy.position_size==0
    strategy.entry("LONG", strategy.long)
if closeCondition
    strategy.close("LONG")
```

> Detail

https://www.fmz.com/strategy/451487

> Last Modified

2024-05-15 10:23:08
