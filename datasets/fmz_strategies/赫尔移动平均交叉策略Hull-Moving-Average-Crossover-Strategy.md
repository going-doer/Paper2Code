
> Name

赫尔移动平均交叉策略Hull-Moving-Average-Crossover-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/b2426d6835d2b25348.png)
[trans]

## 策略概述

该策略基于赫尔移动平均线(Hull Moving Average, HMA)的交叉信号进行交易。赫尔移动平均线是一种旨在减少移动平均线滞后的技术指标,由Alan Hull开发。该策略使用两条不同周期的HMA线,当较短周期的HMA从下向上穿过较长周期的HMA时产生买入信号,反之则产生卖出信号。

## 策略原理

1. 计算赫尔移动平均线(HMA)

HMA的计算过程如下:
- 先计算价格的加权移动平均线(WMA),周期为输入参数length的一半
- 再计算该WMA的WMA,周期为length
- 使用公式: HMA = 2 * WMA(length/2) - WMA(length)
- 对上述结果再次计算WMA,周期为length的平方根,得到最终的HMA

2. 产生交易信号

- 当收盘价上穿HMA时,产生买入信号
- 当收盘价下穿HMA时,产生卖出信号

3. 根据交易信号执行交易

- 买入信号: 开多头仓位
- 卖出信号: 开空头仓位

## 策略优势

1. 赫尔移动平均线相比简单移动平均线和加权移动平均线有更小的滞后性,因此能更快地对价格变化做出反应,提高了策略的灵敏度。

2. 通过两条不同周期的HMA线交叉来产生信号,能有效过滤掉一些噪音和虚假信号,提高信号的可靠性。

3. 参数可调,通过调整HMA的周期参数,可以适应不同的市场和交易品种。

## 策略风险

1. 赫尔移动平均线是一个滞后指标,在趋势反转的早期阶段可能会发出错误信号。

2. 参数选择不当可能导致策略表现不佳。周期选得过长会使策略反应迟钝,周期选得过短则可能导致过多的虚假信号。

3. 像所有基于单一指标的策略一样,该策略在震荡市可能表现不佳,产生较多的虚假信号和亏损交易。

## 优化方向

1. 可以考虑引入其他技术指标或基本面因素作为过滤条件,例如交易量、趋势指标等,以进一步确认HMA交叉信号的有效性。

2. 对于参数优化,可以使用遗传算法、网格搜索等方法在历史数据上进行参数寻优,找到最适合当前市场的参数组合。

3. 在策略中加入止损和止盈机制,控制单次交易的风险和收益。

4. 考虑市场的趋势性,可以通过引入市场趋势判断指标,在趋势型市场中交易,回避震荡市。

## 总结

赫尔移动平均交叉策略是一个简单易用的交易策略,通过HMA的快速响应特性,可以比较及时地捕捉到价格的变化。但像所有策略一样,它也有其局限性,表现会受到市场类型、参数选择的影响。在实际应用中,可以将其与其他分析方法相结合,对信号进行进一步确认。同时,合理的风控措施,如止损止盈、仓位管理等,也是策略稳健运行不可或缺的部分。

|| 

## Overview

This strategy is based on the crossover signals of the Hull Moving Average (HMA). The Hull Moving Average is a technical indicator developed by Alan Hull, which aims to reduce the lag of traditional moving averages. The strategy uses two HMAs with different periods. A buy signal is generated when the shorter-period HMA crosses above the longer-period HMA, and a sell signal is generated when the opposite occurs.

## Principle

1. Calculating the Hull Moving Average (HMA)

The HMA is calculated as follows:
- First, calculate the Weighted Moving Average (WMA) of the price with a period of half the input parameter 'length'
- Then, calculate the WMA of the previous WMA with a period of 'length'
- Use the formula: HMA = 2 * WMA(length/2) - WMA(length)
- Calculate the WMA of the above result again with a period of the square root of 'length' to get the final HMA

2. Generating Trading Signals

- When the closing price crosses above the HMA, a buy signal is generated
- When the closing price crosses below the HMA, a sell signal is generated

3. Executing Trades Based on Signals

- Buy signal: Open a long position
- Sell signal: Open a short position

## Advantages

1. Compared to Simple Moving Average and Weighted Moving Average, the Hull Moving Average has less lag, thus it can react faster to price changes, improving the strategy's responsiveness.

2. By using the crossover of two HMAs with different periods to generate signals, a lot of noise and false signals can be effectively filtered out, enhancing the reliability of the signals.

3. The parameters are adjustable. By tuning the period parameters of the HMAs, the strategy can be adapted to different markets and trading instruments.

## Risks

1. The Hull Moving Average is a lagging indicator, which may generate false signals in the early stages of a trend reversal.

2. Inappropriate parameter selection may lead to poor strategy performance. If the period is too long, the strategy will be sluggish in response; if the period is too short, it may lead to excessive false signals.

3. Like all strategies based on a single indicator, this strategy may not perform well in sideways markets, generating more false signals and losing trades.

## Optimization Directions

1. Consider introducing other technical indicators or fundamental factors as filters, such as trading volume, trend indicators, etc., to further confirm the validity of the HMA crossover signals.

2. For parameter optimization, methods such as genetic algorithms and grid search can be used to find the optimal parameter combination on historical data that best suits the current market.

3. Incorporate stop-loss and take-profit mechanisms into the strategy to control the risk and return of each trade.

4. Consider the trendiness of the market. By introducing indicators for judging market trends, one can trade in trending markets and avoid sideways markets.

## Summary

The Hull Moving Average Crossover strategy is a simple and easy-to-use trading strategy. With the fast-response characteristic of HMA, it can capture price changes in a timely manner. However, like all strategies, it also has its limitations and its performance will be affected by market types and parameter selection. In practical application, it can be combined with other analysis methods to further confirm the signals. At the same time, reasonable risk control measures, such as stop-loss and take-profit, position management, etc., are also indispensable parts for the stable operation of the strategy.

[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_1_close|0|Source: close|high|low|open|hl2|hlc3|hlcc4|ohlc4|
|v_input_2|0|Hull Variation: Ehma|Thma|Hma|
|v_input_3|200|Length(180-200 for floating S/R , 55 for swing entry)|
|v_input_4|true|Length multiplier (Used to view higher timeframes with straight band)|
|v_input_5|false|Show Hull MA from X timeframe? (good for scalping)|
|v_input_6|240|Higher timeframe|
|v_input_7|true|Color Hull according to trend?|
|v_input_8|false|Color candles based on Hull's Trend?|
|v_input_9|true|Show as a Band?|
|v_input_10|true|Line Thickness|
|v_input_11|40|Band Transparency|


> Source (PineScript)

``` pinescript
/*backtest
start: 2023-03-16 00:00:00
end: 2024-03-21 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=4
//Basic Hull Ma Pack tinkered by InSilico 
strategy("RKTD BETA - Hull Suite by InSilico", overlay=true)

//INPUT
src = input(close, title="Source")
modeSwitch = input("Ehma", title="Hull Variation", options=["Hma", "Thma", "Ehma"])
length = input(200, title="Length(180-200 for floating S/R , 55 for swing entry)")
lengthMult = input(1.0, title="Length multiplier (Used to view higher timeframes with straight band)")

useHtf = input(false, title="Show Hull MA from X timeframe? (good for scalping)")
htf = input("240", title="Higher timeframe", type=input.resolution)

switchColor = input(true, "Color Hull according to trend?")
candleCol = input(false,title="Color candles based on Hull's Trend?")
visualSwitch  = input(true, title="Show as a Band?")
thicknesSwitch = input(1, title="Line Thickness")
transpSwitch = input(40, title="Band Transparency",step=5)

// Alert Messages
longAlertMessage = '{"base": "BTC", "quote": "USDT", "price": ' + tostring(close) + ', "action": "long", "orderType": "market", "close":true}'
shortAlertMessage = '{"base": "BTC", "quote": "USDT", "price": ' + tostring(close) + ', "action": "short", "orderType": "market", "close":true}'

//FUNCTIONS
//HMA
HMA(_src, _length) =>  wma(2 * wma(_src, _length / 2) - wma(_src, _length), round(sqrt(_length)))
//EHMA    
EHMA(_src, _length) =>  ema(2 * ema(_src, _length / 2) - ema(_src, _length), round(sqrt(_length)))
//THMA    
THMA(_src, _length) =>  wma(wma(_src,_length / 3) * 3 - wma(_src, _length / 2) - wma(_src, _length), _length)
    
//SWITCH
Mode(modeSwitch, src, len) =>
      modeSwitch == "Hma"  ? HMA(src, len) :
      modeSwitch == "Ehma" ? EHMA(src, len) : 
      modeSwitch == "Thma" ? THMA(src, len/2) : na

//OUT
_hull = Mode(modeSwitch, src, int(length * lengthMult))
HULL = useHtf ? security(syminfo.ticker, htf, _hull) : _hull
MHULL = HULL[0]
SHULL = HULL[2]

//COLOR
hullColor = switchColor ? (HULL > HULL[2] ? #00ff00 : #ff0000) : #ff9800

//PLOT
///< Frame
Fi1 = plot(MHULL, title="MHULL", color=hullColor, linewidth=thicknesSwitch, transp=50)
Fi2 = plot(visualSwitch ? SHULL : na, title="SHULL", color=hullColor, linewidth=thicknesSwitch, transp=50)
alertcondition(crossover(MHULL, SHULL), title="Hull trending up.", message="Hull trending up.")
alertcondition(crossover(SHULL, MHULL), title="Hull trending down.", message="Hull trending down.")
///< Ending Filler
fill(Fi1, Fi2, title="Band Filler", color=hullColor, transp=transpSwitch)
///BARCOLOR
barcolor(color = candleCol ? (switchColor ? hullColor : na) : na)

// Define Buy and Sell Conditions based on crossovers
buyCondition = crossover(MHULL, SHULL)
sellCondition = crossunder(MHULL, SHULL)

// Plotting the Hull Moving Average (HMA)
plot(MHULL, title="MHULL", color=hullColor, linewidth=thicknesSwitch)
plot(SHULL, title="SHULL", color=hullColor, linewidth=thicknesSwitch, transp=transpSwitch)
fill(Fi1, Fi2, title="Band Filler", color=hullColor, transp=transpSwitch)

// Execute Strategy based on Buy and Sell Conditions
strategy.entry("Buy", strategy.long, when=buyCondition, alert_message=longAlertMessage)
strategy.entry("Sell", strategy.short, when=sellCondition, alert_message=shortAlertMessage)

// Additional elements like bar coloring remain unchanged
barcolor(color = candleCol ? (switchColor ? hullColor : na) : na)

```

> Detail

https://www.fmz.com/strategy/445823

> Last Modified

2024-03-22 14:57:10
