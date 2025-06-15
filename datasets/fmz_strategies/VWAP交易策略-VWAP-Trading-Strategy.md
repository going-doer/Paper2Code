
> Name

VWAP交易策略-VWAP-Trading-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1cfeb989f0bfb31c7bc.png)
[trans]

#### 概述

该策略是一个基于EMA、VWAP和成交量的交易策略。主要思路是在特定的交易时间内,当收盘价突破VWAP和EMA,且成交量大于前一根K线的成交量时产生开仓信号。同时设置了止损和止盈,以及在特定时间段内平仓的条件。

#### 策略原理

1. 计算EMA和VWAP指标。
2. 判断是否在指定的交易时间内。
3. 多头开仓条件:收盘价大于VWAP和EMA,成交量大于前一根K线,并且收盘价大于开盘价。
4. 空头开仓条件:收盘价小于VWAP和EMA,成交量大于前一根K线,并且开盘价大于收盘价。
5. 多头平仓条件:收盘价跌破VWAP或EMA,达到止盈或止损点位,或者到达指定的离场时间。
6. 空头平仓条件:收盘价突破VWAP或EMA,达到止盈或止损点位,或者到达指定的离场时间。

#### 策略优势

1. 同时考虑了价格趋势(EMA)、市场公允价值(VWAP)和成交量,开仓条件更加严格,有助于提高策略的胜率。
2. 设置了止损和止盈,以控制风险和锁定利润。
3. 限定了交易时间和离场时间,避免了在非交易时段和持仓过夜的风险。

#### 策略风险

1. 该策略在震荡市中表现可能不佳,因为频繁的突破和回撤可能导致多次开仓和平仓,从而增加交易成本和滑点。
2. 止损点位是固定的,在行情剧烈波动时,可能会被提前触发,导致策略承受较大损失。
3. 该策略未考虑实际的市场深度和委托情况,在实盘交易中可能面临滑点和开仓失败等问题。

#### 策略优化方向

1. 可以考虑加入更多的过滤条件,如ATR、RSI等指标,以进一步确认趋势和动量的强度。
2. 止损和止盈点位可以设置为动态的,如跟随ATR或百分比止损,以适应不同的市场波动。
3. 可以对参数进行优化,如EMA长度、VWAP来源、止损止盈点位等,以提高策略的稳定性和盈利能力。
4. 可以考虑加入仓位管理,如根据波动率或资金比例调整开仓量,以控制整体风险。

#### 总结

该策略通过综合考虑价格趋势、市场公允价值和成交量,在特定的交易时间内进行交易。虽然设置了止损止盈和限定交易时间,但在实际应用中仍需注意震荡市和滑点等风险。未来可以通过加入更多过滤条件、优化参数和仓位管理等方式来提高策略的稳健性和盈利能力。

|| 

#### Overview

This strategy is a trading strategy based on EMA, VWAP, and volume. The main idea is to generate opening signals when the closing price breaks through VWAP and EMA, and the trading volume is greater than the previous candle's volume within a specific trading time. It also sets stop loss and take profit, as well as conditions for closing positions within a specific time period.

#### Strategy Principle

1. Calculate EMA and VWAP indicators.
2. Determine whether it is within the specified trading time.
3. Long entry condition: closing price is greater than VWAP and EMA, volume is greater than the previous candle, and closing price is greater than opening price.
4. Short entry condition: closing price is less than VWAP and EMA, volume is greater than the previous candle, and opening price is greater than closing price.
5. Long exit condition: closing price falls below VWAP or EMA, reaches stop loss or take profit levels, or reaches the specified exit time.
6. Short exit condition: closing price breaks above VWAP or EMA, reaches stop loss or take profit levels, or reaches the specified exit time.

#### Strategy Advantages

1. It considers price trend (EMA), market fair value (VWAP), and trading volume simultaneously, making the opening conditions more strict, which helps to improve the strategy's win rate.
2. It sets stop loss and take profit to control risk and lock in profits.
3. It limits trading time and exit time to avoid risks during non-trading hours and overnight holding.

#### Strategy Risks

1. The strategy may not perform well in a volatile market, as frequent breakthroughs and pullbacks may lead to multiple openings and closings, increasing transaction costs and slippage.
2. The stop loss level is fixed, which may be triggered prematurely when the market fluctuates violently, causing the strategy to suffer significant losses.
3. The strategy does not consider actual market depth and order status, which may face issues such as slippage and opening failures in real trading.

#### Strategy Optimization Direction

1. Consider adding more filtering conditions, such as ATR and RSI indicators, to further confirm the strength of the trend and momentum.
2. Stop loss and take profit levels can be set dynamically, such as following ATR or percentage stop loss, to adapt to different market volatilities.
3. Optimize parameters, such as EMA length, VWAP source, stop loss and take profit levels, etc., to improve the stability and profitability of the strategy.
4. Consider adding position management, such as adjusting the opening volume according to volatility or capital ratio, to control overall risk.

#### Summary

By comprehensively considering price trends, market fair value, and trading volume, this strategy trades within a specific trading time. Although stop loss, take profit, and limited trading time are set, it still needs to pay attention to risks such as volatile markets and slippage in actual application. In the future, the strategy's robustness and profitability can be improved by adding more filtering conditions, optimizing parameters, and managing positions.

[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_int_1|21|EMA Length|
|v_input_source_1_hlc3|0|VWAP Source: hlc3|high|low|open|hl2|close|hlcc4|ohlc4|
|v_input_float_1|100|Stop Loss (points)|
|v_input_float_2|200|Target (points)|
|v_input_1|0950-1430|Only take entry during|
|v_input_2|1515-1525|Exit Trade|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-04-27 00:00:00
end: 2024-04-28 00:00:00
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("EMA, VWAP, Volume Strategy", overlay=true, process_orders_on_close=true)

// Inputs
emaLength = input.int(21, title="EMA Length")
vwapSource = input.source(defval=hlc3, title='VWAP Source')
stopLossPoints = input.float(100, title="Stop Loss (points)")
targetPoints = input.float(200, title="Target (points)")
session = input("0950-1430", title='Only take entry during')
exit = input(defval='1515-1525', title='Exit Trade')

tradein = not na(time(timeframe.period, session))
exit_time = not na(time(timeframe.period, exit))

// Calculate indicators
ema = ta.ema(close, emaLength)
vwapValue = ta.vwap(vwapSource)

// Entry Conditions
longCondition = close > vwapValue and close > ema and volume > volume[1] and close > open and tradein
shortCondition = close < vwapValue and close < ema and volume > volume[1] and open > close and tradein

// Exit Conditions
longExitCondition = ta.crossunder(close, vwapValue) or ta.crossunder(close, ema) or close - strategy.position_avg_price >= targetPoints or close - strategy.position_avg_price <= -stopLossPoints or exit_time
shortExitCondition = ta.crossover(close, vwapValue) or ta.crossover(close, ema) or strategy.position_avg_price - close >= targetPoints or strategy.position_avg_price - close <= -stopLossPoints or exit_time

// Plotting
plot(vwapValue, color=color.blue, title="VWAP")
plot(ema, color=color.green, title="EMA")

// Strategy
if longCondition
    strategy.entry("Long", strategy.long)

if shortCondition
    strategy.entry("Short", strategy.short)

if longExitCondition
    strategy.close('Long', immediately=true)

if shortExitCondition
    strategy.close("Short", immediately=true)

```

> Detail

https://www.fmz.com/strategy/449813

> Last Modified

2024-04-29 14:20:39
