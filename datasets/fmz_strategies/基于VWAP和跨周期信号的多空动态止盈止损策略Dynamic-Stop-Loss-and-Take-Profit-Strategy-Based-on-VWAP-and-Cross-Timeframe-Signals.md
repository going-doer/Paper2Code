
> Name

基于VWAP和跨周期信号的多空动态止盈止损策略Dynamic-Stop-Loss-and-Take-Profit-Strategy-Based-on-VWAP-and-Cross-Timeframe-Signals

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/fe74f9988e6c2d6162.png)
[trans]
## 概述

该策略使用日线的VWAP(成交量加权平均价)作为进场和出场的信号。当收盘价上穿VWAP时触发做多,止损设置在VWAP下方的前一根K线低点,目标价设置在开仓价上方3个点;当收盘价下穿VWAP时触发做空,止损设置在VWAP上方的前一根K线高点,目标价设置在开仓价下方3个点。该策略没有包含出场条件,交易会一直持有直到反向信号出现。

## 策略原理

1. 获取日线的VWAP数据,作为趋势判断和交易信号的依据。
2. 判断当前收盘价是否上穿/下穿VWAP,分别作为做多和做空的触发条件。
3. 做多时,如果前一根K线低点在VWAP之下,则将其作为止损位,否则直接使用VWAP作为止损;做空则相反。
4. 开仓后,分别设置固定3个点的止盈位。
5. 策略持续运行,直到触发反向信号平仓并开立新的仓位。

通过跨周期的VWAP数据判断趋势,同时利用动态止损和固定点数止盈,可以有效把握趋势行情,控制回撤风险,并及时锁定利润。

## 优势分析

1. 简单有效:策略逻辑清晰,仅使用VWAP一个指标,就可以实现趋势判断和信号触发,简单易于实现和优化。
2. 动态止损:根据前一根K线的高低点设置止损,可以更好地适应市场波动,降低风险。
3. 固定点数止盈:以固定点数设置目标价,有助于及时锁定利润,避免利润回吐。
4. 及时止损止盈:策略在触发反向信号时会立即平仓,不会对已有利润造成额外损失,同时会开立新的仓位,捕捉新的趋势行情。

## 风险分析

1. 参数优化:策略使用了固定的3个点作为止盈,实际交易中可能需要根据不同标的和市场特点进行优化,选择最佳参数。
2. 震荡行情:在震荡行情下,频繁的进场和出场可能导致较高的交易成本,影响收益。
3. 趋势持续性:策略依赖于趋势行情,如果市场处于区间震荡,或趋势持续性较差,可能会出现较多的交易信号,带来更多的风险。

## 优化方向

1. 趋势过滤:加入其他趋势指标如移动平均线、MACD等,对趋势进行二次确认,提高信号可靠性。
2. 动态止盈:根据市场波动性、ATR等指标,动态调整止盈点数,以更好地适应市场。
3. 仓位管理:根据账户资金、风险偏好等因素,对每次交易的仓位大小进行动态调整。
4. 交易时段选择:根据标的特性和交易活跃程度,选择最佳的交易时段,提高策略效率。

## 总结

该策略利用跨周期VWAP数据进行趋势判断和信号触发,同时采用动态止损和固定点数止盈的方式控制风险和锁定利润,是一个简单有效的量化交易策略。通过趋势过滤、动态止盈、仓位管理和交易时段选择等方面的优化,可以进一步提高策略的稳健性和收益潜力。但在实际应用中,仍需注意市场特点、交易成本和参数优化等因素,以期获得更好的策略表现。

|| 

## Overview

This strategy uses the VWAP (Volume Weighted Average Price) from the daily timeframe as a signal for entering and exiting trades. When the close price crosses above the VWAP, it triggers a long entry with the stop loss set at the previous candle's low if it's below the VWAP, and the target price set 3 points above the entry price. Conversely, when the close price crosses below the VWAP, it triggers a short entry with the stop loss set at the previous candle's high if it's above the VWAP, and the target price set 3 points below the entry price. This strategy doesn't include an exit condition, so trades remain open until the opposing signal occurs.

## Strategy Principle

1. Obtain the VWAP data from the daily timeframe, which serves as the basis for trend determination and trading signals.
2. Determine whether the current close price crosses above/below the VWAP, acting as the trigger for long and short entries, respectively.
3. For long entries, if the previous candle's low is below the VWAP, it is used as the stop loss; otherwise, the VWAP itself is used. The opposite applies for short entries.
4. After entering a position, set a fixed 3-point take profit level.
5. The strategy continues to run until a reverse signal triggers the position to close and open a new one.

By using cross-timeframe VWAP data to determine trends and leveraging dynamic stop losses and fixed-point take profits, the strategy can effectively capture trending markets, control drawdown risks, and timely lock in profits.

## Advantage Analysis

1. Simplicity and effectiveness: The strategy logic is clear, using only the VWAP indicator for trend determination and signal triggering, making it simple to implement and optimize.
2. Dynamic stop loss: By setting the stop loss based on the previous candle's high or low, the strategy adapts better to market fluctuations and reduces risk.
3. Fixed-point take profit: Setting the target price with a fixed number of points helps to lock in profits promptly and avoid profit erosion.
4. Timely stop loss and take profit: The strategy immediately closes the position when a reverse signal is triggered, preventing additional losses on existing profits. It also opens a new position to capture new trending moves.

## Risk Analysis

1. Parameter optimization: The strategy uses a fixed 3 points for take profit, which may require optimization based on different instruments and market characteristics to select the optimal parameters for actual trading.
2. Choppy markets: In choppy market conditions, frequent entries and exits may lead to higher trading costs, affecting profitability.
3. Trend sustainability: The strategy relies on trending markets. If the market is range-bound or lacks trend sustainability, there may be more trading signals generated, introducing more risk.

## Optimization Directions

1. Trend filtering: Incorporate other trend indicators like moving averages, MACD, etc., to confirm trends and improve signal reliability.
2. Dynamic take profit: Adjust the take profit points dynamically based on market volatility, ATR, or other indicators to better adapt to market conditions.
3. Position sizing: Dynamically adjust the position size for each trade based on account size, risk tolerance, and other factors.
4. Trading session selection: Choose the optimal trading sessions based on the characteristics and liquidity of the instrument to improve strategy efficiency.

## Summary

This strategy utilizes cross-timeframe VWAP data for trend determination and signal triggering while employing dynamic stop losses and fixed-point take profits to control risks and lock in profits. It is a simple and effective quantitative trading strategy. Through optimizations in trend filtering, dynamic take profit, position sizing, and trading session selection, the strategy's robustness and profit potential can be further enhanced. However, when applying the strategy in practice, attention should be paid to market characteristics, trading costs, and parameter optimization to achieve better strategy performance.

[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-03-06 00:00:00
end: 2024-03-07 00:00:00
period: 45m
basePeriod: 5m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy('Pine Script Tutorial Example Strategy 1', overlay=true, initial_capital=1000, default_qty_value=100, default_qty_type=strategy.percent_of_equity)
// fastEMA = ta.ema(close, 24)
// slowEMA = ta.ema(close, 200)
// Higher Time Frame
float sl = na
float tgt = na
posSize = 1
vwap_1d = request.security(syminfo.tickerid, "1D", ta.vwap(close))
// plot(vwap_1d)

// To avoid differences on historical and realtime bars, you can use this technique, which only returns a value from the higher timeframe on the bar after it completes:
// indexHighTF = barstate.isrealtime ? 1 : 0
// indexCurrTF = barstate.isrealtime ? 0 : 1
// nonRepaintingVWAP = request.security(syminfo.tickerid, "1D", close[indexHighTF])[indexCurrTF]
// plot(nonRepaintingVWAP, "Non-repainting VWAP")

enterLong = ta.crossover(close, vwap_1d)
exitLong  = ta.crossunder(close, vwap_1d)

enterShort = ta.crossunder(close, vwap_1d)
exitShort  = ta.crossover(close, vwap_1d)

if enterLong
    sl := low[1]>vwap_1d ?low[1]:vwap_1d
    tgt:=close+3
    strategy.entry("EL", strategy.long, qty=posSize)
    strategy.exit('exitEL', 'EL', stop=sl, limit=tgt)
if enterShort
    sl := high[1]<vwap_1d ?high[1]:vwap_1d
    tgt := close-3
    strategy.entry("ES", strategy.short, qty=posSize)
    strategy.exit('exitES', 'ES', stop=sl, limit=tgt)

// if exitLong
//     strategy.close("EL")
// if exitShort
//     strategy.close("ES")





// goLongCondition1 = ta.crossover(close, vwap_1d)
// timePeriod = time >= timestamp(syminfo.timezone, 2021, 01, 01, 0, 0)
// notInTrade = strategy.position_size <= 0
// if goLongCondition1 and timePeriod and notInTrade
//     stopLoss = low[1]
//     takeProfit = close+3
//     strategy.entry('long', strategy.long)
//     strategy.exit('exit', 'long', stop=stopLoss, limit=takeProfit)
plot(close, color=color.new(#00c510, 0))
plot(vwap_1d, color=color.new(#f05619, 0))
plot(sl, color=color.new(#fbff00, 0))
plot(tgt, color=color.new(#00e1ff, 0))
```

> Detail

https://www.fmz.com/strategy/444041

> Last Modified

2024-03-08 17:37:21
