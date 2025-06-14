
> Name

周二反转策略周末过滤-Turnaround-Tuesday-Strategy-Weekend-Filter

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/193efd44d5ffdd0c77c.png)

[trans]
#### 概述
该策略名为"周二反转策略(周末过滤)",主要思路是基于均线和其他过滤条件,在满足条件的周一开盘买入,周三开盘卖出,以捕捉周二的反转行情。该策略通过对RSI、ATR等指标的过滤,排除5月份等特定时间,以提高策略胜率和收益风险比。

#### 策略原理
1. 使用30日均线作为趋势判断依据,当前一交易日收盘价低于30日均线时,认为趋势向下,符合买入条件之一。
2. 使用3日RSI和10日ATR作为过滤条件,3日RSI小于51且收盘价相对10日ATR小于95%时,认为市场情绪悲观,但无极端行情,符合买入条件。
3. 排除5月份,因为"Sell in May and go away"的效应,股市通常表现疲软。
4. 综合以上条件,在周一且满足所有过滤条件时买入,周三开盘时卖出。

#### 策略优势
1. 基于均线和情绪指标的组合判断,能有效捕捉周二反转行情。
2. 通过对RSI和ATR的双重过滤,排除了极端行情下的交易,提高了策略的胜率和收益风险比。
3. 排除5月份,避开了通常表现不佳的时间段,提高了策略表现。
4. 仅在周一买入周三卖出,交易频率低,手续费成本小。

#### 策略风险
1. 对于趋势很强的行情,反转不明显时,策略表现会不佳。
2. 固定买卖时间可能错失更好的买卖时机,限制了策略的灵活性和收益空间。
3. 依赖指标判断,在市场发生剧烈变化时,指标失效的风险。
4. 月份判断基于历史经验,不代表未来情况一定相同,存在时效性风险。

#### 策略优化方向
1. 可以考虑引入更多有效的过滤指标,如成交量、波动率等,以提高策略的稳健性和适应性。
2. 优化买卖时机的选择,如加入盘中突破确认等条件,提高策略的灵活性和收益空间。
3. 对于持仓周期的优化,可以考虑更长的持仓时间,以更充分地捕捉趋势。
4. 针对不同市场状态,设置不同的参数,提高策略的适应性。
5. 加入仓位管理和风险控制模块,以应对市场的极端情况。

#### 总结
周二反转策略(周末过滤)通过均线、RSI和ATR等指标的组合判断,在特定时间买卖标的,以捕捉周二的反转行情。策略交易频率低,手续费成本小,并通过时间段过滤和指标过滤,提高了策略的胜率和风险收益比。但是,策略也存在一定的局限性和风险,如趋势行情下表现不佳,以及买卖时机和持仓周期固定等。未来可以通过引入更多过滤条件,优化出场时机,动态调整参数,仓位管理和风控等方面进行优化和改进,使策略能够更好地适应多变的市场状态。

|| 

#### Overview
The strategy is named "Turnaround Tuesday Strategy (Weekend Filter)". The main idea is to buy at the Monday open and sell at the Wednesday open when certain conditions based on moving averages and other filters are met, in order to capture the Tuesday turnaround. By filtering with RSI, ATR, and excluding specific times like May, the strategy aims to improve its win rate and risk-reward ratio.

#### Strategy Principles
1. Use the 30-day moving average as a trend determination basis. When the previous trading day's close is below the 30-day MA, it is considered a downtrend and meets one of the buying conditions.
2. Use the 3-day RSI and 10-day ATR as filter conditions. When the 3-day RSI is less than 51 and the close relative to the 10-day ATR is less than 95%, the market sentiment is considered pessimistic but without extreme conditions, meeting the buying conditions.
3. Exclude the month of May due to the "Sell in May and go away" effect, as the stock market tends to be sluggish.
4. Combining the above conditions, buy on Monday when all filter conditions are met, and sell at the Wednesday open.

#### Strategy Advantages
1. The combination of moving average and sentiment indicators can effectively capture the Tuesday turnaround.
2. The dual filtering of RSI and ATR excludes trades in extreme conditions, improving the strategy's win rate and risk-reward ratio.
3. Excluding May avoids trading during typically underperforming periods, enhancing strategy performance.
4. Trading only from Monday to Wednesday results in low trading frequency and small commission costs.

#### Strategy Risks
1. The strategy may underperform when the trend is strong and the reversal is not evident.
2. Fixed buying and selling times may miss better entry and exit points, limiting the strategy's flexibility and profit potential.
3. Relying on indicator judgments risks invalidity when the market changes drastically.
4. Monthly judgments based on historical experience do not guarantee future situations will be the same, posing a timeliness risk.

#### Strategy Optimization Directions
1. Consider introducing more effective filtering indicators, such as volume and volatility, to improve the strategy's robustness and adaptability.
2. Optimize the selection of buying and selling timings, such as adding intraday breakout confirmation conditions, to increase flexibility and profit potential.
3. For holding period optimization, consider longer holding times to more fully capture trends.
4. Set different parameters for different market conditions to enhance the strategy's adaptability.
5. Incorporate position management and risk control modules to cope with extreme market situations.

#### Summary
The Turnaround Tuesday Strategy (Weekend Filter) uses a combination of moving averages, RSI, ATR, and other indicators to buy and sell at specific times, aiming to capture the Tuesday turnaround. The strategy has low trading frequency, small commission costs, and improves its win rate and risk-reward ratio through time period and indicator filtering. However, the strategy also has certain limitations and risks, such as underperformance in trending markets and fixed buying/selling times and holding periods. Future optimizations can introduce more filtering conditions, optimize exit timings, dynamically adjust parameters, manage positions, and control risk to better adapt to changing market conditions.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_1|30|Moving Average Period|
|v_input_2|true|Use RSI Filter|
|v_input_3|true|Use ATR Filter|
|v_input_4|true|Exclude May|
|v_input_5|timestamp(2009-01-01 00:00:00)|Start Backtest|
|v_input_6|timestamp(2025-01-01 00:00:00)|End Backtest|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-03-01 00:00:00
end: 2024-03-31 23:59:59
period: 2h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © muikol  

//@version=5
strategy("Turnaround Tuesday", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.035)

// Inputs for MA period, filter_1, filter_2, month filter, and testing period
ma_period = input(30, title="Moving Average Period")
use_filter_1 = input(true, title="Use RSI Filter")
use_filter_2 = input(true, title="Use ATR Filter")
use_month_filter = input(true, title="Exclude May")
start_date = input(defval=timestamp("2009-01-01 00:00:00"), title="Start Backtest")
end_date = input(defval=timestamp("2025-01-01 00:00:00"), title="End Backtest")

// Data calculations
MA_tt = ta.sma(close, ma_period)
atr10 = ta.atr(10)
rsi3 = ta.rsi(close, 3)
c_1 = close[1]

// Entry conditions
isMonday = dayofweek == dayofweek.monday
bear = close[1] < MA_tt[1]
filter_1 = use_filter_1 ? rsi3[1] < 51 : true
filter_2 = use_filter_2 ? c_1/atr10[1] < 95 : true
notMay = use_month_filter ? month != 5 : true
entryCondition = isMonday and bear and notMay and filter_1 and filter_2

// Date check
inTestPeriod = true
// Exit conditions
isWednesdayOpen = dayofweek == dayofweek.wednesday 

// Entry and exit triggers
if entryCondition and inTestPeriod
    strategy.entry("Buy", strategy.long)

if isWednesdayOpen and strategy.position_size > 0 and inTestPeriod
    strategy.close("Buy")

// Plot the moving average
plot(MA_tt, title="Moving Average", color=color.blue)

```

> Detail

https://www.fmz.com/strategy/449948

> Last Modified

2024-04-30 16:07:45
