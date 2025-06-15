
> Name

挤压回测变形金刚v20-Squeeze-Backtest-Transformer-v20

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1a4ad57451c36889bcf.png)

[trans]
#### 概述
挤压回测变形金刚v2.0是一个基于挤压型策略的量化交易系统。它通过设置进场、止损和止盈百分比,以及最大持仓时间等参数,在特定时间范围内对策略进行回测。该策略支持多方向交易,可以灵活设置交易方向为做多或做空。同时,该策略还提供了丰富的回测期设置选项,可以方便地选择固定时间范围或者最大回测时间。

#### 策略原理
1. 首先根据用户设置的回测期参数,确定回测的起始时间和结束时间。
2. 在回测期内,如果当前没有持仓且价格触及进场价格(根据开仓百分比计算),则开仓并同时设置止损和止盈价格(根据止损和止盈百分比计算)。
3. 如果已经持仓,则取消之前的止盈止损单,重新设置新的止盈止损价格(根据当前持仓均价计算)。
4. 如果设置了最大持仓时间,当持仓时间达到最大值时,强制平仓。
5. 策略支持做多和做空两个方向的交易。

#### 策略优势
1. 参数设置灵活,可以根据不同的市场情况和交易需求进行调整。
2. 支持多方向交易,可以在不同的市场行情下获取收益。
3. 提供了丰富的回测期设置选项,可以方便地进行历史数据回测和分析。
4. 止损和止盈设置可以有效控制风险,提高资金利用效率。
5. 最大持仓时间设置可以避免持仓过久而面临市场风险。

#### 策略风险
1. 进场价格、止损价格和止盈价格的设置对策略收益有很大影响,不当的参数设置可能导致亏损。
2. 市场波动剧烈时,可能出现开仓后立即触发止损的情况,从而导致损失。
3. 如果持仓时触发最大持仓时间平仓,有可能错失后续的获利机会。
4. 策略在某些特殊行情下(如震荡市)表现可能不佳。

#### 策略优化方向
1. 可以考虑引入更多的技术指标或者市场情绪指标,对进场、止损和止盈条件进行优化,提高策略的稳定性和盈利能力。
2. 对于最大持仓时间的设置,可以根据市场波动性和持仓盈亏情况动态调整,避免固定时间平仓可能带来的机会成本。
3. 针对震荡市的特点,可以加入震荡区间突破或者趋势转折确认等逻辑,降低频繁交易带来的成本。
4. 考虑加入仓位管理和资金管理策略,控制单次交易风险敞口,提高资金利用效率和稳定性。

#### 总结
挤压回测变形金刚v2.0是一个基于挤压型策略的量化交易系统,通过灵活的参数设置和多方向交易支持,可以在不同市场环境下进行交易。同时,丰富的回测期设置选项和止盈止损设置可以帮助用户进行历史数据分析和风险控制。但是,策略的表现受参数设置影响较大,需要根据市场特点和交易需求进行优化和改进,以提高策略的稳健性和盈利能力。未来可以考虑引入更多技术指标、动态调整最大持仓时间、优化震荡市策略以及加强仓位和资金管理等方面进行优化。

|| 

#### Overview
Squeeze Backtest Transformer v2.0 is a quantitative trading system based on a squeeze strategy. By setting parameters such as entry, stop loss, take profit percentages, and maximum holding time, it backtests the strategy within a specific time range. The strategy supports multi-directional trading and can flexibly set the trading direction to long or short. At the same time, the strategy also provides rich options for setting the backtest period, which can easily select a fixed time range or the maximum backtest time.

#### Strategy Principle
1. First, determine the start and end time of the backtest based on the backtest period parameters set by the user.
2. During the backtest period, if there is no current position and the price reaches the entry price (calculated based on the opening percentage), open a position and set the stop loss and take profit prices (calculated based on the stop loss and take profit percentages) at the same time.
3. If a position is already held, cancel the previous take profit and stop loss orders and reset new take profit and stop loss prices (calculated based on the current position average price).
4. If the maximum holding time is set, when the holding time reaches the maximum value, force the position to close.
5. The strategy supports trading in both long and short directions.

#### Strategy Advantages
1. Flexible parameter settings can be adjusted according to different market conditions and trading needs.
2. Support multi-directional trading to obtain profits in different market conditions.
3. Provide rich options for setting the backtest period, which can easily conduct historical data backtesting and analysis.
4. Stop loss and take profit settings can effectively control risks and improve capital utilization efficiency.
5. The maximum holding time setting can avoid holding positions for too long and facing market risks.

#### Strategy Risks
1. The setting of the entry price, stop loss price and take profit price has a great impact on the strategy's return. Improper parameter settings may lead to losses.
2. When the market fluctuates violently, a stop loss may be triggered immediately after opening a position, resulting in losses.
3. If the maximum holding time triggers the closing of a position, it may miss the opportunity for subsequent profits.
4. The strategy may not perform well in some special market conditions (such as a sideways market).

#### Strategy Optimization Direction
1. Consider introducing more technical indicators or market sentiment indicators to optimize the conditions for entry, stop loss and take profit to improve the stability and profitability of the strategy.
2. For the setting of the maximum holding time, it can be dynamically adjusted according to market volatility and position profit and loss to avoid the opportunity cost that a fixed time closing may bring.
3. For the characteristics of the sideways market, logic such as sideways range breakthrough or trend reversal confirmation can be added to reduce the cost of frequent trading.
4. Consider adding position management and capital management strategies to control the risk exposure of a single transaction and improve the efficiency and stability of capital utilization.

#### Summary
Squeeze Backtest Transformer v2.0 is a quantitative trading system based on a squeeze strategy that can trade in different market environments through flexible parameter settings and multi-directional trading support. At the same time, rich backtest period setting options and take profit and stop loss settings can help users conduct historical data analysis and risk control. However, the performance of the strategy is greatly affected by parameter settings and needs to be optimized and improved based on market characteristics and trading needs to improve the stability and profitability of the strategy. In the future, we can consider introducing more technical indicators, dynamically adjusting the maximum holding time, optimizing sideways market strategies, and strengthening position and capital management to optimize.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_string_1|0|(?Squeeze Settings)Direction: LONG|SHORT|
|v_input_float_1|1.4|Open, %|
|v_input_float_2|0.6|Close, %|
|v_input_float_3|0.8|Stop Loss, %|
|v_input_bool_1|true|Max Bars To Sell|
|v_input_int_1|10|maxBars|
|v_input_string_2|0|Bind: close|high|open|low|mid (hl)|mid (oc)|
|v_input_bool_2|true|(?Backtesting Period)Fixed Range|
|v_input_string_3|0|rangeStart: 24 Hours|12 Hours|6 Hours|48 Hours|1 Week|2 Weeks|1 Month|Maximum|
|v_input_1|timestamp(12 Apr 2024 00:00 +0000)|Backtesting Start|
|v_input_2|timestamp(20 Apr 2024 00:00 +0000)|Backtesting End|


> Source (PineScript)

``` pinescript
/*backtest
start: 2023-04-22 00:00:00
end: 2024-04-27 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5

strategy(title="Squeeze Backtest by Shaqi v2.0", overlay=true, pyramiding=0, currency="USD", process_orders_on_close=true, commission_type=strategy.commission.percent, commission_value=0.075, default_qty_type=strategy.percent_of_equity, default_qty_value=100, initial_capital=100, backtest_fill_limits_assumption=0)
R0 = "6 Hours"
R1 = "12 Hours"
R2 = "24 Hours"
R3 = "48 Hours"
R4 = "1 Week"
R5 = "2 Weeks"
R6 = "1 Month"
R7 = "Maximum"

BL = "low"
BH = "high"
BO = "open"
BC = "close"
BHL= "mid (hl)"
BOC = "mid (oc)"

LONG = "LONG"
SHORT = "SHORT"

direction = input.string(title="Direction", defval=LONG, options=[LONG, SHORT], group="Squeeze Settings")
strategy.risk.allow_entry_in(direction == LONG ? strategy.direction.long : strategy.direction.short)
openPercent = input.float(1.4, "Open, %", minval=0.01, maxval=100, step=0.1, inline="Percents", group="Squeeze Settings") * 0.01
closePercent = input.float(0.6, "Close, %", minval=0.01, maxval=100, step=0.1, inline="Percents", group="Squeeze Settings") * 0.01
stopPercent = input.float(0.8, "Stop Loss, %", minval=0.01, maxval=100, step=0.1, inline="Percents", group="Squeeze Settings") * 0.01
isMaxBars = input.bool(true, "Max Bars To Sell", inline="MaxBars", group="Squeeze Settings")
maxBars = input.int(10, title="", minval=0, maxval=1000, step=1, inline="MaxBars", group="Squeeze Settings")
bind = input.string(BC, "Bind", options=[BL, BH, BO, BC, BHL, BOC], group="Squeeze Settings")
isRange = input.bool(true, "Fixed Range", inline="Range", group="Backtesting Period")
rangeStart = input.string(R2, "", options=[R0, R1, R2, R3, R4, R5, R6, R7], inline="Range", group="Backtesting Period")
periodStart = input(timestamp("12 Apr 2024 00:00 +0000"), "Backtesting Start", group="Backtesting Period")
periodEnd = input(timestamp("20 Apr 2024 00:00 +0000"), "Backtesting End", group="Backtesting Period")

int startDate = na
int endDate = na
if isRange
    if rangeStart == R0
        startDate := timenow - 21600000
        endDate := timenow
    else if rangeStart == R1
        startDate := timenow - 43200000
        endDate := timenow
    else if rangeStart == R2
        startDate := timenow - 86400000
        endDate := timenow
    else if rangeStart == R3
        startDate := timenow - 172800000
        endDate := timenow
    else if rangeStart == R4
        startDate := timenow - 604800000
        endDate := timenow
    else if rangeStart == R5
        startDate := timenow - 1209600000
        endDate := timenow
    else if rangeStart == R6
        startDate := timenow - 2592000000
        endDate := timenow
    else if rangeStart == R7
        startDate := time
        endDate := timenow
else 
    startDate := periodStart
    endDate := periodEnd
    
float bindOption = na
if bind == BL
    bindOption := low
else if bind == BH
    bindOption := high
else if bind == BO
    bindOption := open
else if bind == BC
    bindOption := close
else if bind == BHL
    bindOption := hl2
else
    bindOption := ohlc4

afterStartDate = (time >= startDate)
beforeEndDate = (time <= endDate)
periodCondition = true
notInTrade = strategy.position_size == 0
inTrade = strategy.position_size != 0

barsFromEntry = ta.barssince(strategy.position_size[0] > strategy.position_size[1])
entry = strategy.position_size[0] > strategy.position_size[1]
entryBar = barsFromEntry == 0
notEntryBar = barsFromEntry != 0
openLimitPrice = direction == LONG ? (bindOption - bindOption * openPercent) : (bindOption + bindOption * openPercent)

closeLimitPriceEntry = openLimitPrice * (direction == LONG ? 1 + closePercent : 1 - closePercent)
closeLimitPrice = strategy.position_avg_price * (direction == LONG ? 1 + closePercent : 1 - closePercent)

stopLimitPriceEntry = direction == LONG ? openLimitPrice - openLimitPrice * stopPercent : openLimitPrice + openLimitPrice * stopPercent
stopLimitPrice = direction == LONG ? strategy.position_avg_price - strategy.position_avg_price * stopPercent : strategy.position_avg_price + strategy.position_avg_price * stopPercent

if periodCondition and notInTrade
    strategy.entry(direction == LONG ? "BUY" : "SELL", direction == LONG ? strategy.long : strategy.short, limit = openLimitPrice, stop = stopLimitPriceEntry)
    strategy.exit("INSTANT", limit = closeLimitPriceEntry, stop = stopLimitPriceEntry, comment_profit = direction == LONG ? 'INSTANT SELL' : 'INSTANT BUY', comment_loss = 'INSTANT STOP')
if inTrade 
    strategy.cancel("INSTANT")
    strategy.exit(direction == LONG ? "SELL" : "BUY", limit = closeLimitPrice, stop = stopLimitPrice, comment_profit = direction == LONG ? "SELL" : "BUY", comment_loss = "STOP")
if isMaxBars and barsFromEntry == maxBars
    strategy.close_all(comment = "TIMEOUT STOP", immediately = true)



showStop = stopPercent <= 0.20

// plot(showStop ? stopLimitPrice : na, title="Stop Loss Limit Order", force_overlay=true, style=plot.style_linebr, color=#c50202, linewidth=1, offset=1)
// plot(closeLimitPrice, title="Take Profit Limit Order", force_overlay=true, style=plot.style_linebr, color = direction == LONG ? color.red : color.blue, linewidth=1, offset=1)
// plot(strategy.position_avg_price, title="Buy Order Filled Price", force_overlay=true, style=plot.style_linebr, color=direction == LONG ? color.blue : color.red, linewidth=1, offset=1)
plot(showStop ? stopLimitPrice : na, title="Stop Loss Limit Order", force_overlay=true, style=plot.style_linebr, color=#c50202, linewidth=1, offset=0)
plot(closeLimitPrice, title="Take Profit Limit Order", force_overlay=true, style=plot.style_linebr, color = direction == LONG ? color.red : color.blue, linewidth=1, offset=0)
plot(strategy.position_avg_price, title="Buy Order Filled Price", force_overlay=true, style=plot.style_linebr, color=direction == LONG ? color.blue : color.red, linewidth=1, offset=0)

plot(openLimitPrice, title="Trailing Open Position Limit Order", style=plot.style_stepline, color=color.new(direction == LONG ? color.blue : color.red, 30), offset=1)
plot(closeLimitPriceEntry, title="Trailing Close Position Limit Order", style=plot.style_stepline, color=color.new(direction == LONG ? color.red : color.blue, 80), offset=1)
plot(stopLimitPriceEntry, title="Trailing Stop Position Limit Order", style=plot.style_stepline, color=color.new(#c50202, 80), offset=1)


```

> Detail

https://www.fmz.com/strategy/449718

> Last Modified

2024-04-28 14:09:26
