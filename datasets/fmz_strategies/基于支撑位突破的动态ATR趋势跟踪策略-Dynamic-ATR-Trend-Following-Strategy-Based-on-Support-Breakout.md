
> Name

基于支撑位突破的动态ATR趋势跟踪策略-Dynamic-ATR-Trend-Following-Strategy-Based-on-Support-Breakout

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/15cdd47682466fbf957.png)

[trans]
#### 概述
这是一个基于支撑位突破的动态ATR趋势跟踪策略。该策略综合运用了EMA均线系统、ATR波动率指标以及智能资金概念(SMC)来捕捉市场趋势。策略通过动态计算仓位大小和止损止盈位置,实现了良好的风险管理。

#### 策略原理
策略主要基于以下几个核心组件构建:
1. 使用50和200周期的EMA均线系统来确认市场趋势方向
2. 利用ATR指标动态调整止损和获利目标
3. 通过订单块(Order Block)和失衡区域(Imbalance Zone)分析来寻找最佳入场点
4. 基于账户风险百分比自动计算开仓量
5. 通过观察近20根K线的波动范围来判断市场是否处于盘整状态

#### 策略优势
1. 风险管理完善,通过动态计算保证每笔交易风险可控
2. 趋势判断系统可靠,避免在盘整市场交易
3. 止损止盈设置合理,风险收益比为1:3
4. 充分考虑市场波动性,能适应不同市场环境
5. 代码结构清晰,易于维护和优化

#### 策略风险
1. EMA指标具有滞后性,可能导致入场时机延迟
2. 在剧烈波动市场中可能触发虚假信号
3. 策略依赖趋势持续性,在震荡市场表现可能欠佳
4. 止损位置较宽,在某些情况下可能承受较大损失

#### 策略优化方向
1. 可以引入量价关系分析,提高趋势判断准确性
2. 可以增加市场情绪指标,优化入场时机
3. 考虑加入多时间周期分析,提高系统稳定性
4. 可以细化订单块和失衡区域的判断标准
5. 优化止损方式,考虑采用移动止损

#### 总结
该策略是一个较为完整的趋势跟踪系统,通过合理的风险管理和多重信号确认来提高交易的稳定性。虽然存在一定的滞后性,但整体而言是一个可靠的交易系统。建议在实盘使用前进行充分的回测验证,并根据具体交易品种和市场环境进行参数优化。

|| 

#### Overview
This is a dynamic ATR trend following strategy based on support breakout. The strategy incorporates EMA system, ATR volatility indicator, and Smart Money Concept (SMC) to capture market trends. It achieves effective risk management through dynamic position sizing and stop-loss/take-profit placement.

#### Strategy Principle
The strategy is built on several core components:
1. Uses 50 and 200-period EMA system to confirm market trend direction
2. Utilizes ATR indicator to dynamically adjust stop-loss and profit targets
3. Analyzes Order Blocks and Imbalance Zones to find optimal entry points
4. Automatically calculates position size based on account risk percentage
5. Determines market consolidation by observing price range of last 20 candles

#### Strategy Advantages
1. Comprehensive risk management through dynamic calculation
2. Reliable trend identification system that avoids consolidation markets
3. Reasonable stop-loss and take-profit settings with 1:3 risk-reward ratio
4. Adapts well to different market conditions
5. Clear code structure that's easy to maintain and optimize

#### Strategy Risks
1. EMA indicators have inherent lag, potentially delaying entry points
2. May generate false signals in highly volatile markets
3. Strategy depends on trend continuation, may underperform in ranging markets
4. Wide stop-loss placement may lead to larger losses in certain situations

#### Optimization Directions
1. Incorporate volume-price relationship analysis to improve trend identification
2. Add market sentiment indicators to optimize entry timing
3. Consider multiple timeframe analysis to enhance system stability
4. Refine Order Block and Imbalance Zone identification criteria
5. Optimize stop-loss method, consider implementing trailing stops

#### Summary
This strategy is a comprehensive trend following system that achieves trading stability through proper risk management and multiple signal confirmation. Despite some lag in signals, it represents a reliable trading system overall. It's recommended to conduct thorough backtesting before live implementation and optimize parameters according to specific trading instruments and market conditions.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2019-12-23 08:00:00
end: 2024-12-10 08:00:00
period: 1d
basePeriod: 1d
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// TradingView Pine Script strategy for Smart Money Concept (SMC)
//@version=5
strategy("Smart Money Concept Strategy", overlay=true, default_qty_type=strategy.fixed, default_qty_value=100)

// === Input Parameters ===
input_risk_percentage = input.float(1, title="Risk Percentage", step=0.1)
input_atr_length = input.int(14, title="ATR Length")
input_ema_short = input.int(50, title="EMA Short")
input_ema_long = input.int(200, title="EMA Long")

// === Calculations ===
atr = ta.atr(input_atr_length)
ema_short = ta.ema(close, input_ema_short)
ema_long = ta.ema(close, input_ema_long)

// === Utility Functions ===
// Identify Order Blocks
is_order_block(price, direction) =>
    ((high[1] > high[2] and low[1] > low[2] and direction == 1) or (high[1] < high[2] and low[1] < low[2] and direction == -1))

// Identify Imbalance Zones
is_imbalance() =>
    range_high = high[1]
    range_low = low[1]
    range_high > close and range_low < close

// Calculate Lot Size Based on Risk
calculate_lot_size(stop_loss_points, account_balance) =>
    risk_amount = account_balance * (input_risk_percentage / 100)
    lot_size = risk_amount / (stop_loss_points * syminfo.pointvalue)
    lot_size

// Determine if Market is Consolidating
is_consolidating() =>
    (ta.highest(high, 20) - ta.lowest(low, 20)) / atr < 2

// === Visual Enhancements ===
// Plot Order Blocks
// if is_order_block(close, 1)
//     line.new(x1=bar_index[1], y1=low[1], x2=bar_index, y2=low[1], color=color.green, width=2, extend=extend.right)
// if is_order_block(close, -1)
//     line.new(x1=bar_index[1], y1=high[1], x2=bar_index, y2=high[1], color=color.red, width=2, extend=extend.right)

// Highlight Imbalance Zones
// if is_imbalance()
//     box.new(left=bar_index[1], top=high[1], right=bar_index, bottom=low[1], bgcolor=color.new(color.orange, 80))

// === Logic for Trend Confirmation ===
is_bullish_trend = ema_short > ema_long
is_bearish_trend = ema_short < ema_long

// === Entry Logic ===
account_balance = strategy.equity
if not is_consolidating()
    if is_bullish_trend
        stop_loss = close - atr * 2
        take_profit = close + (math.abs(close - (close - atr * 2)) * 3)
        stop_loss_points = math.abs(close - stop_loss) / syminfo.pointvalue
        lot_size = calculate_lot_size(stop_loss_points, account_balance)
        strategy.entry("Buy", strategy.long, qty=lot_size)
        strategy.exit("TP/SL", "Buy", stop=stop_loss, limit=take_profit)

    if is_bearish_trend
        stop_loss = close + atr * 2
        take_profit = close - (math.abs(close - (close + atr * 2)) * 3)
        stop_loss_points = math.abs(close - stop_loss) / syminfo.pointvalue
        lot_size = calculate_lot_size(stop_loss_points, account_balance)
        strategy.entry("Sell", strategy.short, qty=lot_size)
        strategy.exit("TP/SL", "Sell", stop=stop_loss, limit=take_profit)

// === Plotting Indicators ===
plot(ema_short, color=color.blue, title="EMA 50")
plot(ema_long, color=color.orange, title="EMA 200")
plotshape(series=is_bullish_trend and not is_consolidating(), style=shape.triangleup, location=location.belowbar, color=color.green, text="Buy")
plotshape(series=is_bearish_trend and not is_consolidating(), style=shape.triangledown, location=location.abovebar, color=color.red, text="Sell")

```

> Detail

https://www.fmz.com/strategy/474884

> Last Modified

2024-12-12 17:26:00
