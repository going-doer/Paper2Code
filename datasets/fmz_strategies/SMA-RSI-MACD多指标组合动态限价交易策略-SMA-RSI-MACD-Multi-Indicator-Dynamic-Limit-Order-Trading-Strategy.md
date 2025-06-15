
> Name

SMA-RSI-MACD多指标组合动态限价交易策略-SMA-RSI-MACD-Multi-Indicator-Dynamic-Limit-Order-Trading-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1947a3f1a718f6364f2.png)

[trans]
#### 概述
该策略是一个结合了多个技术指标的交易系统,主要基于EMA均线交叉、RSI超卖和MACD金叉三重信号确认来开仓,通过动态限价单入场和多重退出机制来管理风险。策略采用9周期和21周期的指数移动平均线(EMA)作为主要趋势指标,结合相对强弱指数(RSI)和移动平均线趋同背离指标(MACD)来过滤交易信号,通过设定限价单距离和固定止盈止损点数来控制风险。

#### 策略原理
策略的核心交易逻辑包括以下几个关键部分：
1. 入场信号基于9周期EMA上穿21周期EMA时触发
2. 入场价格设置在9周期EMA下方指定点数的限价单
3. 交易确认需同时满足RSI低于设定阈值和MACD金叉
4. 出场信号包括MACD死叉、固定止盈止损点数和收盘强制平仓
5. 交易时间限制在早上9:30后到下午3:10前

策略采用限价单入场的方式,可以在更好的价格位置建仓,通过多重技术指标的配合来提高交易的准确性。

#### 策略优势
1. 多重信号确认机制提高了交易的可靠性
2. 限价单入场方式可以获得更好的成交价格
3. 固定止盈止损点数便于风险控制
4. 收盘强制平仓避免隔夜风险
5. 交易时间限制避开了开盘波动
6. EMA指标对趋势反应更快速
7. RSI和MACD的配合可以过滤假信号

#### 策略风险
1. 多重信号确认可能导致错过部分交易机会
2. 限价单可能因价格快速上涨而未能成交
3. 固定点数止损可能在高波动时期损失较大
4. MACD信号可能出现滞后
5. 策略未考虑市场波动率变化的影响
6. 参数优化可能存在过度拟合风险

#### 策略优化方向
1. 引入自适应的止损止盈点数,根据市场波动率动态调整
2. 增加成交量指标作为辅助确认信号
3. 考虑加入趋势强度过滤器
4. 优化限价单距离的计算方法,可考虑使用ATR动态调整
5. 增加市场情绪指标来过滤不利的市场环境
6. 加入仓位管理机制,根据信号强度调整开仓量

#### 总结
这是一个结构完整、逻辑清晰的多指标交易策略,通过均线系统识别趋势,RSI和MACD过滤信号,限价单和多重止损机制控制风险。策略的优势在于信号可靠性高、风险控制完善,但也存在信号滞后和参数优化等问题。通过引入动态参数调整和增加辅助指标,策略还有较大的优化空间。适合稳健型投资者在趋势明确的市场环境中使用。

|| 

#### Overview
This strategy is a multi-technical indicator trading system that primarily uses EMA crossover, RSI oversold conditions, and MACD golden cross for trade confirmation. It employs dynamic limit orders for entry and multiple exit mechanisms for risk management. The strategy uses 9-period and 21-period Exponential Moving Averages (EMA) as primary trend indicators, combined with Relative Strength Index (RSI) and Moving Average Convergence Divergence (MACD) to filter trading signals.

#### Strategy Principles
The core trading logic includes the following key components:
1. Entry signals are triggered when 9-period EMA crosses above 21-period EMA
2. Entry price is set as a limit order below the 9-period EMA at a specified offset
3. Trade confirmation requires RSI below threshold and MACD golden cross
4. Exit signals include MACD death cross, fixed profit/loss points, and forced closing at market end
5. Trading time is restricted between 9:30 AM and 3:10 PM

The strategy uses limit orders for entry to achieve better entry prices and combines multiple technical indicators to improve trading accuracy.

#### Strategy Advantages
1. Multiple signal confirmation mechanism improves trade reliability
2. Limit order entries provide better execution prices
3. Fixed profit/loss points facilitate risk control
4. Forced closing at market end eliminates overnight risk
5. Trading time restrictions avoid opening volatility
6. EMA indicators provide faster trend response
7. RSI and MACD combination helps filter false signals

#### Strategy Risks
1. Multiple signal confirmation may cause missed opportunities
2. Limit orders might not execute in rapid price movements
3. Fixed point stops may result in larger losses during high volatility
4. MACD signals may lag behind price action
5. Strategy doesn't account for changes in market volatility
6. Parameter optimization may lead to overfitting

#### Strategy Optimization Directions
1. Introduce adaptive stop-loss and take-profit points based on market volatility
2. Add volume indicators as additional confirmation signals
3. Consider adding trend strength filters
4. Optimize limit order offset calculation using ATR
5. Include market sentiment indicators to filter unfavorable conditions
6. Add position sizing mechanism based on signal strength

#### Summary
This is a well-structured multi-indicator trading strategy that identifies trends using moving averages, filters signals with RSI and MACD, and controls risk through limit orders and multiple stop mechanisms. The strategy's strengths lie in its signal reliability and comprehensive risk control, though it faces challenges with signal lag and parameter optimization. There is significant room for improvement through dynamic parameter adjustment and additional auxiliary indicators. It is suitable for conservative investors in trending market conditions.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2019-12-23 08:00:00
end: 2024-12-09 08:00:00
period: 1d
basePeriod: 1d
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("SMA 9 & 21 with RSI and MACD Buy Strategy", overlay=true)

// Inputs for Simple Moving Averages
sma_short = ta.ema(close, 9)
sma_long = ta.ema(close, 21)

// Plotting SMA
plot(sma_short, color=color.green, title="SMA 9")
plot(sma_long, color=color.red, title="SMA 21")

// RSI Calculation
rsi_length = input.int(14, title="RSI Length")
rsi_threshold = input.int(70, title="RSI Threshold")
rsi = ta.rsi(close, rsi_length)

// MACD Calculation
macd_fast = input.int(8, title="MACD Fast Length")
macd_slow = input.int(18, title="MACD Slow Length")
macd_signal = input.int(6, title="MACD Signal Length")
[macd_line, signal_line, _] = ta.macd(close, macd_fast, macd_slow, macd_signal)

// Inputs for Limit Order Offset
limit_offset = input.int(50, title="Limit Order Offset", minval=1)  // 50 points below 9 EMA

// User input for specific date
simulationStartDate = input(timestamp("2024-12-01 00:00"), title="Simulation Start Date", group = "Simulation Dates")
simulationEndDate = input(timestamp("2024-12-30 00:00"), title="Simulation End Date", group = "Simulation Dates")

// Declare limit_price as float
var float limit_price = na

// Calculate Limit Order Price
if (sma_short[1] < sma_long[1] and sma_short > sma_long)  // 9 EMA crosses above 21 EMA
    limit_price := sma_short - limit_offset

// Buy Signal Condition (only on the specified date)
buy_condition = not na(limit_price) and rsi < rsi_threshold and ta.crossover(macd_line, signal_line) 

// Sell Signal Condition (MACD crossover down)
sell_condition = ta.crossunder(macd_line, signal_line)

// Track Entry Price for Point-Based Exit
var float entry_price = na

if (buy_condition )
    strategy.order("Buy", strategy.long, comment="Limit Order at 9 EMA - Offset", limit=limit_price)
    label.new(bar_index, limit_price, "Limit Buy", style=label.style_label_up, color=color.green, textcolor=color.white)
    entry_price := limit_price  // Set entry price

// Exit Conditions
exit_by_macd = sell_condition
exit_by_points = not na(entry_price) and ((close >= entry_price + 12) or (close <= entry_price - 12))  // Adjust as per exit points

// Exit all positions at the end of the day
if hour == 15 and minute > 10 and strategy.position_size > 0
    strategy.close_all()  // Close all positions at the end of the day
    strategy.cancel_all()  

// Exit based on sell signal or point movement
if (exit_by_macd or exit_by_points  and strategy.position_size > 0 )
    strategy.close("Buy")
    label.new(bar_index, close, "Close", style=label.style_label_down, color=color.red, textcolor=color.white)

 
```

> Detail

https://www.fmz.com/strategy/474678

> Last Modified

2024-12-11 15:15:49
