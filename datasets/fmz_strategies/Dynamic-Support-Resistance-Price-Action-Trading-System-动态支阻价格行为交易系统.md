
> Name

Dynamic-Support-Resistance-Price-Action-Trading-System-动态支阻价格行为交易系统

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/b347b8db5b73db3d44.png)

[trans]
#### 概述
该策略是一个基于价格行为和动态支撑阻力位的交易系统,通过识别关键价格形态在支撑和阻力位附近进行交易。系统采用16周期的动态支撑阻力计算方法,结合四种经典的反转蜡烛图形态 - 锤子线、流星线、十字星和针形态来捕捉市场潜在的反转机会。策略采用固定百分比的止盈止损来管理风险,并使用敏感度参数来控制入场信号的严格程度。

#### 策略原理
策略的核心是通过动态计算支撑位和阻力位,形成一个价格活动的上下边界。当价格接近这些关键水平时,系统会寻找特定的蜡烛图形态作为反转信号。入场条件需要价格在支撑阻力位的1.8%范围内(默认敏感度)出现反转形态。系统使用35%的资金管理规则,配合16%的止损和9.5%的止盈,有效控制每笔交易的风险在账户总额的5.6%左右。策略通过Pine Script实现,包含了完整的交易管理功能和可视化显示。

#### 策略优势
1. 策略结合了技术分析中最可靠的两个要素:价格形态和支撑阻力,提高了交易信号的可靠性
2. 使用动态计算的支撑阻力位,能够适应市场条件的变化
3. 采用严格的资金管理和风险控制措施,有效防止大幅回撤
4. 策略逻辑清晰,参数可调整性强,便于根据不同市场情况优化
5. 入场信号明确,无主观判断成分,适合自动化交易

#### 策略风险
1. 在高波动市场中,支撑阻力位的有效性可能降低
2. 止损位置相对较远(16%),在剧烈行情中可能承受较大损失
3. 敏感度参数的设置对交易频率和准确性有重要影响
4. 仅依赖价格形态可能错过其他重要的市场信号
5. 需要考虑交易成本对策略收益的影响

#### 策略优化方向
1. 引入成交量作为辅助确认指标,提高信号可靠性
2. 开发自适应的敏感度参数,根据市场波动性动态调整
3. 优化止损设置,考虑使用移动止损或分段止损方案
4. 增加趋势过滤器,避免在强趋势中做反转交易
5. 开发动态仓位管理系统,根据市场情况调整交易规模

#### 总结
这个基于价格行为的交易策略通过结合动态支撑阻力位和经典反转形态,为交易者提供了一个系统化的交易方法。策略的优势在于逻辑清晰、风险可控,但仍需要根据实际交易效果进行持续优化。建议交易者在实盘使用前进行充分的回测和参数优化,并结合市场经验对策略进行个性化调整。 ||

#### Overview
This strategy is a trading system based on price action and dynamic support/resistance levels, executing trades near key price levels when specific candlestick patterns emerge. The system utilizes a 16-period dynamic support/resistance calculation method, combined with four classic reversal candlestick patterns - Hammer, Shooting Star, Doji, and Pin Bar to capture potential market reversals. The strategy employs fixed percentage take-profit and stop-loss levels for risk management and uses a sensitivity parameter to control entry signal strictness.

#### Strategy Principles
The core of the strategy lies in dynamically calculating support and resistance levels to establish price movement boundaries. When price approaches these key levels, the system looks for specific candlestick patterns as reversal signals. Entry conditions require pattern formation within 1.8% (default sensitivity) of support/resistance levels. The system implements a 35% equity management rule with 16% stop-loss and 9.5% take-profit, effectively controlling risk at around 5.6% of total equity per trade. The strategy is implemented in Pine Script with complete trade management functionality and visualization.

#### Strategy Advantages
1. Combines two most reliable elements of technical analysis: price patterns and support/resistance, enhancing signal reliability
2. Uses dynamically calculated support/resistance levels, adapting to changing market conditions
3. Implements strict money management and risk control measures to prevent significant drawdowns
4. Clear strategy logic with adjustable parameters, facilitating optimization for different market conditions
5. Clear entry signals without subjective judgment, suitable for automated trading

#### Strategy Risks
1. Support/resistance effectiveness may decrease in highly volatile markets
2. Relatively wide stop-loss (16%) may lead to significant losses in volatile conditions
3. Sensitivity parameter settings significantly impact trading frequency and accuracy
4. Relying solely on price patterns may miss other important market signals
5. Need to consider trading costs' impact on strategy returns

#### Optimization Directions
1. Introduce volume as a confirmation indicator to improve signal reliability
2. Develop adaptive sensitivity parameters that adjust dynamically based on market volatility
3. Optimize stop-loss settings, consider implementing trailing stops or stepped stop-loss plans
4. Add trend filters to avoid reversal trades during strong trends
5. Develop dynamic position sizing system adjusting trade size based on market conditions

#### Summary
This price action-based trading strategy provides traders with a systematic trading approach by combining dynamic support/resistance levels with classic reversal patterns. The strategy's strengths lie in its clear logic and controllable risk, though continuous optimization based on actual trading results is necessary. Traders are advised to conduct thorough backtesting and parameter optimization before live trading, and customize the strategy based on market experience.[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-11-26 00:00:00
end: 2024-12-03 00:00:00
period: 1h
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © felipemiransan

//@version=5
strategy("Price Action Strategy", overlay=true)

// Settings
length = input.int(16, title="Support and Resistance Length")
sensitivity = input.float(0.018, title="Sensitivity")

// Stop Loss and Take Profit
stop_loss_pct = input.float(16, title="Stop Loss percentage", minval=0.1) / 100
take_profit_pct = input.float(9.5, title="Take Profit percentage", minval=0.1) / 100

// Function to identify a Hammer
isHammer() =>
    body = close - open
    price_range = high - low
    lower_shadow = open - low
    upper_shadow = high - close
    body > 0 and lower_shadow > body * 2 and upper_shadow < body * 0.5 and price_range > 0

// Function to identify a Shooting Star
isShootingStar() =>
    body = open - close
    price_range = high - low
    lower_shadow = close - low
    upper_shadow = high - open
    body > 0 and upper_shadow > body * 2 and lower_shadow < body * 0.5 and price_range > 0

// Function to identify a Doji
isDoji() =>
    body = close - open
    price_range = high - low
    math.abs(body) < (price_range * 0.1)  // Doji has a small body

// Function to identify a Pin Bar
isPinBar() =>
    body = close - open
    price_range = high - low
    lower_shadow = open - low
    upper_shadow = high - close
    (upper_shadow > body * 2 and lower_shadow < body * 0.5) or (lower_shadow > body * 2 and upper_shadow < body * 0.5)

// Support and resistance levels 
support = ta.lowest(low, length)
resistance = ta.highest(high, length)

// Entry criteria
long_condition = (isHammer() or isDoji() or isPinBar()) and close <= support * (1 + sensitivity)
short_condition = (isShootingStar() or isDoji() or isPinBar()) and close >= resistance * (1 - sensitivity)

// Function to calculate stop loss and take profit (long)
calculate_levels(position_size, avg_price, stop_loss_pct, take_profit_pct) =>
    stop_loss_level = avg_price * (1 - stop_loss_pct)
    take_profit_level = avg_price * (1 + take_profit_pct)
    [stop_loss_level, take_profit_level]

// Function to calculate stop loss and take profit (short)
calculate_levels_short(position_size, avg_price, stop_loss_pct, take_profit_pct) =>
    stop_loss_level = avg_price * (1 + stop_loss_pct)
    take_profit_level = avg_price * (1 - take_profit_pct)
    [stop_loss_level, take_profit_level]

// Buy entry order with label
if (long_condition and strategy.opentrades == 0)
    strategy.entry("Buy", strategy.long)
    pattern = isHammer() ? "Hammer" : isDoji() ? "Doji" : isPinBar() ? "Pin Bar" : ""
    label.new(x=bar_index, y=low, text=pattern, color=color.green, textcolor=color.black, size=size.small)

// Sell entry order with label
if (short_condition and strategy.opentrades == 0)
    strategy.entry("Sell", strategy.short)
    pattern = isShootingStar() ? "Shooting Star" : isDoji() ? "Doji" : isPinBar() ? "Pin Bar" : ""
    label.new(x=bar_index, y=high, text=pattern, color=color.red, textcolor=color.black, size=size.small)

// Stop Loss and Take Profit management for open positions
if (strategy.opentrades > 0)
    if (strategy.position_size > 0)  // Long position
        avg_price_long = strategy.position_avg_price  // Average price of long position
        [long_stop_level, long_take_profit_level] = calculate_levels(strategy.position_size, avg_price_long, stop_loss_pct, take_profit_pct)
        strategy.exit("Exit Long", from_entry="Buy", stop=long_stop_level, limit=long_take_profit_level)
    if (strategy.position_size < 0)  // Short position
        avg_price_short = strategy.position_avg_price  // Average price of short position
        [short_stop_level, short_take_profit_level] = calculate_levels_short(strategy.position_size, avg_price_short, stop_loss_pct, take_profit_pct)
        strategy.exit("Exit Short", from_entry="Sell", stop=short_stop_level, limit=short_take_profit_level)

// Visualization of Support and Resistance Levels
plot(support, title="Support", color=color.green, linewidth=2)
plot(resistance, title="Resistance", color=color.red, linewidth=2)





```

> Detail

https://www.fmz.com/strategy/473937

> Last Modified

2024-12-04 15:19:00
