
> Name

动态移动止损双目标价均线交叉策略-Dynamic-Trailing-Stop-Dual-Target-Moving-Average-Crossover-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1d4c9bbca9c0d2a50e3.png)

[trans]
#### 概述

本策略是一个基于移动平均线交叉的交易系统,结合了动态移动止损和双目标利润点的风险管理方法。策略主要依据价格与200期移动平均线的交叉来判断入场时机,同时设置了灵活的止损和获利点,以实现风险控制和利润最大化。

#### 策略原理

1. 入场信号:
   - 多头入场:当价格向上穿过200期移动平均线时
   - 空头入场:当价格向下穿过200期移均线时

2. 风险管理:
   - 初始止损:设置在入场价格的500个点位之外
   - 动态移动止损:当价格向有利方向移动200个点位时,止损会移动到入场价格处
   
3. 利润目标:
   - 第一目标:当价格达到入场价格3000个点位时,平掉75%的仓位
   - 第二目标:剩余25%的仓位会在价格达到入场价格4000个点位时平仓
   - 如果触发动态移动止损,剩余仓位的止损点会设置在入场价格处

4. 仓位管理:
   - 每次交易固定数量的100个单位

#### 策略优势

1. 趋势跟踪:利用移动平均线捕捉市场趋势,有助于在大趋势中获利。

2. 风险控制:采用初始止损和动态移动止损相结合的方式,既限制了最大损失,又能保护已获利润。

3. 利润最大化:通过设置两个目标价,在保证部分利润的同时,还可以继续追踪大趋势。

4. 自动化:策略完全自动化,减少了人为情绪干扰。

5. 灵活性:各项参数如移动平均线周期、止损点、获利点等都可以根据市场情况进行调整。

#### 策略风险

1. 震荡市风险:在横盘震荡市场中,可能会频繁触发假突破信号,导致连续亏损。

2. 滑点风险:在快速行情中,实际成交价可能与理想价格有较大偏差。

3. 过度交易:频繁的交叉信号可能导致过度交易,增加交易成本。

4. 单一指标依赖:仅依赖移动平均线可能会忽视其他重要的市场信息。

5. 固定仓位风险:每次交易固定数量可能不适合所有市场环境。

#### 策略优化方向

1. 多指标结合:考虑引入其他技术指标如RSI、MACD等,与移动平均线结合使用,提高入场信号的可靠性。

2. 动态仓位管理:根据市场波动性和账户余额动态调整交易数量,以更好地控制风险。

3. 市场环境过滤:增加趋势强度指标或波动率指标,在不适合交易的市场环境中避免入场。

4. 参数优化:使用历史数据回测不同的参数组合,找出最优的移动平均线周期、止损点和获利点设置。

5. 时间过滤:考虑加入时间过滤器,避免在波动较大或流动性较差的时间段交易。

6. 加入基本面因素:结合重要经济数据发布或其他基本面事件,调整策略的进出场时机。

#### 总结

动态移动止损双目标价均线交叉策略是一个结合了技术分析和风险管理的量化交易系统。通过移动平均线捕捉市场趋势,同时利用动态止损和多重获利目标来平衡风险和收益。该策略的主要优势在于其自动化程度高、风险控制灵活,并且有潜力在强趋势市场中获得可观收益。然而,使用者需要注意应对震荡市场的风险,并考虑进一步优化策略以提高其适应性和稳定性。通过持续的参数调整、引入额外的市场指标,以及考虑更复杂的仓位管理方法,该策略有望在各种市场环境中取得更好的表现。

|| 

#### Overview

This strategy is a trading system based on moving average crossovers, combining dynamic trailing stops and dual profit targets for risk management. The strategy primarily uses the crossover of price with a 200-period moving average to determine entry points, while implementing flexible stop-loss and take-profit levels to achieve risk control and profit maximization.

#### Strategy Principles

1. Entry Signals:
   - Long Entry: When price crosses above the 200-period moving average
   - Short Entry: When price crosses below the 200-period moving average

2. Risk Management:
   - Initial Stop Loss: Set 500 points away from the entry price
   - Dynamic Trailing Stop: When price moves 200 points in the favorable direction, the stop loss is moved to the entry price
   
3. Profit Targets:
   - First Target: When price reaches 3000 points from the entry, 75% of the position is closed
   - Second Target: The remaining 25% of the position is closed when price reaches 4000 points from the entry
   - If the dynamic trailing stop is triggered, the stop loss for the remaining position is set at the entry price

4. Position Management:
   - Fixed quantity of 100 units per trade

#### Strategy Advantages

1. Trend Following: Utilizes moving averages to capture market trends, aiding in profiting from major market movements.

2. Risk Control: Combines initial stop loss with dynamic trailing stop, limiting maximum loss while protecting accrued profits.

3. Profit Maximization: By setting two target prices, it secures partial profits while continuing to track larger trends.

4. Automation: The strategy is fully automated, reducing emotional interference in trading decisions.

5. Flexibility: Various parameters such as moving average period, stop loss points, and profit targets can be adjusted according to market conditions.

#### Strategy Risks

1. Choppy Market Risk: In range-bound, oscillating markets, frequent false breakout signals may lead to consecutive losses.

2. Slippage Risk: In fast-moving markets, actual execution prices may significantly deviate from ideal prices.

3. Overtrading: Frequent crossover signals may result in excessive trading, increasing transaction costs.

4. Single Indicator Dependency: Relying solely on moving averages may overlook other important market information.

5. Fixed Position Risk: Trading a fixed quantity for each trade may not be suitable for all market environments.

#### Strategy Optimization Directions

1. Multi-Indicator Integration: Consider introducing other technical indicators such as RSI, MACD, etc., to be used in conjunction with moving averages to improve entry signal reliability.

2. Dynamic Position Sizing: Adjust trading quantity based on market volatility and account balance to better control risk.

3. Market Environment Filtering: Add trend strength or volatility indicators to avoid entries in unfavorable market conditions.

4. Parameter Optimization: Use historical data to backtest different parameter combinations to find optimal settings for moving average periods, stop loss points, and profit targets.

5. Time Filtering: Consider adding time filters to avoid trading during highly volatile or low liquidity periods.

6. Fundamental Factor Integration: Incorporate important economic data releases or other fundamental events to adjust strategy entry and exit timing.

#### Conclusion

The Dynamic Trailing Stop Dual Target Moving Average Crossover Strategy is a quantitative trading system that combines technical analysis with risk management. It captures market trends using moving averages while balancing risk and reward through dynamic stop losses and multiple profit targets. The strategy's main advantages lie in its high degree of automation, flexible risk control, and potential for significant returns in strong trend markets. However, users need to be aware of the risks in choppy markets and consider further optimizations to improve adaptability and stability. Through continuous parameter adjustment, introduction of additional market indicators, and consideration of more complex position management methods, this strategy has the potential to perform well in various market environments.

[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-07-29 00:00:00
end: 2024-07-28 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("SOL/USDT Trading Strategy", overlay=true)

// Параметры стратегии
input_quantity = input(2, title="Trade Size (SOL)")
stop_loss_points = input(500, title="Stop Loss Points")
take_profit_points_1 = input(3000, title="First Take Profit Points")
take_profit_points_2 = input(4000, title="Second Take Profit Points")
move_stop_to_entry_points = input(200, title="Move Stop to Entry Points")
ma_period = input(180, title="MA Period")

// Расчет скользящей средней
ma = ta.sma(close, ma_period)

// Условия входа в сделку
long_condition = ta.crossover(close, ma)
short_condition = ta.crossunder(close, ma)

// Текущая цена
var float entry_price = na

// Логика открытия и закрытия сделок
if (long_condition)
    entry_price := close
    strategy.entry("Long", strategy.long, qty=input_quantity)
if (short_condition)
    entry_price := close
    strategy.entry("Short", strategy.short, qty=input_quantity)

// Логика выхода из сделок
if (strategy.position_size > 0)
    if (close >= entry_price + take_profit_points_1 * syminfo.mintick)
        strategy.exit("Partial Take Profit", "Long", qty=0.75 * input_quantity, limit=close)
        strategy.exit("Remaining Take Profit", "Long", qty=0.25 * input_quantity, limit=entry_price + take_profit_points_2 * syminfo.mintick, stop=entry_price)

    if (close >= entry_price + move_stop_to_entry_points * syminfo.mintick)
        strategy.exit("Stop Loss at Entry", "Long", qty=strategy.position_size, stop=entry_price)
    else
        strategy.exit("Take Profit/Stop Loss", "Long", stop=entry_price - stop_loss_points * syminfo.mintick, limit=entry_price + take_profit_points_1 * syminfo.mintick)

if (strategy.position_size < 0)
    if (close <= entry_price - take_profit_points_1 * syminfo.mintick)
        strategy.exit("Partial Take Profit", "Short", qty=0.75 * input_quantity, limit=close)
        strategy.exit("Remaining Take Profit", "Short", qty=0.25 * input_quantity, limit=entry_price - take_profit_points_2 * syminfo.mintick, stop=entry_price)

    if (close <= entry_price - move_stop_to_entry_points * syminfo.mintick)
        strategy.exit("Stop Loss at Entry", "Short", qty=strategy.position_size, stop=entry_price)
    else
        strategy.exit("Take Profit/Stop Loss", "Short", stop=entry_price + stop_loss_points * syminfo.mintick, limit=entry_price - take_profit_points_1 * syminfo.mintick)

// Отображение скользящей средней
plot(ma, title="200 MA", color=color.blue)

```

> Detail

https://www.fmz.com/strategy/458045

> Last Modified

2024-07-29 14:40:23
