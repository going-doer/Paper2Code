
> Name

Trend-Following-with-Breakout-and-Frequency-Filter-Long-Only-基于突破和频率过滤的趋势跟踪策略只做多

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1341cec80f9b70c0cf0.png)

[trans]
#### 概述
该策略是一个基于突破和频率过滤的趋势跟踪策略,只进行多头交易。策略的主要思路是利用EMA指标判断当前趋势方向,在价格突破一定区间内的最高价时产生做多信号,同时使用频率过滤器来控制交易频率,避免过于频繁开仓。策略还设置了止损点来控制风险,并在趋势结束时平仓。

#### 策略原理
1. 计算EMA指标,用于判断当前趋势方向。当收盘价在EMA之上时,认为当前趋势是多头。
2. 计算一定区间内的最高价,作为突破条件。当收盘价突破最短回溯期或最长回溯期内的最高价,并且当前趋势为多头时,产生做多信号。
3. 引入频率过滤器,控制连续开仓的最小间隔时间,避免交易频率过高。
4. 设置止损点,当价格跌破止损价时平仓,控制风险。
5. 定义趋势结束信号,当收盘价跌破EMA时,认为趋势结束,此时如果持有多单则平仓。

#### 策略优势
1. 趋势跟踪:通过EMA指标判断趋势方向,顺应趋势交易,有助于提高策略收益。
2. 突破确认:使用价格突破作为入场信号,能够在趋势初期就及时入场,捕捉更多的利润空间。
3. 频率控制:引入频率过滤器,控制连续开仓的时间间隔,避免过于频繁交易,减少交易成本和风险。
4. 止损保护:设置止损点,当价格反向波动达到一定幅度时及时止损,有效控制下行风险。
5. 动态平仓:根据趋势结束信号动态平仓,能够及时锁定已有收益,避免趋势反转带来的损失。

#### 策略风险
1. 参数敏感:策略的表现对参数选择较为敏感,不同的参数设置可能导致策略表现差异较大。需要对参数进行充分的回测和优化。
2. 突破失败:价格突破并不能保证趋势一定会延续,有可能出现突破失败的情况,导致策略出现连续亏损。
3. 趋势识别:策略依赖EMA指标判断趋势,但EMA指标可能出现滞后或误判的情况,影响策略的准确性。
4. 频繁交易:尽管策略引入了频率过滤器,但在市场波动较大时,仍可能出现频繁开仓平仓的情况,增加交易成本。
5. 止损风险:止损点的设置可能无法完全避免策略的最大回撤,在极端行情下仍有可能出现较大亏损。

#### 策略优化方向
1. 参数优化:对策略的关键参数如EMA长度、回溯期长度、止损百分比等进行优化,寻找最佳参数组合,提高策略稳定性和收益性。
2. 信号过滤:在突破信号产生后,可以引入其他技术指标或条件对信号进行二次确认,提高信号质量,减少误判和虚假信号。
3. 趋势判断:可以尝试使用其他趋势判断指标如MACD、DMI等,或者结合多个指标共同判断趋势,提高趋势识别的准确性。
4. 动态止损:根据市场波动情况动态调整止损点,例如使用ATR指标计算动态止损价,或者引入追踪止损策略,更好地控制风险。
5. 头寸管理:优化头寸管理策略,根据市场波动和账户资金情况动态调整头寸大小,控制单笔交易风险敞口,提高资金利用效率。

#### 总结
该策略是一个基于突破和频率过滤的趋势跟踪策略,通过EMA指标判断趋势方向,使用价格突破作为入场信号,同时引入频率过滤器控制交易频率,并设置止损点控制风险。策略的优势在于趋势跟踪、突破确认、频率控制、止损保护和动态平仓,但同时也存在参数敏感、突破失败、趋势识别、频繁交易和止损风险等潜在风险。为进一步优化策略,可以从参数优化、信号过滤、趋势判断、动态止损和头寸管理等方面入手,提高策略的稳定性和盈利能力。

|| 

#### Overview
This strategy is a trend following strategy based on breakout and frequency filtering, only taking long positions. The main idea of the strategy is to use the EMA indicator to determine the current trend direction, generate a long signal when the price breaks out of the highest price within a certain range, and use a frequency filter to control the trading frequency to avoid opening positions too frequently. The strategy also sets a stop loss point to control risk and closes positions when the trend ends.

#### Strategy Principle
1. Calculate the EMA indicator to determine the current trend direction. When the closing price is above the EMA, it is considered a bullish trend.
2. Calculate the highest price within a certain range as the breakout condition. When the closing price breaks out of the highest price within the shortest or longest lookback period and the current trend is bullish, a long signal is generated.
3. Introduce a frequency filter to control the minimum interval time between consecutive position openings to avoid excessive trading frequency.
4. Set a stop loss point. When the price falls below the stop loss price, close the position to control risk.
5. Define the trend end signal. When the closing price falls below the EMA, the trend is considered to have ended. If a long position is held at this time, close the position.

#### Strategy Advantages
1. Trend following: By using the EMA indicator to determine the trend direction and trading in line with the trend, it helps to improve strategy returns.
2. Breakout confirmation: Using price breakout as the entry signal allows for timely entry at the beginning of the trend, capturing more profit potential.
3. Frequency control: Introducing a frequency filter to control the time interval between consecutive position openings avoids excessive trading and reduces trading costs and risks.
4. Stop loss protection: Setting a stop loss point to promptly stop loss when the price moves in the opposite direction by a certain magnitude effectively controls downside risk.
5. Dynamic position closing: Dynamically closing positions based on the trend end signal allows for timely locking in of existing profits and avoids losses caused by trend reversal.

#### Strategy Risks
1. Parameter sensitivity: The performance of the strategy is relatively sensitive to parameter selection, and different parameter settings may lead to significant differences in strategy performance. Sufficient backtesting and optimization of parameters are required.
2. Breakout failure: Price breakouts do not guarantee that the trend will definitely continue, and there may be cases of breakout failure, resulting in consecutive losses for the strategy.
3. Trend recognition: The strategy relies on the EMA indicator to judge the trend, but the EMA indicator may experience lag or misjudgment, affecting the accuracy of the strategy.
4. Frequent trading: Although the strategy introduces a frequency filter, frequent position opening and closing may still occur when market volatility is high, increasing trading costs.
5. Stop loss risk: The setting of the stop loss point may not completely avoid the maximum drawdown of the strategy, and large losses may still occur in extreme market conditions.

#### Strategy Optimization Directions
1. Parameter optimization: Optimize key parameters of the strategy, such as EMA length, lookback period length, stop loss percentage, etc., to find the optimal parameter combination and improve strategy stability and profitability.
2. Signal filtering: After the breakout signal is generated, other technical indicators or conditions can be introduced to confirm the signal a second time, improving signal quality and reducing misjudgments and false signals.
3. Trend judgment: Try using other trend judgment indicators such as MACD, DMI, etc., or combine multiple indicators to jointly judge the trend and improve the accuracy of trend recognition.
4. Dynamic stop loss: Dynamically adjust the stop loss point according to market volatility conditions, such as using the ATR indicator to calculate the dynamic stop loss price or introducing a trailing stop loss strategy to better control risk.
5. Position management: Optimize the position management strategy, dynamically adjust the position size according to market volatility and account capital conditions, control the risk exposure of a single transaction, and improve capital utilization efficiency.

#### Summary
This strategy is a trend following strategy based on breakout and frequency filtering. It uses the EMA indicator to determine the trend direction, uses price breakout as the entry signal, introduces a frequency filter to control the trading frequency, and sets a stop loss point to control risk. The advantages of the strategy lie in trend following, breakout confirmation, frequency control, stop loss protection, and dynamic position closing, but it also has potential risks such as parameter sensitivity, breakout failure, trend recognition, frequent trading, and stop loss risk. To further optimize the strategy, we can start from aspects such as parameter optimization, signal filtering, trend judgment, dynamic stop loss, and position management to improve the stability and profitability of the strategy.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-05-22 00:00:00
end: 2024-05-27 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Trend Following with Breakout and Frequency Filter (Long Only)", overlay=true)

// 输入参数
emaLength = input.int(50, title="EMA长度")
lookbackPeriodMin = input.int(80, title="最短回溯期")
lookbackPeriodMax = input.int(120, title="最长回溯期")
stopLossPct = input.float(2, title="止损百分比") / 100  // 止损百分比
minHoldBars = input.int(10, title="最小持仓K线数量")  // 最小持仓K线数量

// 计算EMA
ema = ta.ema(close, emaLength)

// 计算最高价和最低价
highestHigh = ta.highest(high, lookbackPeriodMax)
lowestLow = ta.lowest(low, lookbackPeriodMax)

// 定义趋势方向
isBullish = close > ema

// 定义突破信号
breakoutCondition = (ta.crossover(close, highestHigh[lookbackPeriodMin]) or ta.crossover(close, highestHigh[lookbackPeriodMax])) and isBullish

// 计算止损点
stopLossLevelLong = close * (1 - stopLossPct)

// 绘制EMA
plot(ema, title="EMA", color=color.blue)

// 记录上次开仓时间
var float lastEntryTime = na

// 策略执行并标注信号
if (breakoutCondition and (na(lastEntryTime) or (time - lastEntryTime) > minHoldBars * timeframe.multiplier))
    strategy.entry("做多", strategy.long)
    label.new(bar_index, high, text="买入", style=label.style_label_up, color=color.green, textcolor=color.white)
    strategy.exit("止损", from_entry="做多", stop=stopLossLevelLong)
    lastEntryTime := time

// 定义趋势结束信号
exitCondition = close < ema

if (exitCondition and (strategy.position_size > 0) and (time - lastEntryTime) > minHoldBars * timeframe.multiplier)
    strategy.close("做多")
    label.new(bar_index, low, text="卖出", style=label.style_label_down, color=color.red, textcolor=color.white)
```

> Detail

https://www.fmz.com/strategy/452714

> Last Modified

2024-05-28 14:00:24
