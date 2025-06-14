
> Name

艾略特波浪理论4-9脉冲波自动检测交易策略-Elliott-Wave-Theory-4-9-Impulse-Wave-Automatic-Detection-Trading-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/202eb993d2b219ab2ef.png)

[trans]
#### 概述
该策略基于艾略特波浪理论,尝试自动检测脉冲波。它通过寻找连续4根收盘价上涨且当前收盘价高于9天前收盘价的K线组合来定义上升脉冲波;用相反的逻辑来定义下降脉冲波。检测到脉冲波后会产生买卖信号并反转仓位,止损位设在信号K线的低点或高点。由于脉冲波通常伴随快速移动,这种止损方式应该会带来正面结果。另外在强趋势开始前,绿色或红色三角形的聚集通常表明趋势启动前平静市场中的良好入场点。

#### 策略原理  
1. 定义连续收盘价上涨/下跌的周期数consclos(默认为3)和当前收盘价与N天前收盘价比较的天数daysago(默认为9)。
2. 用long_cc和short_cc两个变量分别记录最近consclos根K线是否连续收阳/收阴,若连续则值为1,否则为0。
3. 比较当前收盘价与daysago天前的收盘价,若当前价格更高/低则long_daysago/short_daysago为真。  
4. 组合long_cc、short_cc与long_daysago、short_daysago,得到最终的多空信号long和short。
5. 画出long和short信号对应的绿色和红色三角形。
6. 若long信号出现且当前无多头仓位,则开多并设置止损价为信号K线低点。
7. 若short信号出现且当前无空头仓位,则开空并设置止损价为信号K线高点。

#### 优势分析
1. 能够自动识别艾略特波浪理论中的脉冲浪,减少主观分析的影响。
2. 脉冲浪往往伴随着强劲的趋势,这个策略能抓住这样的行情。
3. 止损位的设置与趋势走向一致,提高了盈亏比。
4. 能发现趋势启动前的潜在入场机会。
5. 参数可调,适用性强。

#### 风险分析  
1. 对于波浪理论的解读可能存在偏差,导致误判。
2. 趋势的持续时间难以预计,有可能止损位设得过于靠近导致止损。
3. 对震荡市可能失效,产生频繁交易。
4. 缺乏头寸管理和资金管理方面的考虑。

#### 优化方向
1. 可以通过回测优化consclos和daysago参数的配置,提高信号准确性。
2. 可以引入趋势确认指标如MACD等,减少噪音。
3. 考虑加入移动止损以更好地保护利润。
4. 在趋势尚不明朗时可以先少量建仓,趋势明朗后再加仓。
5. 对仓位和风险进行控制,如限制单次交易的资金比例、设置最大回撤等。

#### 总结
该策略基于经典的艾略特波浪理论,能够捕捉强劲的趋势行情,具有一定的适用性和盈利潜力。但是波浪理论本身的主观性以及脉冲浪的定义等,都可能影响策略表现。在实际应用中,需要注意参数优化、仓位管理、降低交易频率等问题。通过引入趋势确认指标、移动止损、逐步建仓等手段,可以进一步完善该策略的性能和稳定性。

|| 

#### Overview
This strategy is based on the Elliott Wave Theory and attempts to automatically detect impulse waves. It defines an upward impulse wave by looking for a combination of 4 consecutive up-closing candles where the current close is higher than the close 9 days ago; a downward impulse wave is defined using the opposite logic. Once an impulse wave is detected, it generates buy or sell signals and reverses the position, with the stop loss set at the low or high of the signal candle. Since impulse waves are usually accompanied by rapid movements, this stop loss method should yield positive results. Additionally, the accumulation of green or red triangles before the start of a strong trend often indicates good entry points in a calm market before the trend initiation.

#### Strategy Principles
1. Define the number of periods for consecutive up/down closes as consclos (default 3) and the number of days to compare the current close with the close N days ago as daysago (default 9).
2. Use variables long_cc and short_cc to record whether the most recent consclos candles have consecutively closed up/down. The value is 1 if consecutive, otherwise 0.
3. Compare the current close with the close daysago days ago. If the current price is higher/lower, long_daysago/short_daysago is true.
4. Combine long_cc, short_cc with long_daysago, short_daysago to get the final long and short signals.
5. Plot green and red triangles corresponding to the long and short signals.
6. If a long signal appears and there is no current long position, go long and set the stop loss price to the low of the signal candle.
7. If a short signal appears and there is no current short position, go short and set the stop loss price to the high of the signal candle.

#### Advantage Analysis
1. Automatically identifies impulse waves in Elliott Wave Theory, reducing the influence of subjective analysis.
2. Impulse waves are often accompanied by strong trends, which this strategy can capture.
3. The stop loss placement is consistent with the trend direction, improving the risk-reward ratio.
4. Can discover potential entry opportunities before trend initiation.
5. Parameters are adjustable, making it widely applicable.

#### Risk Analysis
1. There may be deviations in the interpretation of wave theory, leading to misjudgments.
2. The duration of trends is difficult to predict, and the stop loss may be set too close, resulting in being stopped out.
3. May be ineffective in sideways markets, generating frequent trades.
4. Lacks consideration of position sizing and money management.

#### Optimization Directions
1. Optimize the configuration of consclos and daysago parameters through backtesting to improve signal accuracy.
2. Introduce trend confirmation indicators such as MACD to reduce noise.
3. Consider adding trailing stops to better protect profits.
4. When the trend is not yet clear, start with a small position and add to it once the trend becomes clear.
5. Control position sizing and risk, such as limiting the percentage of funds per trade and setting a maximum drawdown.

#### Summary 
This strategy is based on the classic Elliott Wave Theory and can capture strong trend movements with some applicability and profit potential. However, the subjectivity of wave theory itself and the definition of impulse waves may affect the strategy's performance. In practical application, attention should be paid to parameter optimization, position management, reducing trading frequency, etc. By introducing trend confirmation indicators, trailing stops, gradual position building, and other means, the performance and stability of this strategy can be further improved.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_int_1|3|Consecutive close|
|v_input_int_2|9|Days ago|


> Source (PineScript)

``` pinescript
/*backtest
start: 2023-04-20 00:00:00
end: 2024-04-25 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © Smollet

//@version=5
strategy("LW: 4-9 indicator", overlay = true)

consclos = input.int(3, "Consecutive close")
daysago = input.int(9, "Days ago")


var int long_cc = 0
var int short_cc = 0

long_cc := 1
short_cc := 1

for i = 1 to consclos
    long_cc := close[i-1] > close[i] ? long_cc*1 : long_cc*0
    short_cc := close[i-1] < close[i] ? short_cc*1 : short_cc*0

long_daysago = close > close[daysago]
short_daysago = close < close[daysago]



long = long_cc ==1 and long_daysago
short = short_cc ==1 and short_daysago


plotshape(long, style=shape.triangleup, location=location.belowbar, color=color.green)
plotshape(short, style=shape.triangledown, location=location.abovebar, color=color.red)



//Strategy code
if long and strategy.position_size <= 0
    strategy.entry("Long", strategy.long)
    strategy.exit("Long SL", "Long", stop = low)

if short and strategy.position_size >= 0
    strategy.entry("Short", strategy.short)
    strategy.exit("Short SL", "Short", stop = high)

```

> Detail

https://www.fmz.com/strategy/449552

> Last Modified

2024-04-26 17:32:59
