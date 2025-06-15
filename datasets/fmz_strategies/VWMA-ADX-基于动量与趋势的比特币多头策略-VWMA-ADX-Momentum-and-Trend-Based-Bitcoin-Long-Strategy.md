
> Name

VWMA-ADX-基于动量与趋势的比特币多头策略-VWMA-ADX-Momentum-and-Trend-Based-Bitcoin-Long-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/104c7b79ea2771e1315.png)
[trans]
#### 概述
该策略利用了多个移动平均线(VWMA)、平均方向性指数(ADX)以及动向指标(DMI)来捕捉比特币市场中的多头机会。通过结合价格动量、趋势方向和交易量等多个技术指标,该策略旨在找到上涨趋势强劲、动量充足的入场点,同时严格控制风险。

#### 策略原理
1. 使用9日和14日VWMA来判断多头趋势,当短期均线上穿长期均线时产生多头信号。
2. 引入一条由89日最高价和最低价VWMA构建的自适应均线作为趋势过滤,只有当收盘价或开盘价高于该均线时才考虑开仓。
3. 通过ADX和DMI指标来确认趋势强度,只有当ADX大于18且+DI与-DI的差值大于15时,才认为趋势足够强劲。
4. 利用交易量百分位函数过滤出交易量处于60%~95%区间的棒线,避开交易量过低的时段。
5. 设置止损位于前一根K线高点的0.96~0.99倍,并且随着时间框架的增大而递减,以控制风险。
6. 达到预设的持仓时间或价格跌破自适应均线时平仓。

#### 优势分析
1. 结合多个技术指标,从趋势、动量和交易量等多个维度来评估市场状态,信号更加可靠。
2. 自适应均线和交易量过滤机制能够有效过滤掉虚假信号,减少无效交易。
3. 严格的止损设置和持仓时间限制,大大降低了策略的风险敞口。
4. 代码模块化设计,可读性和可维护性较好,便于进一步优化和扩展。

#### 风险分析
1. 当市场处于震荡或趋势不明朗时,该策略可能会产生较多的虚假信号。
2. 止损位置相对较近,在行情波动较大时可能会过早触发止损,导致损失扩大。
3. 缺乏对宏观经济形势和重大事件的考量,面对"黑天鹅"事件可能会失效。
4. 参数设置相对固定,缺乏自适应性,在不同的市场环境下表现可能不稳定。

#### 优化方向
1. 引入更多能够刻画市场环境的指标,如相对强弱指数(RSI)、布林带等,提高信号的可靠性。
2. 对止损位置进行动态优化,例如采用ATR或百分比止损,以应对不同的市场波动状况。
3. 结合宏观经济数据和舆情分析,对策略的风险控制模块进行增强。
4. 采用机器学习算法对参数进行自动优化,提高策略的适应性和稳定性。

#### 总结
VWMA-ADX 比特币多头策略通过综合考虑价格趋势、动量、交易量等多个技术指标,能够较为有效地捕捉比特币市场中的上涨机会。同时,严格的风控措施和清晰的平仓条件,使得该策略的风险得到了较好的控制。但是,该策略也存在一些局限性,如对市场环境变化的适应性不足,以及止损策略有待优化等。未来可以从信号的可靠性、风险控制、参数优化等方面入手,进一步提升策略的稳健性和盈利能力。总的来说,VWMA-ADX 比特币多头策略为投资者提供了一种基于动量和趋势的系统化交易思路,值得进一步探索和改进。

|| 

#### Overview
This strategy utilizes multiple moving averages (VWMA), the Average Directional Index (ADX), and the Directional Movement Indicator (DMI) to capture long opportunities in the Bitcoin market. By combining price momentum, trend direction, and trading volume, the strategy aims to find entry points with strong upward trends and sufficient momentum while strictly controlling risk.

#### Strategy Principles
1. Use the 9-day and 14-day VWMA to determine the long trend. A bullish signal is generated when the short-term moving average crosses above the long-term moving average.
2. Introduce an adaptive moving average constructed from the 89-day highest and lowest price VWMA as a trend filter. Only consider opening a position when the closing price or opening price is above this moving average.
3. Confirm trend strength using the ADX and DMI indicators. The trend is considered strong enough only when the ADX is greater than 18 and the difference between +DI and -DI is greater than 15.
4. Filter out bars with trading volume between the 60% and 95% percentiles using the volume percentile function to avoid periods with low trading volume.
5. Set the stop-loss level at 0.96 to 0.99 times the previous candle's high, decreasing as the time frame increases to control risk.
6. Close the position when the predefined holding time is reached or the price falls below the adaptive moving average.

#### Advantage Analysis
1. By combining multiple technical indicators, the strategy evaluates market conditions from various dimensions such as trend, momentum, and trading volume, making the signals more reliable.
2. The adaptive moving average and trading volume filtering mechanism effectively filter out false signals and reduce invalid trades.
3. Strict stop-loss settings and holding time limits greatly reduce the strategy's risk exposure.
4. The modular design of the code enhances readability and maintainability, facilitating further optimization and expansion.

#### Risk Analysis
1. When the market is fluctuating or the trend is unclear, the strategy may generate more false signals.
2. The stop-loss level is relatively tight, which may trigger premature stop-outs and lead to increased losses during large market fluctuations.
3. The strategy lacks consideration of macroeconomic conditions and significant events, and may fail in the face of "black swan" events.
4. The parameter settings are relatively fixed and lack adaptability, which may result in unstable performance in different market environments.

#### Optimization Directions
1. Introduce more indicators that can capture market conditions, such as the Relative Strength Index (RSI) and Bollinger Bands, to improve signal reliability.
2. Dynamically optimize the stop-loss level, for example, by using Average True Range (ATR) or percentage-based stop-loss, to adapt to different market volatility conditions.
3. Enhance the strategy's risk control module by incorporating macroeconomic data and sentiment analysis.
4. Utilize machine learning algorithms to automatically optimize parameters, improving the strategy's adaptability and stability.

#### Summary
The VWMA-ADX Bitcoin Long Strategy effectively captures upward opportunities in the Bitcoin market by comprehensively considering price trends, momentum, trading volume, and other technical indicators. At the same time, strict risk control measures and clear exit conditions ensure that the strategy's risk is well-controlled. However, the strategy also has some limitations, such as insufficient adaptability to changing market environments and the need for optimized stop-loss strategies. In the future, improvements can be made in terms of signal reliability, risk control, and parameter optimization to further enhance the strategy's robustness and profitability. Overall, the VWMA-ADX Bitcoin Long Strategy provides investors with a systematic trading approach based on momentum and trend, which is worth further exploration and refinement.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-03-01 00:00:00
end: 2024-03-31 23:59:59
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © Q_D_Nam_N_96

//@version=5
strategy("Long BTC Strategy", overlay=true, 
     default_qty_type = strategy.percent_of_equity, 
     default_qty_value = 100, initial_capital = 1000, currency = currency.USD)

Volume_Quartile(vol) =>
    qvol1 = ta.percentile_linear_interpolation(vol, 60,15)
    qvol2 = ta.percentile_linear_interpolation(vol, 60,95)
    vol > qvol1 and vol < qvol2

smma(src, length) =>
	smma =  0.0
	smma := na(smma[1]) ? ta.sma(src, length) : (smma[1] * (length - 1) + src) / length
	smma

ma(source, length, type) =>
    switch type
        "SMA" => ta.sma(source, length)
        "EMA" => ta.ema(source, length)
        "RMA" => ta.rma(source, length)
        "WMA" => ta.wma(source, length)
        "VWMA" => ta.vwma(source, length)
        "HMA" => ta.hma(source, length)
        "SMMA" => smma(source, length)

DMI(len, lensig) =>
    up = ta.change(high)
    down = -ta.change(low)
    plusDM = na(up) ? na : (up > down and up > 0 ? up : 0)
    minusDM = na(down) ? na : (down > up and down > 0 ? down : 0)
    trur = ta.rma(ta.tr, len)
    plus = fixnan(100 * ta.rma(plusDM, len) / trur)+11
    minus = fixnan(100 * ta.rma(minusDM, len) / trur)-11
    sum = plus + minus
    adx = 100 * ta.vwma(math.abs(plus - minus-11) / (sum == 0 ? 1 : sum), lensig)

    [adx, plus, minus]

cond1 = Volume_Quartile(volume*hlcc4)

ma1 = ma(close,9, "VWMA")
// plot(ma1, color = color.blue)
ma2 = ma(close,14, "VWMA")
// plot(ma2, color = color.orange)

n = switch timeframe.period
    "240" => 0.997
    => 0.995

ma3 = (0.1*ma(ta.highest(close,89),89, "VWMA") + 
     0.9*ma(ta.lowest(close,89),89, "VWMA"))*n

plot(ma3, color = color.white)

[adx, plus, minus] = DMI(7, 10)


cond2 = adx > 18 and plus - math.abs(minus) > 15

var int count = 0

if barstate.isconfirmed and strategy.position_size != 0
    count += 1
else
    count := 0

p_roc = 0
if timeframe.period == '240'
    p_roc := 14
else
    p_roc := 10

longCondition = ta.crossover(ma1, ma2) and (close > open ? close > ma3 : open > ma3) and ((ma3 - ma3[1])*100/ma3[1] >= -0.2) and ((close-close[p_roc])*100/close[p_roc] > -2.0)
float alpha = 0.0
float sl_src = high[1]
if (longCondition and cond1 and cond2 and strategy.position_size == 0)
    strategy.entry("buy", strategy.long)
    if timeframe.period == '240'
        alpha := 0.96
        strategy.exit("exit-buy","buy", stop = sl_src*alpha)
        // line.new(bar_index, sl_src*alpha, bar_index+5, sl_src*alpha, width = 2, color = color.white)
    else if timeframe.period == '30'
        alpha := 0.985
        strategy.exit("exit-buy","buy", stop = sl_src*alpha)
        // line.new(bar_index, sl_src*alpha, bar_index+20, sl_src*alpha, width = 2, color = color.white)
    else if timeframe.period == '45'
        alpha := 0.985
        strategy.exit("exit-buy","buy", stop = sl_src*alpha)
        // line.new(bar_index, sl_src*alpha, bar_index+20, sl_src*alpha, width = 2, color = color.white)
    else if timeframe.period == '60'
        alpha := 0.98
        strategy.exit("exit-buy","buy", stop = sl_src*alpha)
        // line.new(bar_index, sl_src*alpha, bar_index+20, sl_src*alpha, width = 2, color = color.white)
    else if timeframe.period == '120'
        alpha := 0.97
        strategy.exit("exit-buy","buy", stop = sl_src*alpha)
        // line.new(bar_index, sl_src*alpha, bar_index+20, sl_src*alpha, width = 2, color = color.white)
    else if timeframe.period == '180'
        alpha := 0.96
        strategy.exit("exit-buy","buy", stop = sl_src*alpha)
        // line.new(bar_index, sl_src*alpha, bar_index+20, sl_src*alpha, width = 2, color = color.white)
    else if timeframe.period == 'D'
        alpha := 0.95
        strategy.exit("exit-buy","buy", stop = sl_src*alpha)
        // line.new(bar_index, sl_src*alpha, bar_index+20, sl_src*alpha, width = 2, color = color.white)
    else 
        alpha := 0.93
        strategy.exit("exit-buy","buy", stop = sl_src*alpha)
        // line.new(bar_index, sl_src*alpha, bar_index+20, sl_src*alpha, width = 2, color = color.white)

period = switch timeframe.period
    "240" => 90
    "180" => 59
    "120" => 35
    "30" => 64
    "45" => 40
    "60" => 66
    "D" => 22
    => 64

if (count > period or close < ma3)
    strategy.close('buy', immediately = true) 
```

> Detail

https://www.fmz.com/strategy/446985

> Last Modified

2024-04-03 17:47:49
