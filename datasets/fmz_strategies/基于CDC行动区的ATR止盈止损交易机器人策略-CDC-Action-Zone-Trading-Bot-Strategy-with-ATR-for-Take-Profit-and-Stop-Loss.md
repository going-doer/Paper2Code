
> Name

基于CDC行动区的ATR止盈止损交易机器人策略-CDC-Action-Zone-Trading-Bot-Strategy-with-ATR-for-Take-Profit-and-Stop-Loss

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/ca0a88661898c92098.png)

[trans]
#### 概述
该策略是一个基于CDC行动区的交易机器人策略。它使用12周期和26周期的指数移动平均线(EMA)来确定市场趋势,当短期EMA在长期EMA上方时做多,反之做空。该策略使用平均真实波幅(ATR)来设置动态止盈和止损水平。止盈水平基于ATR和一个倍数确定,止损水平固定为当前收盘价的5%。

#### 策略原理
1. 计算12周期和26周期的EMA,用于确定市场趋势。
2. 计算ATR,用于设置动态止盈和止损水平。
3. 当短期EMA在长期EMA上方时,发出买入信号并开仓做多。
4. 当短期EMA在长期EMA下方时,发出卖出信号并开仓做空。 
5. 止盈水平基于ATR和一个倍数确定,当价格达到止盈水平时平仓。
6. 止损水平固定为当前收盘价的5%,当价格达到止损水平时平仓。

#### 策略优势
1. 使用EMA来捕捉市场趋势,能有效地适应不同的市场环境。
2. 采用ATR来设置动态止盈水平,可以更好地保护利润。
3. 固定的止损水平有助于控制风险,将损失限制在可接受的范围内。
4. 代码结构清晰,易于理解和修改,适合进一步优化。

#### 策略风险
1. EMA是一个滞后指标,在市场快速变化时可能会发出错误信号。
2. ATR止盈水平可能无法在市场波动较大时及时保护利润。
3. 固定的止损水平可能在某些情况下导致过早平仓,错失潜在利润。
4. 该策略没有考虑交易成本和滑点,实际交易结果可能与回测结果存在差异。

#### 策略优化方向 
1. 尝试使用其他趋势指标,如MACD或移动平均线交叉,来提高信号的准确性。
2. 优化ATR倍数和止盈止损百分比,以更好地适应不同的市场条件。
3. 引入动态止损机制,如跟踪止损或基于波动率的止损,以更好地控制风险。
4. 考虑交易成本和滑点,选择合适的交易品种和交易时段,以提高策略的实际表现。

#### 总结
该策略是一个基于CDC行动区的ATR止盈止损交易机器人策略,通过EMA来捕捉市场趋势,ATR来设置动态止盈水平,并采用固定百分比止损来控制风险。虽然该策略具有一定的优势,但仍存在一些风险和改进空间。通过进一步优化和测试,该策略有望在实际交易中取得良好表现。

|| 

#### Overview
This strategy is a trading bot strategy based on the CDC action zone. It uses the 12-period and 26-period Exponential Moving Averages (EMA) to determine market trends, going long when the short-term EMA is above the long-term EMA and going short when the opposite is true. The strategy employs Average True Range (ATR) to set dynamic take profit and stop loss levels. The take profit level is determined based on ATR and a multiplier, while the stop loss level is fixed at 5% of the current closing price.

#### Strategy Principles
1. Calculate the 12-period and 26-period EMAs to determine market trends.
2. Calculate ATR to set dynamic take profit and stop loss levels.
3. When the short-term EMA is above the long-term EMA, a buy signal is generated, and a long position is opened.
4. When the short-term EMA is below the long-term EMA, a sell signal is generated, and a short position is opened.
5. The take profit level is determined based on ATR and a multiplier, and the position is closed when the price reaches the take profit level.
6. The stop loss level is fixed at 5% of the current closing price, and the position is closed when the price reaches the stop loss level.

#### Strategy Advantages
1. Using EMAs to capture market trends can effectively adapt to different market conditions.
2. Employing ATR to set dynamic take profit levels can better protect profits.
3. Fixed stop loss levels help control risk and limit losses to an acceptable range.
4. The code structure is clear and easy to understand and modify, making it suitable for further optimization.

#### Strategy Risks
1. EMAs are lagging indicators and may generate false signals when the market changes rapidly.
2. ATR-based take profit levels may not protect profits in time during high market volatility.
3. Fixed stop loss levels may lead to premature position closures in some cases, missing out on potential profits.
4. The strategy does not consider trading costs and slippage, so actual trading results may differ from backtesting results.

#### Strategy Optimization Directions
1. Experiment with other trend indicators, such as MACD or moving average crossovers, to improve signal accuracy.
2. Optimize the ATR multiplier and take profit/stop loss percentages to better adapt to different market conditions.
3. Introduce dynamic stop loss mechanisms, such as trailing stops or volatility-based stops, to better control risk.
4. Consider trading costs and slippage, and choose appropriate trading instruments and trading sessions to improve the strategy's actual performance.

#### Summary
This strategy is an ATR-based take profit and stop loss trading bot strategy based on the CDC action zone. It uses EMAs to capture market trends, ATR to set dynamic take profit levels, and fixed percentage stop losses to control risk. Although the strategy has certain advantages, it still has some risks and room for improvement. With further optimization and testing, the strategy may achieve good performance in actual trading.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-05-01 00:00:00
end: 2024-05-31 23:59:59
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("CDC Action Zone Trading Bot with ATR for Take Profit and 5% Stop Loss", overlay=true)

// ดึงข้อมูลราคาปิด
close_price = close

// คำนวณเส้น EMA 12 และ EMA 26
ema12 = ta.ema(close_price, 12)
ema26 = ta.ema(close_price, 26)

// คำนวณ ATR
atr_length = input.int(14, title="ATR Length")
atr = ta.atr(atr_length)

// กำหนด Multiplier สำหรับ ATR Trailing Stoploss
mult_atr_stoploss = input.float(2.5, title="ATR Stoploss Multiplier")

// คำนวณ ATR Trailing Stoploss
prev_stoploss = close_price
for i = 1 to 10
    prev_stoploss := math.max(prev_stoploss, high[i] - mult_atr_stoploss * atr)

// กำหนด Take Profit เป็น ATR Trailing Stoploss
takeProfitPercent = input.float(10, title="Take Profit (%)") / 100
takeProfit = close_price + (close_price - prev_stoploss) * takeProfitPercent

// กำหนด Stop Loss เป็น 5% ของราคาปิดปัจจุบัน
stopLossPercent = input.float(5, title="Stop Loss (%)") / 100
stopLoss = close_price * stopLossPercent

// กำหนดสีแท่งกราฟ
buyColor = input.color(color.green, title="Buy Color")
sellColor = input.color(color.red, title="Sell Color")
neutralColor = input.color(color.gray, title="Neutral Color")
color = if (ema12 > ema26)
    buyColor
else if (ema12 < ema26)
    sellColor
else
    neutralColor

// สัญญาณ Buy
buySignal = (color == buyColor) and (color[1] != buyColor)

// สัญญาณ Sell
sellSignal = (color == sellColor) and (color[1] != sellColor)

// เปิด Position Long
if (buySignal)
    strategy.entry("Long", strategy.long)

// เปิด Position Short
if (sellSignal)
    strategy.entry("Short", strategy.short)

// ปิด Position เมื่อถึง Take profit
if (strategy.position_size > 0 and close_price > takeProfit)
    strategy.exit("Long", profit=takeProfit)

// ปิด Position เมื่อถึง Stop loss
if (strategy.position_size > 0 and close_price < stopLoss)
    strategy.exit("Long", loss=stopLoss)

// ปิด Position เมื่อถึง Take profit
if (strategy.position_size < 0 and close_price < takeProfit)
    strategy.exit("Short", profit=takeProfit)

// ปิด Position เมื่อถึง Stop loss
if (strategy.position_size < 0 and close_price > stopLoss)
    strategy.exit("Short", loss=stopLoss)

```

> Detail

https://www.fmz.com/strategy/453266

> Last Modified

2024-06-03 16:19:32
