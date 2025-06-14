
> Name

TGT基于价格下跌买入策略-TGT-Falling-Buy-Strategy-Based-on-Price-Decline

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1436fb3f86ec5b8f2ac.png)
[trans]
#### 概述
该策略的主要思路是通过监测价格的下跌幅度来进行买入操作。当价格较前一个周期下跌超过5%时,就会触发买入信号,以当前收盘价买入一定数量的仓位。当价格高于买入价格时,就平仓获利了结。该策略利用了市场的波动性,试图抓住价格的短期反弹机会来获利。

#### 策略原理
1. 计算当前收盘价较前一周期收盘价的跌幅百分比。
2. 如果跌幅超过5%,则触发买入信号,以当前收盘价买入一定数量的仓位。买入的数量根据当前账户余额和买入价格计算得出。
3. 记录买入价格和买入数量。
4. 当前价格高于买入价格时,平仓获利了结。
5. 计算盈亏情况,更新账户余额。
6. 在图表上用黄色标记出买入信号发生时的K线。

#### 优势分析
1. 简单易懂:策略逻辑清晰,容易理解和实现。
2. 趋势捕捉:通过买入下跌幅度较大的品种,可以捕捉到价格的短期反弹趋势。
3. 风险控制:买入数量是根据账户余额和当前价格计算得出,控制了每次交易的风险敞口。
4. 及时了结:当价格高于买入价时就果断平仓,不恋战,控制住了风险。
5. 直观表现:在图表上用特殊颜色标记出买入信号,方便观察和分析。

#### 风险分析
1. 频繁交易:该策略以短期波动为主要目标,交易频率可能较高,需要注意手续费成本对收益的影响。
2. 深度回撤:如果在买入后价格出现进一步的大幅下跌,则可能面临一定的回撤风险。
3. 价格波动:策略主要依赖价格的波动性,在波动率较低的市场环境中,策略的效果可能会打折扣。
4. 盈亏平衡:策略对胜率和赔率并没有明确的要求和控制,在实际运行中需要注意策略的整体盈亏平衡能力。

#### 优化方向
1. 止损优化:目前策略在买入后并没有设置止损条件,可以考虑增加一些止损逻辑,比如固定百分比止损或者ATR止损等,来进一步控制单次交易的最大损失。
2. 信号过滤:在产生买入信号后,可以增加一些额外的条件来过滤信号的质量,比如结合均线系统、RSI等指标,或者考虑价格拐点、烛线形态等,以提高信号的胜率和可靠度。
3. 仓位管理:目前策略采用固定资金比例来确定买入数量,可以考虑优化为更动态的仓位管理模型,比如根据价格波动率、账户净值曲线等因素来调整每次买入的数量。
4. 多品种协同:该策略的思路可以用于多个品种,通过品种之间的关联性分析以及资金分配管理,可能会取得更好的效果。

#### 总结
该策略以价格短期下跌超过特定幅度作为买入信号,抓住价格的反弹机会来获利,逻辑简单易懂。策略的优势在于对趋势的捕捉和风险的控制,但是频繁交易、深度回撤、价格波动等风险也需要注意。未来可以从止损优化、信号过滤、仓位管理、多品种协同等方面对策略进行进一步的优化和改进,以期获得更稳健的效果。

|| 

#### Overview
The main idea of this strategy is to perform a buy operation by monitoring the decline in price. When the price falls by more than 5% compared to the previous period, a buy signal is triggered, and a certain amount of position is bought at the current closing price. When the price is higher than the buying price, the position is closed to take profits. This strategy takes advantage of market volatility and tries to capture short-term price rebound opportunities to make profits.

#### Strategy Principle
1. Calculate the percentage decline of the current closing price compared to the previous period's closing price.
2. If the decline exceeds 5%, a buy signal is triggered, and a certain amount of position is bought at the current closing price. The quantity purchased is calculated based on the current account balance and the buying price.
3. Record the buying price and the quantity purchased.
4. When the current price is higher than the buying price, close the position to take profits.
5. Calculate the profit and loss situation and update the account balance.
6. Mark the candlestick with a yellow color on the chart when the buy signal occurs.

#### Advantage Analysis
1. Simple and easy to understand: The strategy logic is clear and easy to understand and implement.
2. Trend capture: By buying varieties with a larger decline, it can capture the short-term rebound trend of the price.
3. Risk control: The purchase quantity is calculated based on the account balance and the current price, controlling the risk exposure of each trade.
4. Timely closing: When the price is higher than the buying price, the position is decisively closed, not holding on, controlling the risk.
5. Visual representation: The buy signal is marked with a special color on the chart, which is convenient for observation and analysis.

#### Risk Analysis
1. Frequent trading: This strategy mainly targets short-term fluctuations, and the trading frequency may be relatively high. Attention needs to be paid to the impact of transaction costs on returns.
2. Deep drawdown: If the price experiences a further significant decline after buying, it may face a certain drawdown risk.
3. Price volatility: The strategy mainly relies on price volatility, and in a market environment with low volatility, the effect of the strategy may be discounted.
4. Profit and loss balance: The strategy does not have clear requirements and controls on the win rate and loss rate, and attention needs to be paid to the overall profit and loss balancing ability of the strategy in actual operation.

#### Optimization Direction
1. Stop-loss optimization: Currently, the strategy does not set a stop-loss condition after buying. Consideration can be given to adding some stop-loss logic, such as fixed percentage stop-loss or ATR stop-loss, to further control the maximum loss of a single transaction.
2. Signal filtering: After generating a buy signal, some additional conditions can be added to filter the quality of the signal, such as combining moving average systems, RSI and other indicators, or considering price turning points, candlestick patterns, etc., to improve the win rate and reliability of the signal.
3. Position management: Currently, the strategy uses a fixed capital ratio to determine the purchase quantity. Consideration can be given to optimizing it into a more dynamic position management model, such as adjusting the purchase quantity according to factors such as price volatility and account equity curve.
4. Multi-variety collaboration: The idea of this strategy can be applied to multiple varieties. Through the correlation analysis between varieties and fund allocation management, better results may be achieved.

#### Summary
This strategy uses the short-term price decline exceeding a specific amplitude as a buy signal, capturing the rebound opportunity of the price to make profits. The logic is simple and easy to understand. The advantages of the strategy lie in trend capture and risk control, but risks such as frequent trading, deep drawdown, and price volatility also need to be noted. In the future, the strategy can be further optimized and improved from aspects such as stop-loss optimization, signal filtering, position management, and multi-variety collaboration, in order to obtain more robust results.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-06-01 00:00:00
end: 2024-06-06 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © Thgoodtrader

//@version=5
strategy("TGT Falling Buy", overlay=true, margin_long=100, margin_short=100)
var float buy_price = na
var float open_price = na
var float open_weekend = na 
var float close_weekend = na 
var bool trade=false
var float balance = 1000
// Definir el precio de compra inicial y la cantidad inicial
var float qty = na
// Verificar si el día de la semana es sábado (6) o domingo (0)
es_sabado = dayofweek == 1
es_domingo = dayofweek == 7
es_viernes = dayofweek == 6

// Calcular el valor del saldo inicial
balance_initial = balance

change_percent = ((close - close[1]) / close[1]) * 100
is_last_candle_negative = close < open
is_change_above_threshold = change_percent < -5
// Cambiar el color de la última vela si cumple las condiciones
barcolor(is_last_candle_negative and is_change_above_threshold ? color.yellow : na)
bgcolor(is_last_candle_negative and is_change_above_threshold ? color.yellow : na, transp=80)
// Guardar el precio de compra cuando se cumpla la condición del 5%
if is_change_above_threshold 
    // Calcular la cantidad basada en el precio de compra y el saldo
    qty := balance / close
    // Guardar el precio de compra
    buy_price := close
    open_price := open
    strategy.entry("Buy Trading",strategy.long,qty)
    alert("Comprar BTC", alert.freq_once_per_bar_close)
    trade :=true
//if (((close - strategy.position_avg_price) / strategy.position_avg_price) * 100 ) > 2
if close > strategy.position_avg_price
    // Calcular el valor de ganancia o pérdida
    pnl = (close - strategy.position_avg_price) * qty
    // Actualizar el saldo
    balance := balance_initial + pnl
    strategy.close("Buy Trading")
alertcondition(is_change_above_threshold, title = "Buy 5% Discount", message = "Buy Position")
alertcondition(close > strategy.position_avg_price, title = "Close Trade", message = "Close Buy Position")   
```

> Detail

https://www.fmz.com/strategy/453656

> Last Modified

2024-06-07 15:33:26
