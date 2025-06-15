
> Name

突破最高价EMA交叉策略-BreakHigh-EMA-Crossover-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1587d81ec67a06232b3.png)
[trans]
#### 概述

突破最高价EMA交叉策略是一个基于价格突破和指数移动平均线(EMA)交叉的交易策略。该策略使用指定周期内的最高价作为买入信号,EMA作为卖出信号。当收盘价突破指定周期内的最高价时,策略会产生买入信号;当收盘价跌破EMA时,策略会产生卖出信号。该策略还设置了止损价格,以控制风险。此外,该策略还提供了多个参数供用户自定义,以适应不同的交易风格和市场环境。

#### 策略原理

突破最高价EMA交叉策略的核心原理是利用价格突破和EMA交叉来捕捉市场趋势。当价格突破指定周期内的最高价时,表明市场可能进入上升趋势,因此策略会产生买入信号。同时,EMA作为一个趋势跟踪指标,当价格跌破EMA时,表明上升趋势可能结束,因此策略会产生卖出信号。

该策略使用以下步骤来实现交易:

1. 计算指定周期内的最高价作为突破买入价格。
2. 计算EMA作为卖出信号。
3. 当收盘价突破突破买入价格时,如果当前没有持仓,策略会产生买入信号。
4. 当收盘价跌破EMA时,如果当前有持仓,策略会产生卖出信号。
5. 计算指定周期内的最低价作为止损价格。
6. 如果价格跌破止损价格,策略会立即平仓。

通过以上步骤,该策略可以在市场上升趋势中获利,同时使用止损来控制下行风险。

#### 策略优势

突破最高价EMA交叉策略有以下优势:

1. 趋势跟踪:该策略利用价格突破和EMA交叉来捕捉市场趋势,可以在上升趋势中获利。
2. 风险控制:该策略使用止损价格来控制下行风险,可以有效降低策略的最大回撤。
3. 参数灵活:该策略提供了多个参数供用户自定义,如周期、风险比例、是否使用止损等,可以根据不同的交易风格和市场环境进行调整。
4. 简单有效:该策略逻辑简单清晰,易于理解和实现,同时在趋势市场中可以获得不错的收益。

#### 策略风险

尽管突破最高价EMA交叉策略有一定的优势,但它也存在以下风险:

1. 市场波动风险:在市场波动较大的情况下,该策略可能会产生较多的虚假信号,导致频繁交易和资金损失。
2. 趋势转折风险:当市场趋势发生转折时,该策略可能会延迟卖出,导致利润回吐或转亏为亏损。
3. 参数设置风险:该策略的表现依赖于参数的设置,如周期、风险比例等。如果参数设置不当,可能会导致策略表现不佳。

为了缓解这些风险,可以考虑以下措施:

1. 适当调整参数:根据不同的市场环境和交易品种,适当调整策略参数,如增大周期、降低风险比例等,以减少虚假信号和频繁交易。
2. 结合其他指标:可以结合其他技术指标,如RSI、MACD等,以确认趋势和信号的有效性,提高策略的可靠性。
3. 设置合理的止损:设置合理的止损价格,既可以控制下行风险,又不会过早止损,导致错失利润机会。

#### 策略优化方向

为了进一步提高突破最高价EMA交叉策略的性能,可以考虑以下优化方向:

1. 动态调整参数:根据市场波动性和趋势强度,动态调整策略参数,如在波动较大时增大周期,在趋势较强时提高风险比例等,以适应不同的市场环境。
2. 引入多空机制:在原有的多头交易基础上,引入空头交易机制,在下跌趋势中也能获利,提高策略的适应性和收益性。
3. 优化止损和止盈:优化止损和止盈的设置,如使用移动止损、部分止盈等方法,以更好地控制风险和锁定利润。
4. 结合基本面分析:将基本面分析与技术分析相结合,如在企业财报、经济数据发布等重要事件前后,调整策略的仓位和参数,以应对可能的市场变化。

通过以上优化措施,可以提高突破最高价EMA交叉策略的稳定性、适应性和收益性,使其能够在更多的市场环境中获得良好的表现。

#### 总结

突破最高价EMA交叉策略是一个简单有效的趋势跟踪策略,通过利用价格突破和EMA交叉来捕捉市场趋势,同时使用止损来控制下行风险。该策略逻辑清晰,参数灵活,易于理解和实现。尽管该策略存在一定的风险,如市场波动风险、趋势转折风险和参数设置风险,但可以通过适当的风险控制措施来缓解这些风险,如调整参数、结合其他指标和设置合理的止损等。此外,该策略还有进一步优化的空间,如动态调整参数、引入多空机制、优化止损和止盈,以及结合基本面分析等,以提高策略的性能和适应性。总的来说,突破最高价EMA交叉策略是一个值得尝试和优化的量化交易策略。

|| 

#### Overview

The BreakHigh EMA Crossover Strategy is a trading strategy based on price breakout and Exponential Moving Average (EMA) crossover. The strategy uses the highest price within a specified period as the buy signal and the EMA as the sell signal. When the closing price breaks above the highest price within the specified period, the strategy generates a buy signal. When the closing price falls below the EMA, the strategy generates a sell signal. The strategy also sets a stop-loss price to control risk. Additionally, the strategy provides multiple parameters for users to customize to adapt to different trading styles and market conditions.

#### Strategy Principle

The core principle of the BreakHigh EMA Crossover Strategy is to capture market trends using price breakout and EMA crossover. When the price breaks above the highest price within a specified period, it indicates that the market may enter an uptrend, so the strategy generates a buy signal. At the same time, the EMA serves as a trend-following indicator. When the price falls below the EMA, it indicates that the uptrend may end, so the strategy generates a sell signal.

The strategy uses the following steps to implement trading:

1. Calculate the highest price within the specified period as the breakout buy price.
2. Calculate the EMA as the sell signal.
3. When the closing price breaks above the breakout buy price, if there is no current position, the strategy generates a buy signal.
4. When the closing price falls below the EMA, if there is a current position, the strategy generates a sell signal.
5. Calculate the lowest price within the specified period as the stop-loss price.
6. If the price falls below the stop-loss price, the strategy immediately closes the position.

Through the above steps, the strategy can profit from the rising trend in the market while using stop-loss to control downside risk.

#### Strategy Advantages

The BreakHigh EMA Crossover Strategy has the following advantages:

1. Trend tracking: The strategy uses price breakout and EMA crossover to capture market trends and can profit from uptrends.
2. Risk control: The strategy uses a stop-loss price to control downside risk, which can effectively reduce the maximum drawdown of the strategy.
3. Parameter flexibility: The strategy provides multiple parameters for users to customize, such as period, risk ratio, whether to use stop-loss, etc., which can be adjusted according to different trading styles and market conditions.
4. Simple and effective: The strategy logic is simple and clear, easy to understand and implement, and can achieve good returns in trending markets.

#### Strategy Risks

Although the BreakHigh EMA Crossover Strategy has certain advantages, it also has the following risks:

1. Market volatility risk: In cases of high market volatility, the strategy may generate more false signals, leading to frequent trading and capital losses.
2. Trend reversal risk: When the market trend reverses, the strategy may delay selling, resulting in profit retracement or turning profits into losses.
3. Parameter setting risk: The performance of the strategy depends on the setting of parameters, such as period, risk ratio, etc. If the parameters are set improperly, it may lead to poor performance of the strategy.

To mitigate these risks, the following measures can be considered:

1. Properly adjust parameters: According to different market conditions and trading instruments, properly adjust strategy parameters, such as increasing the period, reducing the risk ratio, etc., to reduce false signals and frequent trading.
2. Combine with other indicators: Combine with other technical indicators, such as RSI, MACD, etc., to confirm the validity of trends and signals and improve the reliability of the strategy.
3. Set reasonable stop-loss: Set a reasonable stop-loss price, which can control downside risk and not stop loss too early, resulting in missed profit opportunities.

#### Strategy Optimization Directions

To further improve the performance of the BreakHigh EMA Crossover Strategy, the following optimization directions can be considered:

1. Dynamic parameter adjustment: According to market volatility and trend strength, dynamically adjust strategy parameters, such as increasing the period when volatility is high, increasing the risk ratio when the trend is strong, etc., to adapt to different market conditions.
2. Introduce long-short mechanism: On the basis of the original long trading, introduce a short trading mechanism to profit from downtrends as well, improving the adaptability and profitability of the strategy.
3. Optimize stop-loss and take-profit: Optimize the setting of stop-loss and take-profit, such as using trailing stop-loss, partial take-profit, etc., to better control risks and lock in profits.
4. Combine with fundamental analysis: Combine fundamental analysis with technical analysis, such as adjusting the position and parameters of the strategy before and after important events such as corporate earnings reports and economic data releases, to cope with possible market changes.

Through the above optimization measures, the stability, adaptability, and profitability of the BreakHigh EMA Crossover Strategy can be improved, enabling it to achieve good performance in more market environments.

#### Summary

The BreakHigh EMA Crossover Strategy is a simple and effective trend-following strategy that captures market trends by using price breakout and EMA crossover while using stop-loss to control downside risk. The strategy logic is clear, parameters are flexible, and it is easy to understand and implement. Although the strategy has certain risks, such as market volatility risk, trend reversal risk, and parameter setting risk, these risks can be mitigated through appropriate risk control measures, such as adjusting parameters, combining with other indicators, and setting reasonable stop-loss. In addition, the strategy has further optimization space, such as dynamic parameter adjustment, introducing long-short mechanism, optimizing stop-loss and take-profit, and combining with fundamental analysis, etc., to improve the performance and adaptability of the strategy. Overall, the BreakHigh EMA Crossover Strategy is a quantitative trading strategy worth trying and optimizing.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_int_1|34|Number of previous bars(34,52 Recommend)|
|v_input_1|false|Show BackGround Color|
|v_input_2|true|Show Line|
|v_input_3|true|Show Buy/Sell Signal|
|v_input_4|2.5|% of Risk Per Trade|
|v_input_5|9|Lowest price of the previous number of bars|
|v_input_6|true|Start Strategy|
|v_input_7|false|Use Stoploss Price|
|v_input_8|false|Compound Profit|
|v_input_timeframe_1|D|** Fix chart to which time frame ? **)|
|v_input_9|true|use_date_range|
|v_input_int_2|2012|From Year|
|v_input_int_3|true|From Month|
|v_input_int_4|true|From Day|
|v_input_int_5|9999|To Year|
|v_input_int_6|true|To Month|
|v_input_int_7|true|To Day|


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-02-01 00:00:00
end: 2024-02-29 23:59:59
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// @version = 5
strategy(title="BreakHigh Strategy", overlay=true)
Period = input.int(34, "Number of previous bars(34,52 Recommend)")
showbg = input(defval = false,title = "Show BackGround Color")
showema = input(defval = true ,title = "Show Line")
MarkBuySig = input(defval = true ,title = "Show Buy/Sell Signal")

Risk_Per_Trade = input(2.5, '% of Risk Per Trade') / 100  // Risk% Per Trade Switch
SLDAY = input(title='Lowest price of the previous number of bars', defval=9)
Buysig = input(defval=true, title='Start Strategy')
UseSl = input(defval=false, title='Use Stoploss Price')
Compound = input(defval = false ,title =  "Compound Profit")
xtf = input.timeframe(title='** Fix chart to which time frame ? **)', defval='D')


//BUY
float buyLine = na
buyLine := ta.highest(high,Period)[1] 
plot(showema ? buyLine : na, linewidth=1, style=plot.style_linebr, color=color.new(color.green, 0))

//SELL
output = ta.ema(close, Period)
show = request.security(syminfo.tickerid, xtf, output)
FastL = plot(showema ? show : na, color=color.new(color.white, 0), linewidth=2, title='Slow EMA')

//Buy-Sell Signal
Green = close > buyLine   // Buy
Red = close < show // Sell

buycond = Green and Green[1] == 0
sellcond = Red and Red[1] == 0

bullish = ta.barssince(buycond) < ta.barssince(sellcond)
bearish = ta.barssince(sellcond) < ta.barssince(buycond)

buy = bearish[1] and buycond
sell = bullish[1] and sellcond

plotshape(MarkBuySig ? buy : na, style=shape.labelup, text='Buy Next Bar', textcolor=color.new(color.black, 0), location=location.belowbar, color=color.new(color.green, 0))
plotshape(MarkBuySig ? sell : na, style=shape.labeldown, text='Sell Next Bar', textcolor=color.new(color.black, 0), location=location.abovebar, color=color.new(color.red, 0))
bgcolor(showbg ? bullish ? color.new(color.green,90) : color.new(color.red,90) : na )


// === BACKTEST RANGE === //
use_date_range = input(true)
FromYear = input.int(defval=2012, title='From Year', minval=1950)
FromMonth = input.int(defval=1, title='From Month', minval=1)
FromDay = input.int(defval=1, title='From Day', minval=1)
ToYear = input.int(defval=9999, title='To Year', minval=1950)
ToMonth = input.int(defval=1, title='To Month', minval=1)
ToDay = input.int(defval=1, title='To Day', minval=1)
in_date_range = use_date_range ? time > timestamp(FromYear, FromMonth, FromDay, 00, 00) and time < timestamp(ToYear, ToMonth, ToDay, 23, 59) : true

//****************************************************************************//

//////////////////////////////////////////////
//    define strategy entry / exit          //
//////////////////////////////////////////////

//****************************************************************************//
// LONG CONDITIONS

Select_Long_Condition_1 = close > buyLine // Buy when Have Signal
Open_Long_Condition = Select_Long_Condition_1 and strategy.opentrades == 0

//****************************************************************************//
// STOP LOSS Price

float longSL = na
longSL := Open_Long_Condition ? ta.lowest(low, SLDAY)[1] : longSL[1]  


//****************************************************************************//
// Cal StopLoss

Long_Entry_Price = close
Diff_OPEN_to_SL = math.abs(Long_Entry_Price - longSL)

// Exit CONDITIONS

Exit_Long_Condition = close < show // Sell when Have Signal

//****************************************************************************//
// POSITION SIZE CAP

strategy.initial_capital = 50000

float portSize = Compound ? strategy.netprofit + strategy.initial_capital : strategy.initial_capital
float LossAmoutUnit = portSize * Risk_Per_Trade //50
float PercentSL = ( Diff_OPEN_to_SL / Long_Entry_Price ) * 100
float PositionSize = LossAmoutUnit / Diff_OPEN_to_SL


//****************************************************************************//
// ENTRY/EXIT

if Buysig
    if Open_Long_Condition and in_date_range 
        strategy.entry('LONG', strategy.long, qty=PositionSize)


if Exit_Long_Condition and in_date_range
    strategy.close('LONG')
if close < longSL and UseSl
    strategy.close('LONG')

//****************************************************************************//
// PLOT STOP LOSS

longPlotSL = strategy.opentrades > 0 and strategy.position_size > 0 ? longSL : na
// label.new(bar_index, high, text=str.tostring(longPlotSL),color=color.white, textcolor=color.black)
plot(longPlotSL, title="", linewidth=2, style=plot.style_linebr, color=color.new(color.red, 0))



//****************************************************************************//


```

> Detail

https://www.fmz.com/strategy/446537

> Last Modified

2024-03-29 14:39:27
