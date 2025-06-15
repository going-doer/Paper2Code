
> Name

日内锤子反转形态多头策略Intraday-Hammer-Reversal-Pattern-Long-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/e08122ee4d83f8aa37.png)
[trans]
## 概述
该策略利用日内锤子反转形态和后续绿色蜡烛结合,寻找潜在的上涨机会。当出现锤子反转形态,且下一根蜡烛为绿色上涨蜡烛时,策略开仓做多。止损位置设置在锤子蜡烛的低点,止盈位置设置为开仓价格的1.5倍。

## 策略原理
锤子形态是一种常见的技术形态,常出现在下跌趋势尾声,预示着趋势反转的到来。典型的锤子形态具有以下特征:
1. 整体蜡烛实体较小,通常低于整个蜡烛高低范围的30%。
2. 下影线较长,至少是蜡烛实体长度的2倍。
3. 上影线很短或者没有,最多不超过蜡烛开盘价的1%。  

当锤子形态确认后,如果下一根蜡烛为绿色上涨蜡烛,且低点高于锤子蜡烛的低点,则形成看涨信号,此时入场做多。止损设置在锤子蜡烛低点,以控制风险;止盈设置为开仓价的1.5倍,以获取潜在利润。

## 优势分析
1. 锤子形态是常见的反转形态,配合趋势背景使用胜率较高。
2. 严格限制锤子形态和后续看涨蜡烛的形态,提高信号质量。
3. 止损位置设置在锤子蜡烛低点,风险可控。
4. 止盈位置设置为1.5R,具备不错的盈亏比。

## 风险分析
1. 即使形态和后续走势满足策略条件,行情仍有反复甚至继续下跌的风险。 
2. 锤子蜡烛低点止损位置较近,一旦触发止损,单次损失相对较大。
3. 趋势转折初期波动较大,策略面临较高的价格波动风险。

## 优化方向  
1. 可以考虑引入更多技术指标,如RSI、MACD等,结合指标状态来提高信号有效性。
2. 对锤子形态和后续看涨蜡烛的形态定义可以进一步优化,如引入更多量化标准。
3. 止盈止损位置设置可以进一步优化,如采用动态止盈或者移动止损策略。
4. 考虑市场趋势状态,在上涨趋势中寻找锤子形态可能胜率更高。

## 总结
日内锤子反转形态多头策略充分利用了锤子形态反转的特点,结合后续绿色蜡烛的确认,在两个连续K线形态基础上形成看涨信号。同时,策略采用固定的止盈止损比例,控制了风险暴露水平,也让盈亏比维持在较高水平。但是,该策略对于形态的定义相对简单,缺乏其他技术指标的验证,在实际应用中可能面临较高的信号失效率。此外,由于止损位置设置相对较近,策略也面临单次损失较高的问题。未来可以从信号确认、风险控制等方面对策略进行进一步的优化和改进,以提升整体稳定性和盈利能力。

|| 

## Overview
This strategy uses the intraday hammer reversal pattern in combination with a subsequent green candle to find potential upside opportunities. When a hammer reversal pattern appears and the next candle is a green upward candle, the strategy opens a long position. The stop loss is set at the low of the hammer candle, and the take profit is set at 1.5 times the entry price.

## Strategy Principles 
The hammer pattern is a common technical pattern that often appears at the end of a downtrend, signaling the arrival of a trend reversal. A typical hammer pattern has the following characteristics:
1. The overall candle body is relatively small, usually less than 30% of the entire candle's high-low range.
2. The lower shadow is long, at least twice the length of the candle body.
3. The upper shadow is very short or nonexistent, at most not exceeding 1% of the candle's opening price.

When the hammer pattern is confirmed, if the next candle is a green upward candle and the low is higher than the low of the hammer candle, a bullish signal is formed and a long position is entered. The stop loss is set at the low of the hammer candle to control risk, and the take profit is set at 1.5 times the entry price to capture potential profits.

## Advantage Analysis
1. The hammer pattern is a common reversal pattern and has a high win rate when used in combination with trend context.
2. Strict restrictions on the hammer pattern and subsequent bullish candle shape improve signal quality.
3. Setting the stop loss at the low of the hammer candle makes risk controllable.
4. Setting the take profit at 1.5R provides a decent risk-reward ratio.

## Risk Analysis  
1. Even if the pattern and subsequent price action satisfy the strategy conditions, there is still a risk of the market repeating or continuing to fall.
2. With the stop loss set close to the low of the hammer candle, a single loss is relatively large once triggered.  
3. Volatility is high in the early stages of a trend reversal, exposing the strategy to high price volatility risk.

## Optimization Directions
1. Consider introducing more technical indicators, such as RSI and MACD, to improve signal validity in combination with indicator status.
2. The definitions of the hammer pattern and subsequent bullish candle can be further optimized, such as introducing more quantitative criteria.
3. Take profit and stop loss settings can be further optimized, such as using dynamic take profit or trailing stop strategies.  
4. Consider market trend conditions, as hammer patterns found in uptrends may have higher win rates.

## Summary
The intraday hammer reversal pattern long strategy makes full use of the reversal characteristics of the hammer pattern, combined with confirmation from a subsequent green candle, to form a bullish signal based on two consecutive candle patterns. At the same time, the strategy uses a fixed risk-reward ratio to control risk exposure and maintain a high risk-reward ratio. However, the strategy's definition of patterns is relatively simple and lacks verification from other technical indicators, which may face a high signal failure rate in practical applications. In addition, because the stop loss is set relatively close, the strategy also faces the problem of high single losses. In the future, the strategy can be further optimized and improved in terms of signal confirmation and risk control to enhance overall stability and profitability.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-03-09 00:00:00
end: 2024-03-14 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Hammer Pattern and Follow-Up Green Candle Strategy", overlay=true)

// Detecting a Hammer candle
isHammer() =>
    bodySize = math.abs(close[1] - open[1])
    lowerWickSize = open[1] - low[1]
    upperWickSize = high[1] - open[1] // For a red candle, the upper wick is from the open to the high
    bodyIsSmall = bodySize <= (high[1] - low[1]) * 0.3 // Body is less than 30% of the entire candle range
    lowerWickIsLong = lowerWickSize >= bodySize * 2 // Lower wick is at least twice the body length
    noUpperWick = upperWickSize == 0 or high[1] <= open[1] * 1.01 // No upper wick or very small
    close[1] < open[1] and bodyIsSmall and lowerWickIsLong and noUpperWick

// Check if the current candle is green with no or small tail
isGreenWithNoSmallTail() =>
    close > open

// Entry condition
entryCondition = isHammer() and isGreenWithNoSmallTail() and low >low[1]

// Calculate stop loss and take profit levels
stopLossLevel = low[1]
profitTargetLevel = close * 1.5
//Calculate position bodySize
positionSize = 50000 / close

// Execute strategy
if (entryCondition)
    strategy.entry("Hammer Buy", strategy.long,qty=positionSize)
    strategy.exit("Take Profit / Stop Loss", "Hammer Buy", stop=stopLossLevel, limit=profitTargetLevel)


```

> Detail

https://www.fmz.com/strategy/444985

> Last Modified

2024-03-15 17:13:23
