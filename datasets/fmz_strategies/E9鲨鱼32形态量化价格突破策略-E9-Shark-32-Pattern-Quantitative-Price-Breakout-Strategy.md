
> Name

E9鲨鱼32形态量化价格突破策略-E9-Shark-32-Pattern-Quantitative-Price-Breakout-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/157f21fc7b785ed813d.png)

[trans]
#### 概述
该策略是一个基于价格形态识别的量化交易系统,核心是识别并利用"鲨鱼32"这种特殊的K线形态进行交易。策略通过对高点和低点的连续性变化进行分析,在形态确认后设定关键价格水平,并在突破这些水平时进行交易。该策略结合了形态识别、趋势跟踪和价格突破等多个技术分析要素,建立了一个完整的交易系统。

#### 策略原理
策略的核心在于识别"鲨鱼32"形态,这种形态需要满足以下条件:前两根K线的低点连续走低,同时高点连续走高。当形态确认后,策略会锁定形态起始K线的高点和低点作为关键价格水平。系统在价格突破这些关键水平时开仓:当收盘价突破锁定的高点时做多,突破锁定的低点时做空。策略采用了投影目标线作为获利目标,止损位置则通过百分比参数来灵活设定。

#### 策略优势
1. 形态识别准确:通过严格的数学定义来识别形态,避免了主观判断
2. 风险管理完善:包含了清晰的止损和获利目标设置
3. 视觉反馈清晰:使用不同颜色的线条和背景来标示形态和交易信号
4. 过滤重复信号:每个形态只允许执行一次交易,避免过度交易
5. 目标设置合理:基于形态波动幅度来设定获利目标,具有良好的风险收益比

#### 策略风险
1. 震荡市场风险:在横盘震荡市场可能产生频繁的假突破信号
2. 滑点风险:在快速行情中可能面临较大的滑点
3. 单一形态依赖:过度依赖单一形态可能错过其他交易机会
4. 参数敏感性:止损和获利目标的参数设置对策略表现影响较大

#### 策略优化方向
1. 加入成交量确认:可以结合成交量变化来确认突破的有效性
2. 引入市场环境过滤:增加趋势强度指标来过滤不利的市场环境
3. 优化止损方式:可以考虑使用动态止损来提高策略的适应性
4. 增加时间过滤:加入交易时间段过滤来避免特定时段的波动
5. 完善资金管理:增加仓位管理模块来优化资金利用效率

#### 总结
E9鲨鱼32形态量化价格突破策略是一个结构完整、逻辑清晰的交易系统。它通过严格的形态定义和清晰的交易规则,构建了一个可量化执行的交易策略。策略的风险管理体系完善,视觉反馈清晰,便于交易者理解和执行。通过建议的优化方向,策略还有进一步提升的空间。该策略适合追求系统化交易的投资者,但使用时需要注意市场环境的适应性和参数的优化。

||

#### Overview
This strategy is a quantitative trading system based on pattern recognition, focusing on identifying and trading the "Shark-32" candlestick pattern. The strategy analyzes continuous changes in highs and lows, sets key price levels after pattern confirmation, and executes trades on breakouts of these levels. It combines pattern recognition, trend following, and price breakout elements to create a complete trading system.

#### Strategy Principles
The core principle lies in identifying the "Shark-32" pattern, which requires consecutive lower lows and higher highs in the previous two candles. Upon pattern confirmation, the strategy locks the high and low of the initial pattern candle as key price levels. The system enters positions when price breaks these key levels: long entries on breaks above the locked high, and short entries on breaks below the locked low. The strategy uses projected target lines for profit objectives and percentage-based parameters for flexible stop-loss placement.

#### Strategy Advantages
1. Accurate pattern recognition: Uses strict mathematical definitions to identify patterns, avoiding subjective judgment
2. Comprehensive risk management: Includes clear stop-loss and profit target settings
3. Clear visual feedback: Uses different colored lines and backgrounds to mark patterns and trading signals
4. Filtered repeat signals: Allows only one trade per pattern, preventing overtrading
5. Rational target setting: Sets profit targets based on pattern amplitude, providing good risk-reward ratios

#### Strategy Risks
1. Choppy market risk: May generate frequent false breakout signals in ranging markets
2. Slippage risk: May face significant slippage in fast-moving markets
3. Single pattern dependency: Over-reliance on one pattern may miss other trading opportunities
4. Parameter sensitivity: Strategy performance heavily depends on stop-loss and profit target parameter settings

#### Strategy Optimization Directions
1. Add volume confirmation: Incorporate volume changes to confirm breakout validity
2. Implement market environment filters: Add trend strength indicators to filter unfavorable market conditions
3. Optimize stop-loss methods: Consider dynamic stop-loss to improve strategy adaptability
4. Add time filters: Incorporate trading session filters to avoid specific volatile periods
5. Enhance money management: Add position sizing module to optimize capital efficiency

#### Summary
The E9 Shark-32 Pattern Quantitative Price Breakout Strategy is a well-structured trading system with clear logic. It builds a quantifiable trading strategy through strict pattern definitions and clear trading rules. The strategy features a comprehensive risk management system and clear visual feedback, making it easy for traders to understand and execute. Through the suggested optimization directions, there's room for further improvement. This strategy is suitable for investors seeking systematic trading approaches, but attention must be paid to market environment adaptability and parameter optimization.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-10-01 00:00:00
end: 2024-10-31 23:59:59
period: 2h
basePeriod: 2h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//║                                                                                                             ║
//║ ░▒▓████████▓▒░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░▒▓███████▓▒░   ░▒▓████████▓▒░▒▓██████▓▒░  ║
//║    ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒   ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░ ║
//║    ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒.  ░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░ ║
//║    ░▒▓█▓▒░   ░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░ ░▒▓███████▓▒░.  ░▒▓██████▓▒░ ░▒▓███████▓▒░ ║
//║    ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒.  ░▒▓█▓▒░            ░▒▓█▓▒░ ║
//║    ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒.  ░▒▓█▓▒░            ░▒▓█▓▒░ ║
//║    ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒.  ░▒▓████████▓▒░▒▓██████▓▒░  ║
//║                                                                                                             ║
//╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

//@version=5
strategy("E9 Shark-32 Pattern Strategy with Target Lines", shorttitle="E9 Shark-32 Strategy", overlay=true)

// Inputs for background color settings
bgcolorEnabled = input(true, title="Enable Background Color")
bgcolorColor = input.color(color.new(color.blue, 90), title="Background Color")

// Inputs for bar color settings
barcolorEnabled = input(true, title="Enable Bar Color")
barcolorColor = input.color(color.rgb(240, 241, 154), title="Bar Color")

// Inputs for target lines settings
targetLinesEnabled = input(true, title="Enable Target Lines")
targetLineColor = input.color(color.white, title="Target Line Color")
targetLineThickness = input.int(1, title="Target Line Thickness", minval=1, maxval=5)

// Define Shark-32 Pattern
shark32 = low[2] < low[1] and low[1] < low and high[2] > high[1] and high[1] > high

// Initialize color variables for bars
var color barColorCurrent = na
var color barColor1 = na
var color barColor2 = na

// Update color variables based on Shark-32 pattern
barColorCurrent := barcolorEnabled and (shark32 or shark32[1] or shark32[2]) ? barcolorColor : na
barColor1 := barcolorEnabled and (shark32[1] or shark32[2]) ? barcolorColor : na
barColor2 := barcolorEnabled and shark32[2] ? barcolorColor : na

// Apply the bar colors to the chart
barcolor(barColorCurrent, offset=-2, title="Shark-32 Confirmed Current")
barcolor(barColor1, offset=-3, title="Shark-32 Confirmed Previous Bar 1")
barcolor(barColor2, offset=-4, title="Shark-32 Confirmed Previous Bar 2")

// Variables for locking the high and low of confirmed Shark-32
var float patternHigh = na
var float patternLow = na
var float upperTarget = na
var float lowerTarget = na

// Once Shark-32 pattern is confirmed, lock the patternHigh, patternLow, and target lines
if shark32
    patternHigh := high[2]  // The high of the first bar in Shark-32 pattern
    patternLow := low[2]    // The low of the first bar in Shark-32 pattern

    // Calculate the upper and lower white target lines
    upperTarget := patternHigh + (patternHigh - patternLow)  // Dotted white line above
    lowerTarget := patternLow - (patternHigh - patternLow)   // Dotted white line below

// Initialize variables for the lines
var line greenLine = na
var line redLine = na
var line upperTargetLine = na
var line lowerTargetLine = na

// Draw the lines based on the locked patternHigh, patternLow, and target lines
// if shark32
//     future_bar_index_lines = bar_index + 10

//     // Draw lines based on locked patternHigh and patternLow
//     greenLine := line.new(x1=bar_index[2], y1=patternHigh, x2=future_bar_index_lines, y2=patternHigh, color=color.green, width=2, extend=extend.none)
//     redLine := line.new(x1=bar_index[2], y1=patternLow, x2=future_bar_index_lines, y2=patternLow, color=color.red, width=2, extend=extend.none)

//     // Draw dotted white lines if targetLinesEnabled is true
//     if targetLinesEnabled
//         upperTargetLine := line.new(x1=bar_index[2], y1=upperTarget, x2=future_bar_index_lines, y2=upperTarget, color=targetLineColor, width=targetLineThickness, style=line.style_dotted, extend=extend.none)
//         lowerTargetLine := line.new(x1=bar_index[2], y1=lowerTarget, x2=future_bar_index_lines, y2=lowerTarget, color=targetLineColor, width=targetLineThickness, style=line.style_dotted, extend=extend.none)

//     // Create a box to fill the background between the red and green lines
//     if bgcolorEnabled
//         box.new(left=bar_index[2], top=patternHigh, right=future_bar_index_lines, bottom=patternLow, bgcolor=bgcolorColor)

// -------------------------------------------------------------------------
// Strategy Entry and Exit Parameters
// -------------------------------------------------------------------------

// Input parameters for stop loss
longStopLoss = input.float(1.0, title="Long Stop Loss (%)", minval=0.1)  // Percentage-based stop loss for long
shortStopLoss = input.float(1.0, title="Short Stop Loss (%)", minval=0.1)  // Percentage-based stop loss for short

// Variable to track if a trade has been taken
var bool tradeTaken = false

// Reset the flag when a new Shark-32 pattern is confirmed
if shark32
    tradeTaken := false

// Entry conditions only trigger after the Shark-32 is confirmed
longCondition = ta.crossover(close, patternHigh) and not tradeTaken  // Long entry when close crosses above locked patternHigh
shortCondition = ta.crossunder(close, patternLow) and not tradeTaken  // Short entry when close crosses below locked patternLow

// Trigger long and short trades based on the crossover conditions
if (longCondition)
    label.new(bar_index, high, "Long Trigger", style=label.style_label_down, color=color.green, textcolor=color.white, size=size.small)
    strategy.entry("Shark-32 Long", strategy.long)
    tradeTaken := true  // Set the flag to true after a trade is taken

if (shortCondition)
    label.new(bar_index, low, "Short Trigger", style=label.style_label_up, color=color.red, textcolor=color.white, size=size.small)
    strategy.entry("Shark-32 Short", strategy.short)
    tradeTaken := true  // Set the flag to true after a trade is taken

// Exit long trade based on the upper target line (upper white dotted line) as take profit
if strategy.position_size > 0
    strategy.exit("Take Profit Long", "Shark-32 Long", limit=upperTarget, stop=close * (1 - longStopLoss / 100))

// Exit short trade based on the lower target line (lower white dotted line) as take profit
if strategy.position_size < 0
    strategy.exit("Take Profit Short", "Shark-32 Short", limit=lowerTarget, stop=close * (1 + shortStopLoss / 100))

```

> Detail

https://www.fmz.com/strategy/471691

> Last Modified

2024-11-12 14:51:17
