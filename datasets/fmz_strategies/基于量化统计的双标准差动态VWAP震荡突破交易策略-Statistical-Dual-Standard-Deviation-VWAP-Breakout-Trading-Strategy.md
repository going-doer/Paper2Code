
> Name

基于量化统计的双标准差动态VWAP震荡突破交易策略-Statistical-Dual-Standard-Deviation-VWAP-Breakout-Trading-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1468f1eabae818b298f.png)

[trans]
#### 概述
该策略是一个基于VWAP(成交量加权平均价格)和标准差通道的趋势突破策略。它通过计算VWAP和上下标准差通道,构建了一个动态的价格波动区间,用于捕捉价格向上突破的交易机会。策略主要依靠标准差带的突破信号进行交易,并设置了利润目标和订单间隔来控制风险。

#### 策略原理
1. 核心指标计算:
- 使用日内HL2价格和成交量计算VWAP
- 基于价格波动计算标准差
- 设置1.28倍标准差上下通道

2. 交易逻辑:
- 入场条件:价格下穿下轨后回升至上方
- 出场条件:达到预设利润目标
- 设置最小下单间隔以避免频繁交易

#### 策略优势
1. 统计学基础
- 基于VWAP的价格中枢参考
- 利用标准差衡量波动性
- 动态调整交易区间

2. 风险控制
- 设定固定利润目标
- 控制交易频率
- 仅做多策略降低风险

#### 策略风险
1. 市场风险
- 剧烈波动可能导致假突破
- 趋势转折点难以准确把握
- 单边下跌行情损失加大

2. 参数风险
- 标准差倍数设置敏感
- 利润目标设置需要优化
- 交易间隔影响收益表现

#### 优化方向
1. 信号优化
- 增加趋势判断过滤器
- 结合成交量变化确认
- 添加其他技术指标验证

2. 风险管理优化
- 动态设置止损位置
- 根据波动率调整仓位
- 完善订单管理机制

#### 总结
这是一个结合了统计学原理和技术分析的量化交易策略。通过VWAP和标准差带的配合,构建了相对可靠的交易系统。策略的核心优势在于其科学的统计学基础和完善的风险控制机制,但仍需要在实际应用中不断优化参数和交易逻辑。
||
#### Overview
This strategy is a trend breakout system based on VWAP (Volume Weighted Average Price) and standard deviation channels. It constructs a dynamic price range by calculating VWAP and standard deviation bands to capture upward breakout opportunities. The strategy mainly relies on standard deviation band breakthrough signals for trading, with profit targets and order intervals to control risk.

#### Strategy Principles
1. Core Indicator Calculation:
- Calculate VWAP using intraday HL2 prices and volume
- Compute standard deviation based on price volatility
- Set 1.28 times standard deviation upper and lower bands

2. Trading Logic:
- Entry condition: price crosses below lower band then rises above it
- Exit condition: reaches preset profit target
- Minimum order interval to avoid frequent trading

#### Strategy Advantages
1. Statistical Foundation
- VWAP-based price center reference
- Volatility measurement using standard deviation
- Dynamic trading range adjustment

2. Risk Control
- Fixed profit target setting
- Trading frequency control
- Long-only strategy reduces risk

#### Strategy Risks
1. Market Risks
- False breakouts during high volatility
- Difficulty in accurately timing trend reversals
- Increased losses in downtrend markets

2. Parameter Risks
- Sensitivity to standard deviation multiplier settings
- Profit target optimization needed
- Trading interval affects performance

#### Optimization Directions
1. Signal Optimization
- Add trend filter
- Incorporate volume change confirmation
- Include additional technical indicators

2. Risk Management Optimization
- Dynamic stop-loss placement
- Position sizing based on volatility
- Enhanced order management system

#### Summary
This is a quantitative trading strategy combining statistical principles and technical analysis. Through the combination of VWAP and standard deviation bands, it builds a relatively reliable trading system. The core advantages lie in its scientific statistical foundation and comprehensive risk control mechanisms, but continuous optimization of parameters and trading logic is still needed in practical applications.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2019-12-23 08:00:00
end: 2025-01-04 08:00:00
period: 1d
basePeriod: 1d
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5 
strategy("VWAP Stdev Bands Strategy (Long Only)", overlay=true)

// Standard Deviation Inputs
devUp1 = input.float(1.28, title="Stdev above (1)")
devDn1 = input.float(1.28, title="Stdev below (1)")

// Show Options
showPrevVWAP = input(false, title="Show previous VWAP close?")
profitTarget = input.float(2, title="Profit Target ($)", minval=0) // Profit target for closing orders
gapMinutes = input.int(15, title="Gap before new order (minutes)", minval=0) // Gap for placing new orders

// VWAP Calculation
var float vwapsum = na
var float volumesum = na
var float v2sum = na
var float prevwap = na // Track the previous VWAP
var float lastEntryPrice = na // Track the last entry price
var int lastEntryTime = na // Track the time of the last entry

start = request.security(syminfo.tickerid, "D", time)
newSession = ta.change(start)

vwapsum := newSession ? hl2 * volume : vwapsum[1] + hl2 * volume
volumesum := newSession ? volume : volumesum[1] + volume
v2sum := newSession ? volume * hl2 * hl2 : v2sum[1] + volume * hl2 * hl2

myvwap = vwapsum / volumesum
dev = math.sqrt(math.max(v2sum / volumesum - myvwap * myvwap, 0))

// Calculate Upper and Lower Bands
lowerBand1 = myvwap - devDn1 * dev
upperBand1 = myvwap + devUp1 * dev

// Plot VWAP and Bands with specified colors
plot(myvwap, style=plot.style_line, title="VWAP", color=color.green, linewidth=1)
plot(upperBand1, style=plot.style_line, title="VWAP Upper (1)", color=color.blue, linewidth=1)
plot(lowerBand1, style=plot.style_line, title="VWAP Lower (1)", color=color.red, linewidth=1)

// Trading Logic (Long Only)
longCondition = close < lowerBand1 and close[1] >= lowerBand1 // Price crosses below the lower band

// Get the current time in minutes
currentTime = timestamp("GMT-0", year(timenow), month(timenow), dayofmonth(timenow), hour(timenow), minute(timenow))

// Check if it's time to place a new order based on gap
canPlaceNewOrder = na(lastEntryTime) or (currentTime - lastEntryTime) >= gapMinutes * 60 * 1000

// Close condition based on profit target
if (strategy.position_size > 0)
    if (close - lastEntryPrice >= profitTarget)
        strategy.close("B")
        lastEntryTime := na // Reset last entry time after closing

// Execute Long Entry
if (longCondition and canPlaceNewOrder)
    strategy.entry("B", strategy.long)
    lastEntryPrice := close // Store the entry price
    lastEntryTime := currentTime // Update the last entry time

    // Add label for the entry
    label.new(bar_index, close, "B", style=label.style_label_down, color=color.green, textcolor=color.white, size=size.small)

// Optional: Plot previous VWAP for reference
prevwap := newSession ? myvwap[1] : prevwap[1]
plot(showPrevVWAP ? prevwap : na, style=plot.style_circles, color=close > prevwap ? color.green : color.red)
```

> Detail

https://www.fmz.com/strategy/477602

> Last Modified

2025-01-06 16:31:36
