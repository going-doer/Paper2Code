
> Name

动态达瓦斯箱体突破与均线趋势确认交易系统-Dynamic-Darvas-Box-Breakout-with-Moving-Average-Trend-Confirmation-Trading-System

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1191b5ea6ced73fead6.png)

[trans]
#### 概述
本文介绍的是一个结合达瓦斯箱体(Darvas Box)和25周期移动平均线(MA25)的趋势跟踪交易系统。该策略通过识别价格盘整区间形成的箱体,并结合均线趋势确认,在突破时捕捉强势行情。系统设计充分考虑了趋势延续性和假突破过滤,为交易者提供了一个完整的市场进出场框架。

#### 策略原理
策略主要包含三个核心组成部分:
1. 达瓦斯箱体的构建:系统通过计算过去5个周期的最高价和最低价来确定箱体边界。箱体顶部由新高点确定,底部由相应区间内的最低点确定。
2. 均线趋势确认:引入25周期简单移动平均线作为趋势过滤器,只有当价格位于MA25之上时才考虑开仓。
3. 交易信号生成:
   - 买入信号:价格突破箱体顶部且位于MA25之上
   - 卖出信号:价格跌破箱体底部

#### 策略优势
1. 趋势跟踪能力强:
   - 通过箱体突破捕捉趋势起始
   - 结合MA25过滤,确保在主趋势方向交易
2. 信号质量优化:
   - 双重确认机制降低假突破风险
   - 明确的进出场条件,避免主观判断
3. 风险控制完善:
   - 箱体底部自然形成止损位
   - MA25提供额外的趋势保护

#### 策略风险
1. 震荡市场风险:
   - 频繁突破可能导致连续止损
   - 建议在强趋势市场使用
2. 滞后性风险:
   - 箱体形成需要时间,可能错过部分行情
   - MA25作为中期均线存在一定滞后
3. 资金管理风险:
   - 需要合理设置每次交易的资金比例
   - 建议结合波动率动态调整仓位

#### 策略优化方向
1. 参数优化:
   - 可根据不同市场特征调整箱体周期
   - MA周期可以根据市场周期特征调整
2. 信号增强:
   - 可添加成交量确认机制
   - 考虑引入动态止损机制
3. 风险控制增强:
   - 添加波动率过滤器
   - 实现动态仓位管理

#### 总结
该策略通过结合经典的达瓦斯箱体理论和移动平均线趋势跟踪,构建了一个稳健的交易系统。系统的主要优势在于能够有效捕捉趋势性行情,同时通过多重过滤机制控制风险。虽然存在一定的滞后性,但通过合理的参数优化和风险管理,该策略能够在趋势市场中获得稳定表现。建议交易者在实盘使用时,重点关注市场环境的选择,并根据实际情况动态调整参数设置。

|| 

#### Overview
This article introduces a trend following trading system that combines Darvas Box and 25-period Moving Average (MA25). The strategy identifies price consolidation zones through box formation and confirms trends with moving averages to capture strong market movements during breakouts. The system design thoroughly considers trend continuation and false breakout filtering, providing traders with a complete framework for market entry and exit.

#### Strategy Principles
The strategy consists of three core components:
1. Darvas Box Construction: The system determines box boundaries by calculating the highest and lowest prices over 5 periods. The box top is determined by new highs, while the bottom is set by the lowest point within the corresponding range.
2. Moving Average Trend Confirmation: A 25-period simple moving average is introduced as a trend filter, only considering positions when price is above MA25.
3. Trade Signal Generation:
   - Buy Signal: Price breaks above box top and is above MA25
   - Sell Signal: Price breaks below box bottom

#### Strategy Advantages
1. Strong Trend Following Capability:
   - Captures trend initiation through box breakouts
   - MA25 filtering ensures trading in primary trend direction
2. Signal Quality Optimization:
   - Dual confirmation mechanism reduces false breakout risk
   - Clear entry and exit conditions avoid subjective judgment
3. Comprehensive Risk Control:
   - Box bottom naturally forms stop-loss level
   - MA25 provides additional trend protection

#### Strategy Risks
1. Choppy Market Risk:
   - Frequent breakouts may lead to consecutive stops
   - Recommended for use in strong trend markets
2. Lag Risk:
   - Box formation requires time, may miss initial moves
   - MA25 as medium-term average has inherent lag
3. Money Management Risk:
   - Requires proper allocation of capital per trade
   - Suggested to dynamically adjust position size with volatility

#### Strategy Optimization Directions
1. Parameter Optimization:
   - Box period adjustable based on market characteristics
   - MA period can be adjusted to market cycle features
2. Signal Enhancement:
   - Can add volume confirmation mechanism
   - Consider implementing dynamic stop-loss
3. Risk Control Enhancement:
   - Add volatility filter
   - Implement dynamic position sizing

#### Summary
The strategy builds a robust trading system by combining classic Darvas Box theory with moving average trend following. Its main advantage lies in effectively capturing trending markets while controlling risk through multiple filtering mechanisms. Although there is some inherent lag, the strategy can achieve stable performance in trending markets through proper parameter optimization and risk management. Traders are advised to focus on market environment selection and dynamically adjust parameters based on actual conditions when implementing the strategy.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-10-01 00:00:00
end: 2024-10-31 23:59:59
period: 1h
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("DARVAS BOX with MA25 Buy Condition", overlay=true, shorttitle="AEG DARVAS")

// Input for box length
boxp = input.int(5, "BOX LENGTH")

// Calculate 25-period moving average
ma25 = ta.sma(close, 25)

// Lowest low and highest high within the box period
LL = ta.lowest(low, boxp)
k1 = ta.highest(high, boxp)
k2 = ta.highest(high, boxp - 1)
k3 = ta.highest(high, boxp - 2)

// New high detection
NH = ta.valuewhen(high > k1[1], high, 0)

// Logic to detect top and bottom of Darvas Box
box1 = k3 < k2
TopBox = ta.valuewhen(ta.barssince(high > k1[1]) == boxp - 2 and box1, NH, 0)
BottomBox = ta.valuewhen(ta.barssince(high > k1[1]) == boxp - 2 and box1, LL, 0)

// Plot the top and bottom Darvas Box lines
plot(TopBox, linewidth=3, color=color.green, title="Top Box")
plot(BottomBox, linewidth=3, color=color.red, title="Bottom Box")
plot(ma25, color=#2195f31e, linewidth=2, title="ma25")

// --- Buy and Sell conditions ---

// Buy when price breaks above the Darvas Box AND MA15
buyCondition = ta.crossover(close, TopBox) and close > ma25

// Sell when price drops below the Darvas Box
sellCondition = ta.crossunder(close, BottomBox)

// --- Buy and Sell Signals ---

// Plot BUY+ and SELL labels
plotshape(series=buyCondition, title="Buy+ Signal", location=location.abovebar, color=#72d174d3, style=shape.labeldown, text="BUY")
plotshape(series=sellCondition, title="Sell Signal", location=location.belowbar, color=color.rgb(234, 62, 62, 28), style=shape.labelup, text="SELL")

// --- Strategy execution ---

if (buyCondition)
    strategy.entry("Buy", strategy.long)

if (sellCondition)
    strategy.close("Buy")

```

> Detail

https://www.fmz.com/strategy/472252

> Last Modified

2024-11-18 16:02:45
