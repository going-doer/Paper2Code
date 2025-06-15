
> Name

基于123点位反转的动态持仓期策略-Dynamic-Holding-Period-Strategy-Based-on-123-Point-Reversal-Pattern

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/994efb5209428c5e23.png)

[trans]
#### 概述
本策略是一个基于市场价格形态识别的量化交易系统,主要通过识别123点位反转形态来捕捉市场潜在的反转机会。策略结合了动态持仓期管理和移动平均线过滤,通过多重条件验证来提高交易的准确性。该策略采用精确的数学模型来定义入场点,并使用200日均线作为辅助退出条件,形成了一个完整的交易系统。

#### 策略原理
策略的核心逻辑基于价格形态识别,具体包含以下关键要素:
1. 入场条件设计
- 当日最低价需低于前一日最低价
- 前一日最低价需低于3天前最低价
- 2天前最低价需低于4天前最低价  
- 2天前最高价需低于3天前最高价
以上四个条件同时满足时,系统会发出做多信号。

2. 退出机制设计
- 设定默认持仓期为7天
- 使用200日简单移动平均线(SMA)作为动态退出条件
- 当价格触及或超过200日均线时触发平仓信号
- 持仓时间达到设定天数后自动平仓

#### 策略优势
1. 形态识别准确性高
- 采用多重条件验证机制
- 通过价格高低点的相对位置关系严格定义入场条件
- 降低了误判概率

2. 风险控制完善
- 设定固定持仓期限制最大损失
- 使用长期均线作为趋势过滤器
- 具备双重退出机制保护盈利

3. 操作规则明确
- 入场退出条件清晰明确
- 参数可根据市场情况灵活调整
- 便于实盘执行和回测验证

#### 策略风险
1. 形态识别局限性
- 在震荡市场可能产生虚假信号
- 剧烈波动时段准确率下降
- 需要配合其他技术指标验证

2. 参数优化风险
- 固定持仓期可能不适合所有市场环境
- 移动平均线周期选择影响策略表现
- 过度优化可能导致过拟合

3. 市场适应性风险
- 在强趋势市场中反转信号可靠性降低
- 不同市场条件下表现差异较大
- 需要定期评估策略有效性

#### 策略优化方向
1. 入场信号优化
- 增加成交量确认机制
- 引入动量指标作为辅助判断
- 考虑加入波动率过滤器

2. 退出机制完善
- 实现动态持仓期管理
- 增加移动止损功能
- 开发多层次利润目标

3. 风险控制增强
- 建立仓位管理系统
- 设计回撤控制机制
- 添加市场情绪指标

#### 总结
该策略通过严格的形态识别和完善的风险控制体系,为交易者提供了一个可靠的市场反转捕捉工具。虽然存在一定的局限性,但通过持续优化和适当的参数调整,该策略能够在不同市场环境下保持稳定的表现。建议交易者在实际应用中结合市场经验,对策略进行针对性调整,以获得更好的交易效果。

|| 

#### Overview
This strategy is a quantitative trading system based on market price pattern recognition, primarily designed to capture potential market reversal opportunities by identifying 123-point reversal patterns. The strategy combines dynamic holding period management with moving average filtering, utilizing multiple condition verification to enhance trading accuracy. It employs precise mathematical models for entry point definition and uses a 200-day moving average as an auxiliary exit condition, forming a complete trading system.

#### Strategy Principles
The core logic is based on price pattern recognition, including the following key elements:
1. Entry Condition Design
- Current day's low must be lower than previous day's low
- Previous day's low must be lower than the low from three days ago
- The low from two days ago must be lower than the low from four days ago
- The high from two days ago must be lower than the high from three days ago
When all four conditions are met simultaneously, the system generates a long signal.

2. Exit Mechanism Design
- Default holding period set to 7 days
- Uses 200-day Simple Moving Average (SMA) as dynamic exit condition
- Triggers position closure when price touches or exceeds the 200-day moving average
- Automatic position closure when holding period reaches the set duration

#### Strategy Advantages
1. High Pattern Recognition Accuracy
- Multiple condition verification mechanism
- Strict entry conditions based on relative price high/low positions
- Reduced false signal probability

2. Comprehensive Risk Control
- Fixed holding period limits maximum loss
- Long-term moving average as trend filter
- Dual exit mechanism to protect profits

3. Clear Operating Rules
- Explicit entry and exit conditions
- Flexible parameters adjustable to market conditions
- Easy to implement and backtest

#### Strategy Risks
1. Pattern Recognition Limitations
- May generate false signals in choppy markets
- Reduced accuracy during periods of extreme volatility
- Requires validation with other technical indicators

2. Parameter Optimization Risks
- Fixed holding period may not suit all market environments
- Moving average period selection affects strategy performance
- Over-optimization may lead to overfitting

3. Market Adaptability Risks
- Reduced reliability of reversal signals in strong trend markets
- Performance varies across different market conditions
- Requires periodic strategy effectiveness evaluation

#### Strategy Optimization Directions
1. Entry Signal Enhancement
- Add volume confirmation mechanism
- Incorporate momentum indicators as auxiliary judgment
- Consider adding volatility filters

2. Exit Mechanism Improvement
- Implement dynamic holding period management
- Add trailing stop loss functionality
- Develop multi-level profit targets

3. Risk Control Enhancement
- Establish position management system
- Design drawdown control mechanism
- Add market sentiment indicators

#### Summary
The strategy provides traders with a reliable market reversal capture tool through strict pattern recognition and comprehensive risk control systems. While certain limitations exist, continuous optimization and appropriate parameter adjustments enable the strategy to maintain stable performance across different market environments. Traders are advised to combine market experience with strategy-specific adjustments in practical applications to achieve better trading results.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2019-12-23 08:00:00
end: 2024-11-11 00:00:00
period: 1d
basePeriod: 1d
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © EdgeTools

//@version=5
strategy("123 Reversal Trading Strategy", overlay=true)

// Input for number of days to hold the trade
daysToHold = input(7, title="Days to Hold Trade")

// Input for 20-day moving average
maLength = input(200, title="Moving Average Length")

// Calculate the 20-day moving average
ma20 = ta.sma(close, maLength)

// Define the conditions for the 123 reversal pattern (bullish reversal)
// Condition 1: Today's low is lower than yesterday's low
condition1 = low < low[1]

// Condition 2: Yesterday's low is lower than the low three days ago
condition2 = low[1] < low[3]

// Condition 3: The low two days ago is lower than the low four days ago
condition3 = low[2] < low[4]

// Condition 4: The high two days ago is lower than the high three days ago
condition4 = high[2] < high[3]

// Entry condition: All conditions must be true
entryCondition = condition1 and condition2 and condition3 and condition4

// Exit condition: Close the position after a certain number of bars or when the price reaches the 20-day moving average
exitCondition = ta.barssince(entryCondition) >= daysToHold or close >= ma20

// Execute buy and sell signals
if (entryCondition)
    strategy.entry("Buy", strategy.long)
if (exitCondition)
    strategy.close("Buy")


```

> Detail

https://www.fmz.com/strategy/471698

> Last Modified

2024-11-12 15:15:46
