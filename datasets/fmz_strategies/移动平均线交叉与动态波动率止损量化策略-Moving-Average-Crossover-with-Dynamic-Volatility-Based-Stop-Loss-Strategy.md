
> Name

移动平均线交叉与动态波动率止损量化策略-Moving-Average-Crossover-with-Dynamic-Volatility-Based-Stop-Loss-Strategy

> Author

ianzeng123

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/2d808067861f1eb72f04c.png)
![IMG](https://www.fmz.com/upload/asset/2d85b2134d351dd0ca484.png)


[trans]

#### 概述

这个量化交易策略是一个结合了移动平均线交叉、相对强弱指标(RSI)过滤和基于平均真实范围(ATR)的动态止损机制的综合性系统。该策略主要用于捕捉中长期趋势，同时通过RSI指标避免在过度超买或超卖的市场环境中入场，并利用ATR指标设置动态止损位，以适应市场波动性的变化。策略在15分钟时间框架上表现尤为出色，能够在捕获日内趋势和避免低时间框架过度噪音之间取得良好平衡。

#### 策略原理

策略的核心逻辑基于以下几个关键组件：

1. **移动平均线交叉信号**：策略使用两条简单移动平均线(SMA)，分别是50周期短期均线和200周期长期均线。当短期均线低于长期均线且RSI值大于30时，系统会触发做多信号。这种设计旨在识别潜在的趋势转变点。

2. **RSI过滤机制**：策略利用14周期的RSI指标进行入场过滤。具体而言，当RSI值高于30时才允许做多，这有助于避免在深度超卖区域盲目入场。代码中虽然保留了做空条件的框架，但当前版本主要聚焦于做多策略。

3. **ATR动态止损**：策略运用14周期的ATR指标来计算动态止损位。止损位设置为入场价格减去(ATR值 × 倍数)，其中ATR倍数默认为1.0。这种动态止损机制能够根据市场的实际波动性自适应调整，在高波动期间提供更宽松的止损空间，在低波动期间保持更紧密的风险控制。

4. **风险回报比例**：策略实现了基于风险回报比(RRR)的止盈设置，默认值为1.5。止盈位计算为入场价格加上((入场价格 - 止损价格) × 风险回报比)，确保每笔交易的潜在收益与承担的风险保持恰当比例。

#### 策略优势

1. **趋势跟踪与过滤相结合**：策略不仅使用移动平均线交叉捕捉趋势变化，还通过RSI指标进行过滤，减少了错误信号，提高了入场质量。

2. **动态风险管理**：基于ATR的止损机制是该策略的一大亮点，它能够根据市场波动性动态调整止损距离，避免了固定止损在高波动环境中过早触发的问题，同时在低波动期保持适当的风险控制。

3. **风险回报比优化**：通过预设的风险回报比，策略确保每笔交易的潜在收益与风险成比例，这有助于长期资金增长，即使在胜率不高的情况下。

4. **交易可视化**：策略包含了对止损位和止盈位的实时绘制，以及对已完成交易的标记功能，这大大提高了策略运行的可视化程度，便于回测分析和策略优化。

5. **资金管理集成**：策略默认使用账户总值的百分比进行头寸管理，这种方法比固定手数更为灵活，能够随着账户规模的变化自动调整交易规模。

#### 策略风险

1. **趋势反转风险**：虽然策略使用移动平均线识别趋势，但在市场突然反转时，可能会导致较大亏损。解决方法是考虑引入更敏感的短期指标作为辅助确认，或调整RSI阈值以提高对反转的敏感度。

2. **参数敏感性**：策略的关键参数如SMA周期、RSI阈值、ATR倍数等，对性能有显著影响。不同市场环境可能需要不同的参数设置，因此需要进行充分的历史回测以找到最优参数组合。

3. **单边市场局限性**：当前版本主要聚焦于做多策略，在持续下跌的市场中可能表现不佳。解决方案是激活代码中的做空条件，实现双向交易能力。

4. **止损过宽风险**：在极端高波动期间，ATR值可能大幅增加，导致止损距离过宽，潜在亏损增大。可以考虑为ATR倍数设置上限，或结合固定金额止损与ATR动态止损的混合方案。

5. **交易频率不确定性**：由于策略依赖中长期移动平均线交叉，可能导致交易信号稀疏，影响资金利用效率。解决方法是考虑增加短期交易信号作为补充，或在主要趋势确立后使用更短期的指标进行加仓。

#### 策略优化方向

1. **多时间框架分析整合**：当前策略仅在单一时间框架上运行，可以考虑整合多时间框架分析，例如使用更高时间框架确认主趋势方向，然后在较低时间框架寻找入场点，提高入场精度。

2. **做空逻辑的完善**：激活并优化策略中的做空逻辑，使其能够在下跌市场中同样有效。这可能需要调整做空的RSI阈值（如RSI大于70时做空），以及针对不同市场方向设置不同的参数。

3. **引入成交量指标**：考虑将成交量指标整合到入场逻辑中，只有在成交量确认的情况下才执行交易信号，这有助于减少假突破带来的亏损。

4. **优化止盈策略**：当前策略使用固定的风险回报比设置止盈，可以考虑实现部分利润锁定或追踪止盈，以在趋势持续发展时获取更多利润。

5. **增加交易时段过滤**：针对具有明显时段特性的市场，可以添加时间过滤器，避免在低流动性或高不确定性时段交易。

6. **参数自适应机制**：实现基于历史波动率或其他市场特性的参数自适应调整机制，使策略能够根据市场环境变化自动优化参数。

#### 总结

这个基于移动平均线交叉、RSI过滤和ATR动态止损的量化策略，提供了一个均衡的交易框架，特别适合中长期趋势交易。其核心优势在于将技术指标分析与动态风险管理无缝结合，既能捕捉趋势变化，又能根据市场波动性调整风险敞口。

虽然策略存在参数敏感性和单向交易的局限，但通过建议的优化方向，如多时间框架分析、完善做空逻辑、引入成交量确认等，这些问题都可以得到有效改善。特别是，将动态参数调整机制与更复杂的止盈策略相结合，有望进一步提升策略的稳健性和盈利能力。

对于寻求中长期趋势交易，同时重视风险控制的交易者来说，这一策略提供了一个坚实的起点，通过个性化调整和持续优化，有潜力成为一个高效的交易系统。

|| 

#### Overview

This quantitative trading strategy is a comprehensive system that combines moving average crossovers, Relative Strength Index (RSI) filtering, and dynamic stop-loss mechanisms based on Average True Range (ATR). The strategy is primarily designed to capture medium to long-term trends while avoiding entry in extremely overbought or oversold market conditions through RSI indicators, and employing ATR-based dynamic stop-loss to adapt to changing market volatility. The strategy performs particularly well on the 15-minute timeframe, achieving a good balance between capturing intraday trends and avoiding excessive noise often seen in lower timeframes.

#### Strategy Principles

The core logic of the strategy is based on several key components:

1. **Moving Average Crossover Signals**: The strategy utilizes two Simple Moving Averages (SMA), a 50-period short-term SMA and a 200-period long-term SMA. A long entry signal is triggered when the short-term SMA is below the long-term SMA and the RSI value is greater than 30. This design aims to identify potential trend reversal points.

2. **RSI Filtering Mechanism**: The strategy employs a 14-period RSI indicator for entry filtering. Specifically, long entries are only permitted when the RSI value is above 30, which helps avoid blind entries in deeply oversold areas. While the framework for short conditions is preserved in the code, the current version primarily focuses on long strategies.

3. **ATR Dynamic Stop-Loss**: The strategy uses a 14-period ATR indicator to calculate dynamic stop-loss levels. The stop-loss is set at the entry price minus (ATR value × multiplier), where the ATR multiplier defaults to 1.0. This dynamic stop-loss mechanism adapts to actual market volatility, providing wider stop-loss space during high volatility periods and maintaining tighter risk control during low volatility periods.

4. **Risk-Reward Ratio**: The strategy implements take-profit settings based on a Risk-Reward Ratio (RRR), with a default value of 1.5. The take-profit level is calculated as the entry price plus ((entry price - stop-loss price) × risk-reward ratio), ensuring that the potential gain for each trade maintains an appropriate proportion to the risk taken.

#### Strategy Advantages

1. **Combination of Trend Following and Filtering**: The strategy not only uses moving average crossovers to capture trend changes but also filters through the RSI indicator, reducing false signals and improving entry quality.

2. **Dynamic Risk Management**: The ATR-based stop-loss mechanism is a highlight of this strategy, as it dynamically adjusts stop-loss distances based on market volatility, avoiding the problem of fixed stops triggering too early in high volatility environments while maintaining appropriate risk control during low volatility periods.

3. **Risk-Reward Ratio Optimization**: Through a preset risk-reward ratio, the strategy ensures that the potential return for each trade is proportional to the risk, which aids long-term capital growth even when the win rate may not be high.

4. **Trade Visualization**: The strategy includes real-time plotting of stop-loss and take-profit levels, as well as marking of completed trades, which greatly enhances the visualization of strategy execution, facilitating backtesting analysis and strategy optimization.

5. **Integrated Capital Management**: By default, the strategy uses a percentage of the account's total value for position sizing, which is more flexible than fixed lot sizing and can automatically adjust trading size as the account grows or shrinks.

#### Strategy Risks

1. **Trend Reversal Risk**: Although the strategy uses moving averages to identify trends, sudden market reversals can lead to significant losses. The solution is to consider introducing more sensitive short-term indicators as auxiliary confirmation or adjusting RSI thresholds to increase sensitivity to reversals.

2. **Parameter Sensitivity**: Key parameters such as SMA periods, RSI thresholds, and ATR multipliers have a significant impact on performance. Different market environments may require different parameter settings, so thorough historical backtesting is needed to find the optimal parameter combinations.

3. **Unidirectional Market Limitations**: The current version primarily focuses on long strategies and may not perform well in continuously declining markets. The solution is to activate the short conditions in the code to implement bidirectional trading capabilities.

4. **Wide Stop-Loss Risk**: During extremely high volatility periods, ATR values may increase significantly, leading to excessively wide stop-loss distances and increased potential losses. Consider setting upper limits for ATR multipliers or implementing a hybrid approach combining fixed monetary stop-losses with ATR dynamic stop-losses.

5. **Trading Frequency Uncertainty**: As the strategy relies on medium to long-term moving average crossovers, it may result in sparse trading signals, affecting capital utilization efficiency. The solution is to consider adding short-term trading signals as supplements or using shorter-term indicators for position additions after the main trend is established.

#### Strategy Optimization Directions

1. **Multiple Timeframe Analysis Integration**: The current strategy operates on a single timeframe. Consider integrating multiple timeframe analysis, such as using higher timeframes to confirm the main trend direction and then looking for entry points on lower timeframes to improve entry precision.

2. **Refinement of Short Logic**: Activate and optimize the short logic in the strategy to make it equally effective in declining markets. This may require adjusting RSI thresholds for shorts (such as shorting when RSI is above 70) and setting different parameters for different market directions.

3. **Introduction of Volume Indicators**: Consider integrating volume indicators into the entry logic, executing trading signals only when confirmed by volume, which helps reduce losses from false breakouts.

4. **Optimization of Take-Profit Strategy**: The current strategy uses a fixed risk-reward ratio for take-profit setting. Consider implementing partial profit locking or trailing stops to capture more profit when trends continue to develop.

5. **Adding Trading Session Filters**: For markets with obvious session characteristics, add time filters to avoid trading during low liquidity or high uncertainty sessions.

6. **Parameter Adaptive Mechanism**: Implement parameter adaptive adjustment mechanisms based on historical volatility or other market characteristics, allowing the strategy to automatically optimize parameters according to changing market environments.

#### Summary

This quantitative strategy based on moving average crossovers, RSI filtering, and ATR dynamic stop-loss provides a balanced trading framework particularly suited for medium to long-term trend trading. Its core advantage lies in the seamless integration of technical indicator analysis with dynamic risk management, capable of both capturing trend changes and adjusting risk exposure based on market volatility.

Although the strategy has limitations in parameter sensitivity and unidirectional trading, these issues can be effectively addressed through the suggested optimization directions, such as multiple timeframe analysis, refined short logic, volume confirmation, etc. In particular, combining dynamic parameter adjustment mechanisms with more sophisticated take-profit strategies promises to further enhance the robustness and profitability of the strategy.

For traders seeking medium to long-term trend trading while emphasizing risk control, this strategy provides a solid starting point that, through personalized adjustments and continuous optimization, has the potential to become an efficient trading system.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-04-02 00:00:00
end: 2025-04-01 00:00:00
period: 1d
basePeriod: 1d
exchanges: [{"eid":"Futures_Binance","currency":"BNB_USDT"}]
*/

//@version=6
strategy(title=" VS-NTC> NASDQ100 Long MA+RSI+ATR", shorttitle="VS-NTC> Long NASDQ100 MA+RSI+ATR", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// ————— Inputs —————
smaLenShort  = input.int(50,  title="Short SMA Length")
smaLenLong   = input.int(200, title="Long SMA Length")
rsiLen       = input.int(14,  title="RSI Length")
atrPeriod    = input.int(14,  title="ATR Period")
atrMult      = input.float(1.0, title="Stop-Loss ATR Multiplier", step=0.1)
rrRatio      = input.float(1.5, title="Risk-to-Reward Ratio",    step=0.1)

// ————— Indicator Calculations —————
smaShort = ta.sma(close, smaLenShort)
smaLong  = ta.sma(close, smaLenLong)
rsiVal   = ta.rsi(close, rsiLen)
atrVal   = ta.atr(atrPeriod)

// ————— Entry Conditions —————
// Long Condition: 50SMA > 200SMA and RSI < 70
longCondition = (smaShort < smaLong) and (rsiVal > 30)
// Short Condition: 50SMA < 200SMA and RSI > 30 (example: avoid oversold)
// Or use RSI > 70 to short if the market is overbought.
shortCondition = false
// shortCondition = (smaShort > smaLong) and (rsiVal < 35)

// ————— Entry Logic —————
if longCondition
    strategy.entry(id="Long", direction=strategy.long)

if shortCondition
    strategy.entry(id="Short", direction=strategy.short)

// ————— Stop-Loss & Take-Profit Calculation —————
var float stopPrice       = na
var float takeProfitPrice = na

// If we have a position open, we determine SL & TP differently for Long or Short.
if strategy.position_size > 0
    // We are in a Long trade
    stopPrice       := strategy.position_avg_price - (atrVal * atrMult)
    takeProfitPrice := strategy.position_avg_price + ((strategy.position_avg_price - stopPrice) * rrRatio)

    strategy.exit("Exit SL/TP", stop=stopPrice, limit=takeProfitPrice)
else if strategy.position_size < 0
    // We are in a Short trade
    stopPrice       := strategy.position_avg_price + (atrVal * atrMult)
    // For short, the distance from entry to stop is (stopPrice - entry)
    // So the take-profit is entry - that same distance times RR
    takeProfitPrice := strategy.position_avg_price - ((stopPrice - strategy.position_avg_price) * rrRatio)

    strategy.exit("Exit SL/TP", stop=stopPrice, limit=takeProfitPrice)
else
    // No open position → reset plots to na
    stopPrice       := na
    takeProfitPrice := na

// ————— Plot the Planned Stop-Loss & Take-Profit —————
plot(stopPrice,       title="Stop Loss",   color=color.red,   linewidth=2)
plot(takeProfitPrice, title="Take Profit", color=color.green, linewidth=2)


// ————— Label Each Closed Trade (Wins & Losses) —————
var int lastClosedTradeCount = 0
currentClosedCount = strategy.closedtrades

// If there's at least one new closed trade, label it
if currentClosedCount > lastClosedTradeCount
    newTradeIndex = currentClosedCount - 1

    tradeProfit  = strategy.closedtrades.profit(newTradeIndex)
    exitBarIndex = strategy.closedtrades.exit_bar_index(newTradeIndex)
    exitPrice    = strategy.closedtrades.exit_price(newTradeIndex)

    // Win label if profit > 0
    if tradeProfit > 0
        labelText  = "Win: " + str.tostring(tradeProfit)
        labelStyle = label.style_label_up
        labelColor = color.new(color.green, 0)
        label.new(exitBarIndex, exitPrice, text=labelText, style=labelStyle, color=labelColor, size=size.tiny)

    // Loss label if profit < 0
    if tradeProfit < 0
        labelText  = "Loss: " + str.tostring(tradeProfit)
        labelStyle = label.style_label_down
        labelColor = color.new(color.red, 0)
        label.new(exitBarIndex, exitPrice, text=labelText, style=labelStyle, color=labelColor, size=size.tiny)

    lastClosedTradeCount := currentClosedCount
```

> Detail

https://www.fmz.com/strategy/489152

> Last Modified

2025-04-02 11:08:39
