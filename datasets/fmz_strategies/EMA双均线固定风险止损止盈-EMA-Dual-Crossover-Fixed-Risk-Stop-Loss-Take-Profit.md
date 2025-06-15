
> Name

EMA双均线固定风险止损止盈-EMA-Dual-Crossover-Fixed-Risk-Stop-Loss-Take-Profit

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1aae70800c516f5a39b.png)

[trans]
#### 概述
该策略采用双EMA均线交叉作为交易信号,快线周期为65,慢线周期为240。同时使用成交量作为过滤条件,只有在当前成交量大于指定阈值时才会进行交易。策略对每笔交易设置固定风险金额($10),并根据风险金额动态计算仓位大小。当快线上穿慢线且成交量满足条件时做多,快线下穿慢线且成交量满足条件时做空。止损位和止盈位根据固定价格距离设置,做多时止损位在开仓价下方$100,止盈位在开仓价上方$1500;做空时止损位在开仓价上方$100,止盈位在开仓价下方$1500。

#### 策略原理
1. 计算两条EMA均线,快线(ema_fast)周期为65,慢线(ema_slow)周期为240。
2. 判断是否发生多头交叉(bullish_crossover)或空头交叉(bearish_crossover)。
3. 设置成交量阈值(volume_threshold),只有当前成交量大于该阈值时才进行交易。
4. 设置每笔交易的固定风险金额(risk_per_trade)为$10。
5. 根据风险金额和止损距离(stop_loss_distance)计算仓位大小(position_size)。
6. 当多头交叉发生且成交量满足条件时,开多仓,止损位设置在开仓价下方$100,止盈位设置在开仓价上方$1500。
7. 当空头交叉发生且成交量满足条件时,开空仓,止损位设置在开仓价上方$100,止盈位设置在开仓价下方$1500。

#### 策略优势
1. 双均线交叉能够较好地捕捉到市场趋势,65/240周期组合可以过滤掉大部分噪音,只关注主要趋势。
2. 引入成交量过滤条件,可以避免在成交量较低时交易,降低市场波动风险。
3. 固定风险金额的仓位管理方式,能够有效控制每笔交易的风险敞口,避免单笔交易损失过大。
4. 基于价格距离的动态止损和止盈设置,可以让盈利空间大于亏损空间,从而提高策略长期表现。
5. 适用于BTC/USD等高波动品种,能够充分捕捉其波动带来的投资机会。

#### 策略风险
1. EMA作为趋势跟踪指标,存在趋势反转时滞后的问题,可能导致延迟入场或延迟出场。
2. 固定风险金额可能无法动态适应市场波动情况,在极端行情下(如暴涨暴跌)表现欠佳。
3. 成交量阈值的设置具有一定主观性,阈值设置不当可能影响策略效果。
4. 止损位和止盈位的固定设置可能与市场实际波动幅度不匹配,导致频繁止损或止盈离场。
5. 策略在震荡行情中表现可能欠佳,频繁交叉可能导致连续亏损交易。

#### 策略优化方向 
1. 考虑引入更多均线组合作为过滤条件,如加入中期均线,构建多均线系统来提高信号可靠性。
2. 优化仓位管理方式,如采用百分比风险法或凯利公式等动态调整仓位,以适应不同市场状态。
3. 对成交量阈值进行参数优化,找到最佳阈值设置以提高策略稳定性。
4. 优化止损止盈位置设置,根据最新市场波动情况实时调整,增加与市场相适应的灵活性。
5. 在趋势型方法中加入一定的对冲成分,如PSAR等反趋势指标辅助判断,增强震荡市应对能力。

#### 总结
该策略采用65/240双均线交叉作为趋势判断依据,同时结合成交量过滤条件来改进信号可靠性。固定风险仓位管理和固定价格止损止盈设置,可以一定程度上控制风险并让盈亏比倾向于有利方向。但策略也存在趋势把握相对滞后、仓位管理灵活性不足、止损止盈缺乏动态调整等问题。未来可以从构建多均线系统、优化仓位管理、动态止损止盈、引入对冲指标等角度对策略进行优化和改进,以期获得更加稳定和可靠的交易表现。

|| 

#### Overview
This strategy employs a dual EMA crossover approach as trading signals, with the fast EMA having a period of 65 and the slow EMA having a period of 240. It also uses volume as a filter condition, only executing trades when the current volume exceeds a specified threshold. The strategy sets a fixed risk amount ($10) for each trade and dynamically calculates position sizes based on the risk amount. When the fast EMA crosses above the slow EMA and the volume condition is met, it enters a long position. Conversely, when the fast EMA crosses below the slow EMA and the volume condition is satisfied, it enters a short position. Stop loss and take profit levels are set based on fixed price distances, with the stop loss placed $100 below the entry price and the take profit placed $1500 above the entry price for long positions, and vice versa for short positions.

#### Strategy Principles
1. Calculate two EMA lines: the fast EMA (ema_fast) with a period of 65 and the slow EMA (ema_slow) with a period of 240.
2. Determine whether a bullish crossover (bullish_crossover) or a bearish crossover (bearish_crossover) occurs.
3. Set a volume threshold (volume_threshold) and only execute trades when the current volume exceeds this threshold.
4. Set a fixed risk amount (risk_per_trade) of $10 for each trade.
5. Calculate the position size (position_size) based on the risk amount and the stop loss distance (stop_loss_distance).
6. When a bullish crossover occurs and the volume condition is met, enter a long position with the stop loss set $100 below the entry price and the take profit set $1500 above the entry price.
7. When a bearish crossover occurs and the volume condition is met, enter a short position with the stop loss set $100 above the entry price and the take profit set $1500 below the entry price.

#### Strategy Advantages
1. The dual EMA crossover approach can effectively capture market trends, with the 65/240 period combination filtering out most noise and focusing on the main trends.
2. Introducing the volume filter condition helps avoid trading during periods of low volume, reducing market volatility risks.
3. The fixed risk amount position sizing method effectively controls the risk exposure of each trade, preventing excessive losses from a single trade.
4. The dynamic stop loss and take profit settings based on price distances allow for a larger profit potential than the loss potential, improving the long-term performance of the strategy.
5. Suitable for highly volatile instruments like BTC/USD, enabling the strategy to fully capture investment opportunities arising from price fluctuations.

#### Strategy Risks
1. As a trend-following indicator, EMA may lag in detecting trend reversals, potentially leading to delayed entries or exits.
2. The fixed risk amount may not dynamically adapt to market volatility conditions, resulting in suboptimal performance during extreme market movements (e.g., sharp rises or falls).
3. The setting of the volume threshold involves a certain level of subjectivity, and improper threshold settings may impact the strategy's effectiveness.
4. The fixed stop loss and take profit levels may not match the actual market volatility, leading to frequent stop-outs or profit-takings.
5. The strategy may underperform in choppy markets, with frequent crossovers potentially resulting in consecutive losing trades.

#### Strategy Optimization Directions
1. Consider introducing more EMA combinations as filter conditions, such as incorporating intermediate-term EMAs to build a multi-EMA system for improving signal reliability.
2. Optimize the position sizing approach, such as adopting a percentage risk method or the Kelly Criterion to dynamically adjust positions based on different market states.
3. Perform parameter optimization on the volume threshold to find the optimal threshold setting for enhancing strategy stability.
4. Optimize the stop loss and take profit level settings, adjusting them in real-time based on the latest market volatility conditions to increase flexibility and adaptability to the market.
5. Incorporate certain hedging components into the trend-following approach, such as utilizing counter-trend indicators like PSAR to assist in judging market oscillations and improve the strategy's ability to handle choppy markets.

#### Summary
This strategy employs a 65/240 dual EMA crossover as the basis for trend determination, combined with a volume filter condition to improve signal reliability. The fixed risk position sizing and fixed price stop loss/take profit settings can control risks to a certain extent and tilt the risk-reward ratio in a favorable direction. However, the strategy also faces issues such as relatively lagging trend detection, insufficient flexibility in position sizing, and lack of dynamic adjustments for stop loss and take profit levels. Future optimizations and improvements can focus on constructing a multi-EMA system, optimizing position sizing, implementing dynamic stop loss and take profit mechanisms, and incorporating hedging indicators to achieve more stable and reliable trading performance.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-05-06 00:00:00
end: 2024-05-13 00:00:00
period: 3m
basePeriod: 1m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("EMA Crossover Strategy with 1:3 RR, Volume Filter, and Custom Stop Loss/Take Profit (BTC)", overlay=true, currency="USD", initial_capital=100)

// Define EMA lengths
ema_length_fast = 65
ema_length_slow = 240

// Calculate EMAs
ema_fast = ta.ema(close, ema_length_fast)
ema_slow = ta.ema(close, ema_length_slow)

// Define crossover conditions
bullish_crossover = ta.crossover(ema_fast, ema_slow)
bearish_crossover = ta.crossunder(ema_fast, ema_slow)

// Plot EMAs
plot(ema_fast, color=color.blue, title="Fast EMA")
plot(ema_slow, color=color.red, title="Slow EMA")

// Define volume filter
volume_threshold = 1000 // Adjust as needed

// Define risk amount per trade
risk_per_trade = 0.5 // $10 USD

// Calculate position size based on risk amount
stop_loss_distance = 100
take_profit_distance = 1500
position_size = risk_per_trade / syminfo.mintick / stop_loss_distance

// Execute trades based on crossovers and volume filter
if (bullish_crossover and volume > volume_threshold)
    strategy.entry("Buy", strategy.long, qty=position_size)
    strategy.exit("Exit", "Buy", stop=close - stop_loss_distance, limit=close + take_profit_distance)
if (bearish_crossover and volume > volume_threshold)
    strategy.entry("Sell", strategy.short, qty=position_size)
    strategy.exit("Exit", "Sell", stop=close + stop_loss_distance, limit=close - take_profit_distance)

```

> Detail

https://www.fmz.com/strategy/451392

> Last Modified

2024-05-14 15:48:48
