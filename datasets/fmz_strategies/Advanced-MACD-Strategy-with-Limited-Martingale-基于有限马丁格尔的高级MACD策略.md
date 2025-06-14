
> Name

Advanced-MACD-Strategy-with-Limited-Martingale-基于有限马丁格尔的高级MACD策略

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/19d896963817e1d58ec.png)
[trans]
#### 概述
该策略结合了MACD指标和有限的马丁格尔资金管理方法,旨在捕捉市场趋势变化时的交易机会。当MACD快线与慢线发生金叉时产生买入信号,死叉时产生卖出信号。同时,策略采用了有限的马丁格尔方法来控制回撤,最多加仓3次。策略对单笔交易设置了1%的固定止盈止损。

#### 策略原理
1. 计算MACD指标的快线、慢线和信号线。
2. 判断快线与慢线的交叉情况,金叉做多,死叉做空。
3. 设置固定的单笔交易量(0.01)。
4. 记录上一笔交易的净利润。
5. 若当前净利润小于上一笔交易,且加仓次数小于3次,则将下一笔交易量翻倍,加仓次数加1;否则重置交易量和加仓次数。
6. 对每笔多单,当价格上涨1%时止盈,下跌1%时止损;空单则反之。
7. 在图表上标记买卖点。

#### 策略优势
1. 结合了趋势跟踪指标MACD和马丁格尔资金管理,能较好地把握趋势行情。
2. 设置了固定止盈止损,控制了单笔交易风险。
3. 采用有限的马丁格尔加仓,在趋势延续时能获得更高收益。
4. 加仓次数最多为3次,避免了过度加仓导致的爆仓风险。
5. 图表标记买卖信号,方便观察策略效果。

#### 策略风险
1. MACD指标可能出现信号与价格背离的情况,导致误判。
2. 固定的止盈止损比例,可能错失更大的利润空间或承担更大损失。
3. 马丁格尔加仓虽然限制在3次以内,但在震荡行情下连续亏损时,仍有爆仓的风险。
4. 策略未考虑市场异常波动的情况,如瞬间跳空,可能导致无法按预期成交。

#### 策略优化方向
1. 可以考虑引入趋势确认指标,如MA,过滤MACD的信号。
2. 优化止盈止损的设置,如采用ATR或百分比作为动态止损。
3. 对加仓次数和加仓比例进行优化,控制回撤风险。
4. 设置异常行情的应对机制,如价格跳空时暂停交易等。
5. 考虑引入仓位管理,根据市场波动状况动态调整仓位。

#### 总结
该策略通过MACD指标捕捉趋势,同时采用有限马丁格尔控制回撤,能在趋势行情中取得不错的效果。但策略也存在一定的风险,如信号失效、固定止损等。通过引入其他指标、优化参数设置、仓位管理等方法,可以进一步提升该策略的稳健性和收益性。

|| 

#### Overview
This strategy combines the MACD indicator with a limited Martingale money management method to capture trading opportunities when market trends change. A buy signal is generated when the MACD fast line crosses above the slow line, and a sell signal is generated when the fast line crosses below the slow line. At the same time, the strategy uses a limited Martingale method to control drawdowns, with a maximum of 3 additional positions. The strategy sets a fixed take profit and stop loss of 1% for each trade.

#### Strategy Principles
1. Calculate the fast line, slow line, and signal line of the MACD indicator.
2. Determine the crossover of the fast and slow lines, going long on a bullish crossover and short on a bearish crossover.
3. Set a fixed trading volume (0.01) for each trade.
4. Record the net profit of the previous trade.
5. If the current net profit is less than the previous trade and the number of additional positions is less than 3, double the next trading volume and increase the number of additional positions by 1; otherwise, reset the trading volume and number of additional positions.
6. For each long position, take profit when the price rises by 1% and stop loss when it falls by 1%; vice versa for short positions.
7. Mark buy and sell points on the chart.

#### Strategy Advantages
1. Combines the MACD trend-following indicator with Martingale money management, which can better capture trending markets.
2. Sets fixed take profit and stop loss levels to control individual trade risk.
3. Uses limited Martingale position sizing to achieve higher returns when trends continue.
4. Limits the maximum number of additional positions to 3, avoiding the risk of excessive position sizing leading to account blowouts.
5. Marks buy and sell signals on the chart for easy observation of strategy performance.

#### Strategy Risks
1. The MACD indicator may experience divergence between signals and price, leading to misjudgment.
2. Fixed take profit and stop loss ratios may miss out on larger profit opportunities or incur greater losses.
3. Although Martingale position sizing is limited to 3 times, there is still a risk of account blowouts when experiencing consecutive losses in choppy markets.
4. The strategy does not consider abnormal market fluctuations, such as sudden gaps, which may result in inability to execute as expected.

#### Strategy Optimization Directions
1. Consider introducing trend confirmation indicators, such as MA, to filter MACD signals.
2. Optimize the take profit and stop loss settings, such as using ATR or percentages for dynamic stop losses.
3. Optimize the number and ratio of additional positions to control drawdown risk.
4. Set up mechanisms to deal with abnormal market conditions, such as suspending trading when prices gap.
5. Consider introducing position sizing to dynamically adjust positions based on market volatility.

#### Summary
This strategy captures trends through the MACD indicator while using limited Martingale to control drawdowns, which can achieve good results in trending markets. However, the strategy also has certain risks, such as signal failure and fixed stop losses. By introducing other indicators, optimizing parameter settings, position sizing, and other methods, the robustness and profitability of this strategy can be further improved.

[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-04-01 00:00:00
end: 2024-04-30 23:59:59
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Advanced MACD Strategy with Limited Martingale", overlay=true, initial_capital=100)

// MACD 설정
fastLength = 15
slowLength = 30
signalSmoothing = 9
[macdLine, signalLine, _] = ta.macd(close, fastLength, slowLength, signalSmoothing)

// 계약수 및 이전 거래 결과 기록
var float contractSize = 0.01
var int martingaleCount = 0 // 마틴게일 카운트
var float lastTradeResult = 0

// 매수 및 매도 조건
longCondition = ta.crossover(macdLine, signalLine)
shortCondition = ta.crossunder(macdLine, signalLine)

// 매수 신호
if (longCondition)
    strategy.entry("Long", strategy.long, qty=contractSize)
    lastTradeResult := strategy.netprofit

// 매도 신호
if (shortCondition)
    strategy.entry("Short", strategy.short, qty=contractSize)
    lastTradeResult := strategy.netprofit

// 익절 및 손절 조건
strategy.close("Long", when=(close / strategy.position_avg_price >= 1.01))
strategy.close("Short", when=(strategy.position_avg_price / close >= 1.01))
strategy.close("Long", when=(close / strategy.position_avg_price <= 0.99))
strategy.close("Short", when=(strategy.position_avg_price / close <= 0.99))

// 마틴게일 전략 적용
if (strategy.netprofit < lastTradeResult)
    if (martingaleCount < 3)
        contractSize := contractSize * 2
        martingaleCount := martingaleCount + 1
    else
        contractSize := 0.01
        martingaleCount := 0
else
    contractSize := 0.01
    martingaleCount := 0

// 매수, 매도 포인트 화살표로 표시
plotshape(series=longCondition, location=location.belowbar, color=color.green, style=shape.labelup, text="Buy")
plotshape(series=shortCondition, location=location.abovebar, color=color.red, style=shape.labeldown, text="Sell")
```

> Detail

https://www.fmz.com/strategy/451072

> Last Modified

2024-05-11 17:24:43
