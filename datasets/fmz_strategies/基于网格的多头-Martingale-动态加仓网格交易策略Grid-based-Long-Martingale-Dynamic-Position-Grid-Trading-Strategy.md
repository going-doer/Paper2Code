
> Name

基于网格的多头-Martingale-动态加仓网格交易策略Grid-based-Long-Martingale-Dynamic-Position-Grid-Trading-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/15857ddb0720d9acfd9.png)
[trans]

## 策略概述

该策略是一个基于网格的多头 Martingale 动态加仓网格交易策略。主要思想是在价格触及网格线时,根据已有仓位数量动态调整每次的加仓量,同时设置总的最大开仓数量。当价格上涨触及止盈价格时,平掉所有多头仓位。

## 策略原理

1. 根据 "Grid Size" 参数将价格划分为等距的网格。 
2. 在每个网格线位置挂多头限价单。
3. 当前价格触及网格线时:
   - 如果当前开仓数量为0,则以起始仓位 "start_lot" 开仓。
   - 如果当前开仓数量大于0,且当前网格价格低于上一个开仓价格,则在当前网格价格开新的多头仓位,仓位大小为上一仓位的 "multifactor" 倍。
4. 最大开仓数量由 "Max Open Orders" 参数控制。
5. 开仓后,在比开仓均价高出 "TakeProfit" 点的位置设置止盈线。
6. 当前价格触及止盈线时,平掉所有多头仓位,重置各项参数。

通过这种方式,在行情下跌过程中逐步加大仓位,当行情回升触及止盈线时获利了结。

## 策略优势

1. 动态加仓:根据当前开仓数量,动态调整每次加仓量,在行情持续下跌时逐步加大仓位,提高策略收益潜力。
2. 参数灵活:通过 "Grid Size"、"start_lot"、"multifactor" 等参数可以灵活控制网格大小、初始仓位、加仓倍数等。
3. 风险可控:通过 "Max Open Orders" 参数控制最大开仓数量,防止过度加仓。同时设置止盈线,在获利时及时了结,控制回撤。

## 策略风险

1. 无止损:该策略没有设置止损,如果行情一直向下突破,可能面临较大亏损风险。
2. 参数敏感:策略表现对 "multifactor" 等参数比较敏感,不适当的参数可能带来风险。
3. 高波动率:该策略在行情波动较大时频繁开仓、平仓,可能带来额外的滑点和手续费成本。

风险防范措施:
1. 结合风险承受能力,谨慎设置 "multifactor" 等参数。必要时可在代码中添加止损逻辑。
2. 仔细回测和模拟交易,选择合适的参数。
3. 评估策略在高波动行情下的表现。如果必要,可通过调整参数或限制使用场景等方式规避风险。

## 优化方向

1. 加入趋势判断:在开仓时根据价格走势判断趋势,如趋势看跌,则不开新的多头仓位,以降低风险。
2. 动态调整止盈:根据价格走势、波动率等指标,动态调整止盈位置,在趋势强劲时适当提高止盈点位,增强盈利能力。
3. 优化仓位管理:在加仓时,除了考虑加仓倍数,还可以结合账户资金、当前持仓量等,对加仓量进行更精细的控制。
4. 结合其他信号:将网格交易和其他趋势判断、震荡判断等指标信号结合,综合判断行情,指导交易。

这些优化可以提高策略适应性,更好地把握行情,提高盈利潜力和稳健性。同时通过更精细的仓位控制和风险管理,降低回撤,提高风险收益比。

## 总结

该基于网格的多头 Martingale 动态加仓网格交易策略,通过逐步加仓的方式试图在行情下跌过程中降低持仓均价,并在上涨时获利了结。策略通过参数设置具有较强的灵活性。但同时也潜在较大风险,需要仔细评估和控制风险。若能在此基础上加入趋势判断、动态止盈、仓位优化等,有望进一步提升策略性能。策略实现了价格触及网格线时自动开仓加仓,价格触及止盈线时自动全平仓的功能,整体逻辑较为清晰,但也有优化空间。策略适合在对行情和参数进行充分评估的基础上谨慎使用。

|| 

## Strategy Overview

This strategy is a grid-based long Martingale dynamic position grid trading strategy. The main idea is to dynamically adjust the position size based on the number of existing positions when the price hits grid lines, while setting a maximum total number of open positions. When the price rises and hits the take profit level, all long positions are closed.

## Strategy Principles

1. Divide the price into evenly spaced grids based on the "Grid Size" parameter.
2. Place long limit orders at each grid line.
3. When the current price hits a grid line:
   - If the current number of open positions is 0, open a position with the "start_lot" size.
   - If the current number of open positions is greater than 0 and the current grid price is lower than the previous entry price, open a new long position at the current grid price with a size that is "multifactor" times the previous position size.
4. The maximum number of open positions is controlled by the "Max Open Orders" parameter.
5. After opening a position, set a take profit level at "TakeProfit" points above the average entry price.
6. When the current price hits the take profit level, close all long positions and reset parameters.

In this way, the position size gradually increases during downtrends, and profits are taken when the price recovers and hits the take profit level.

## Strategy Advantages

1. Dynamic position sizing: Dynamically adjust the position size for each entry based on the current number of open positions. This gradually increases exposure during continued downtrends, enhancing the strategy's profit potential.
2. Flexible parameters: The "Grid Size", "start_lot", "multifactor" and other parameters allow flexible control over grid size, initial position size, position scaling factor, etc.
3. Controllable risk: The "Max Open Orders" parameter controls the maximum number of open positions to prevent overexposure. At the same time, setting a take profit level allows timely profit taking to control drawdowns.

## Strategy Risks

1. No stop loss: The strategy does not have stop losses. If the price continues breaking down, it may face significant loss risks.
2. Parameter sensitivity: The strategy's performance is quite sensitive to parameters like "multifactor". Inappropriate parameters may bring risks.
3. High volatility: The strategy may frequently open and close positions during highly volatile market conditions, potentially incurring extra slippage and commission costs.

Risk control measures:
1. Cautiously set parameters like "multifactor" based on risk tolerance. Stop loss logic can be added to the code if necessary.
2. Carefully backtest and paper trade to select appropriate parameters.
3. Evaluate the strategy's performance in highly volatile market conditions. If necessary, risk can be mitigated by adjusting parameters or restricting usage scenarios.

## Optimization Directions

1. Add trend detection: Determine the trend based on price action when opening positions. If the trend appears bearish, avoid opening new long positions to reduce risk.
2. Dynamic take profit: Dynamically adjust the take profit level based on price action, volatility and other indicators. Raise the take profit level when the trend is strong to enhance profitability.
3. Optimize position management: When adding to positions, besides considering the scaling factor, also incorporate account balance, current exposure, etc. to more precisely control position sizing.
4. Combine with other signals: Integrate grid trading with other trend detection, oscillation detection and other indicator signals to comprehensively judge the market and guide trading.

These optimizations can improve the strategy's adaptability to better capture market moves, enhance profit potential and robustness. At the same time, more precise position and risk management can reduce drawdowns and improve the risk-reward ratio.

## Summary

This grid-based long Martingale dynamic position grid trading strategy attempts to lower the average holding price during downtrends by gradually adding positions, and takes profits when prices rise. The strategy offers strong flexibility through parameter settings. However, it also carries significant potential risks that require careful assessment and control. If trend detection, dynamic take profit, position optimization and other features can be added, the strategy's performance may be further enhanced. The strategy realizes functions to automatically open and add positions when prices hit grid lines, and automatically close all positions when prices hit the take profit level. The overall logic is relatively clear, but there is room for optimization. The strategy is suitable for cautious use after thoroughly evaluating market conditions and parameters.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_11|0.5|Grid Size|
|v_input_1|0|(?TimeFrame)Started TimeFrame: 15|5|1|30|1H|4H|1D|1W|1M|
|v_input_2|3|(?Strategy Settings)Slippage by open order|
|v_input_3|0.01|Start lot|
|v_input_4|2|Started average since Order #|
|v_input_5|10|Max Open Orders|
|v_input_6|0|Direction: Only Long|Only Short|Long & Short|
|v_input_7|0|Type input: By Close|By grid line|
|v_input_8|1.5|Multifactor|
|v_input_9|2|Number of precision|
|v_input_10|true|TakeProfit|


> Source (PineScript)

``` pinescript
/*backtest
start: 2023-03-16 00:00:00
end: 2024-03-21 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © lagerta13
//@version=4
strategy("Grid A.", shorttitle="Grid(A)", overlay=true, format=format.price, precision=4, pyramiding = 100)

input_tf=input("15", "Started TimeFrame", 
 options = ["1", "5", "15", "30", "1H", "4H", "1D", "1W", "1M"],
 group="TimeFrame") 

// avg_tf=input("5", "Average TimeFrame", 
//  options = ["1", "5", "15", "30", "1H", "4H", "1D", "1W", "1M"],
//  group="TimeFrame")

slip_Hilo = input(3.0, "Slippage by open order", group="Strategy Settings")
start_lot = input(0.01, "Start lot", group="Strategy Settings")
start_avg = input(2, "Started average since Order #", group="Strategy Settings")
MaxTrades_Hilo = input(10, "Max Open Orders", group="Strategy Settings")
dropdown_selection = input("Only Long", "Direction", options=["Only Long", "Only Short", "Long & Short"],
 group="Strategy Settings")
type_selection = input("By Close", "Type input", options=["By Close", "By grid line"],
 group="Strategy Settings")

multifactor = input(1.5, "Multifactor", group="Strategy Settings")
precision_lot = input(2, "Number of precision", group="Strategy Settings")
take_profit = input(1, "TakeProfit", group="Strategy Settings")

// PipStep_S1 = input(30)
// PipStepX_S1 = input(1.2)
// dinamicStep = input(false, "Dinamic Step for AVG")

get_size_lot_order(number, multi, prec, avg_from, lot_from) =>
	res = lot_from
	for i = 1 to number
		if i >= avg_from
			res := round(res * multi, precision = prec)
	res

var float[] entry_levels = array.new_float(MaxTrades_Hilo + 1)

for i = 0 to MaxTrades_Hilo
    array.push(entry_levels, 0)

gridSize = input(0.5, title="Grid Size")
gridLevels = int(close / gridSize) * gridSize

var int num_open_orders = 0
var float sum_price_orders = 0
var float entry_lot = 0

buy_condition = num_open_orders < MaxTrades_Hilo and gridLevels[0]<gridLevels[1] and dropdown_selection != "Only Short"

if (buy_condition)

    if num_open_orders == 0
        lot = get_size_lot_order(num_open_orders, multifactor, precision_lot, start_avg, start_lot)
        sum_price_orders := sum_price_orders + gridLevels[1] * lot 

        strategy.entry("buy" + tostring(num_open_orders), true, qty=lot, limit=gridLevels[1]+slip_Hilo) 
        // strategy.order("buy" + tostring(num_open_orders), true, qty=lot, limit=gridLevels[1]) 

        array.set(entry_levels, num_open_orders, gridLevels[1])

        entry_lot := entry_lot + lot
        num_open_orders := num_open_orders + 1

    else
        if gridLevels[1] < (array.get(entry_levels, num_open_orders - 1))
            lot = get_size_lot_order(num_open_orders, multifactor, precision_lot, start_avg, start_lot)
            sum_price_orders := sum_price_orders + gridLevels[1] * lot 
            entry_lot := entry_lot + lot

            strategy.entry("buy" + tostring(num_open_orders), true, qty=lot, limit=gridLevels[1]+slip_Hilo) 

            // +" S:" + tostring(sum_price_orders / (entry_lot)) + " Prev:" + tostring(array.get(entry_levels, num_open_orders - 1))
            // strategy.order("buy" + tostring(num_open_orders), true, qty=lot, limit=gridLevels[1]) 
            array.set(entry_levels, num_open_orders, gridLevels[1])

            num_open_orders := num_open_orders + 1


take = sum_price_orders > 0 and take_profit + (sum_price_orders / entry_lot) < high ? high : na
plotshape(take, location = location.belowbar, color = color.white)


strategy.exit("tp", comment = "TP " + tostring(num_open_orders), qty = entry_lot, limit = take_profit + (sum_price_orders / entry_lot))


if sum_price_orders > 0 and take_profit + (sum_price_orders / entry_lot) <= high
    num_open_orders := 0
    sum_price_orders := 0
    entry_lot := 0
    for i = 0 to MaxTrades_Hilo
        array.set(entry_levels, i, 0)

plot(gridLevels, color=color.blue, style=plot.style_circles, linewidth=2)

 
```

> Detail

https://www.fmz.com/strategy/445826

> Last Modified

2024-03-22 15:12:33
