
> Name

基于斐波那契回调和成交量增量的交易策略-Delta-Volume-with-Fibonacci-Levels-Trading-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/117e0fad60aeaca620f.png)

[trans]
#### 概述
该策略是一种基于成交量增量(Delta Volume)和斐波那契回调(Fibonacci Retracement)的交易策略。它通过比较一段时间内买方和卖方的成交量来判断市场趋势,同时利用斐波那契回调线来确定进出场点。当买方成交量超过卖方成交量,且价格突破61.8%的斐波那契回调线时进场做多;当卖方成交量超过买方成交量,且价格跌破38.2%的斐波那契回调线时平仓。

#### 策略原理
1. 计算指定周期内的买方成交量和卖方成交量,并存储在数组中。
2. 计算成交量增量(Delta Volume),即买方成交量减去卖方成交量。
3. 计算指定周期内的最高价和最低价,并根据它们计算38.2%和61.8%的斐波那契回调线。
4. 当成交量增量大于0(买方成交量大于卖方成交量),且收盘价高于61.8%的斐波那契回调线时,开仓做多。
5. 当成交量增量小于0(卖方成交量大于买方成交量),且收盘价低于38.2%的斐波那契回调线时,平仓。

#### 策略优势
1. 结合成交量和价格两个维度,可以更全面地判断市场趋势。
2. 利用斐波那契回调线作为进出场点,有明确的技术支撑。
3. 成交量增量指标可以反映市场供需关系,是一个领先指标。
4. 参数可调,适用于不同的市场和交易品种。

#### 策略风险
1. 在震荡市中,频繁的进出场可能导致较高的交易成本。
2. 如果市场出现剧烈波动,价格可能会快速突破斐波那契回调线,导致错过最佳进出场点。
3. 该策略依赖于历史数据进行计算,对于新上市的交易品种或数据缺失的情况,可能会影响策略的有效性。

#### 策略优化方向
1. 可以考虑引入其他技术指标,如移动平均线、RSI等,以确认趋势和进出场点。
2. 对于不同的市场和交易品种,可以优化成交量增量和斐波那契回调的计算周期和参数。
3. 在进场后,可以设置移动止损或止盈,以控制风险和锁定利润。
4. 可以结合市场情绪指标,如Fear & Greed Index,对策略进行动态调整。

#### 总结
该策略通过结合成交量增量和斐波那契回调线,在趋势形成初期进场,在趋势可能反转时出场,以捕捉市场的主要趋势。但在震荡市中可能会面临频繁交易的风险,因此需要结合其他指标和风控手段进行优化。总体而言,该策略思路清晰,逻辑严谨,可以作为一个基础策略进行进一步的开发和应用。

|| 

#### Overview
This strategy is a trading strategy based on Delta Volume and Fibonacci Retracement. It determines the market trend by comparing the volume of buyers and sellers over a period of time, while using Fibonacci retracement lines to determine entry and exit points. When the buyer's volume exceeds the seller's volume and the price breaks through the 61.8% Fibonacci retracement line, it enters a long position; when the seller's volume exceeds the buyer's volume and the price falls below the 38.2% Fibonacci retracement line, it closes the position.

#### Strategy Principle
1. Calculate the buyer's volume and seller's volume for the specified period and store them in arrays.
2. Calculate the Delta Volume, which is the buyer's volume minus the seller's volume.
3. Calculate the highest and lowest prices for the specified period, and calculate the 38.2% and 61.8% Fibonacci retracement lines based on them.
4. When the Delta Volume is greater than 0 (buyer's volume is greater than seller's volume) and the closing price is higher than the 61.8% Fibonacci retracement line, open a long position.
5. When the Delta Volume is less than 0 (seller's volume is greater than buyer's volume) and the closing price is lower than the 38.2% Fibonacci retracement line, close the position.

#### Strategy Advantages
1. By combining volume and price dimensions, it can more comprehensively judge the market trend.
2. Using Fibonacci retracement lines as entry and exit points has clear technical support.
3. Delta Volume indicator can reflect the supply and demand relationship in the market, which is a leading indicator.
4. Parameters are adjustable and applicable to different markets and trading varieties.

#### Strategy Risks
1. In the oscillating market, frequent entries and exits may lead to higher transaction costs.
2. If the market fluctuates drastically, prices may quickly break through Fibonacci retracement lines, leading to missing the best entry and exit points.
3. The strategy relies on historical data for calculation. For newly listed trading varieties or data missing situations, it may affect the effectiveness of the strategy.

#### Strategy Optimization Direction
1. Consider introducing other technical indicators, such as moving averages, RSI, etc., to confirm trends and entry/exit points.
2. For different markets and trading varieties, optimize the calculation period and parameters of Delta Volume and Fibonacci Retracement.
3. After entering a position, set a trailing stop loss or take profit to control risk and lock in profits.
4. Combine with market sentiment indicators, such as the Fear & Greed Index, to dynamically adjust the strategy.

#### Summary
By combining Delta Volume and Fibonacci Retracement lines, this strategy enters when a trend is forming and exits when the trend may reverse, in order to capture the main trend of the market. However, it may face the risk of frequent trading in the oscillating market, so it needs to be optimized with other indicators and risk control measures. Overall, the strategy is clear in thought, logically rigorous, and can be used as a basic strategy for further development and application.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-05-09 00:00:00
end: 2024-05-14 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Delta Volume with Fibonacci Levels Strategy", overlay=true)

// Input pour la période de calcul du volume et du delta
N = input(14, title="Période du Delta Volume")
fibLength = input(21, title="Fibonacci Lookback Period")

// Choix de la barre pour l'entrée et la sortie des trades
entryPriceType = input.string("close", title="Entry Price Type", options=["open", "close"])
exitPriceType = input.string("close", title="Exit Price Type", options=["open", "close"])

// Correction des dates de début et de fin pour le backtest
startDate = input(defval = timestamp("2021-01-01"), title = "Start Date")
endDate = input(defval = timestamp("2022-01-01"), title = "End Date")

// Calcul des volumes des acheteurs et des vendeurs
buyerVolume = array.new_float()
sellerVolume = array.new_float()

// Mise à jour des volumes à chaque bougie
buyVol = close > open ? volume : 0
sellVol = close < open ? volume : 0
array.unshift(buyerVolume, buyVol)
array.unshift(sellerVolume, sellVol)

// Gardez seulement les N dernières valeurs pour le delta volume
if array.size(buyerVolume) > N
    array.pop(buyerVolume)
if array.size(sellerVolume) > N
    array.pop(sellerVolume)

// Calcul du delta de volume
sumBuyerVolume = array.sum(buyerVolume)
sumSellerVolume = array.sum(sellerVolume)
deltaVolume = sumBuyerVolume - sumSellerVolume

// Calcul du plus haut et du plus bas pour Fibonacci
highestPrice = ta.highest(high, fibLength)
lowestPrice = ta.lowest(low, fibLength)

// Fibonacci Levels
fib382 = lowestPrice + (highestPrice - lowestPrice) * 0.5
fib618 = lowestPrice + (highestPrice - lowestPrice) * 0.786


// Vérification des dates pour le backtest
bool isInDateRange = true

// Conditions d'entrée et de sortie
entryPrice = entryPriceType == "open" ? open : close
exitPrice = exitPriceType == "open" ? open : close

// Acheter quand le volume des acheteurs dépasse celui des vendeurs, le prix est au-dessus du niveau 61.8% de Fibonacci
if isInDateRange and deltaVolume > 0 and entryPrice > fib618
    strategy.entry("Buy", strategy.long)

// Vendre quand le volume des vendeurs dépasse celui des acheteurs, le prix est en dessous du niveau 38.2% de Fibonacci
if isInDateRange and deltaVolume < 0 and exitPrice < fib382
    strategy.close("Buy")

// Affichage des niveaux de Fibonacci et du delta de volume
plot(fib382, color=color.red, title="Fibonacci 38.2%")
plot(fib618, color=color.green, title="Fibonacci 61.8%")
plot(deltaVolume, color=deltaVolume > 0 ? color.green : color.red, title="Delta Volume")

```

> Detail

https://www.fmz.com/strategy/451491

> Last Modified

2024-05-15 10:45:58
