
> Name

SMA均线交叉策略SMA-Moving-Average-Crossover-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/5f12ece77c6bf08525.png)

[trans]
### 概述
该策略是一个简单的SMA均线交叉策略。它使用两条不同周期的简单移动平均线(SMA),当快线从下向上穿过慢线时开仓做多,当快线从上向下穿过慢线时平仓。该策略可以自定义两条均线的长度,以及回测的起始和结束日期。

该策略的主要思路是利用均线的趋势特性和均线交叉的信号特性来进行交易。当快线在慢线之上时,说明当前处于上升趋势,应该持有多头头寸;当快线在慢线之下时,说明当前处于下降趋势,应该空仓观望。

### 策略原理
1. 计算两条不同周期的SMA,周期长度可以自定义。
2. 判断当前是否在回测时间窗口内,如果不在则不进行任何操作。
3. 如果快线从下向上穿过慢线,则开仓做多。
4. 如果快线从上向下穿过慢线,则平掉所有多头头寸。
5. 其余情况下空仓观望,不进行任何操作。

### 优势分析
1. 简单易懂,逻辑清晰,适合初学者学习和使用。
2. 均线是一个被广泛使用的技术指标,其趋势特性比较明显,能够较好地反映当前的市场趋势。
3. 均线交叉是一个经典的趋势追踪信号,可以快速地捕捉到趋势的变化。
4. 可以自定义均线周期和回测时间窗口,灵活性较强。
5. 适用于趋势性较强的品种和时间周期。

### 风险分析
1. 均线具有一定的滞后性,在市场波动较大、趋势反复时,可能会出现频繁的交叉信号,导致交易次数过多,手续费成本增加。
2. 该策略只能捕捉到单边上涨行情,对于震荡行情和单边下跌行情无能为力。
3. 均线参数的选择需要根据不同品种和时间周期进行优化,不同参数可能表现差异较大。
4. 该策略没有任何止损措施,在行情剧烈波动时可能面临较大回撤风险。

### 优化方向
1. 可以考虑加入适当的止损措施,比如基于ATR的移动止损,以控制单笔交易的最大亏损。
2. 可以考虑加入一些过滤条件,比如交易量、波动率等,以过滤掉一些假信号。
3. 可以考虑对参数进行优化,比如使用遗传算法等智能算法寻找最优参数组合。
4. 可以考虑将其他技术指标或交易信号与均线交叉相结合,比如MACD、RSI等,以提高策略的可靠性和有效性。

### 总结
SMA均线交叉策略是一个简单易懂、经典实用的趋势追踪策略,适合初学者学习和使用。它利用了均线的趋势特性和均线交叉的信号特性,可以快速捕捉到市场趋势的变化。但是该策略也存在一些局限性和风险,如滞后性、频繁交易、缺乏止损等。因此在实际应用中,需要根据具体情况进行适当的优化和改进,以提高策略的稳定性和盈利能力。

|| 

### Overview
This strategy is a simple SMA moving average crossover strategy. It uses two Simple Moving Averages (SMAs) with different lengths. When the fast MA crosses above the slow MA, it enters a long position. When the fast MA crosses below the slow MA, it closes the long position. The lengths of the two MAs can be customized, as well as the start and end dates for backtesting.

The main idea of this strategy is to utilize the trend characteristics of moving averages and the signal characteristics of MA crossovers for trading. When the fast MA is above the slow MA, it indicates an upward trend and a long position should be held. When the fast MA is below the slow MA, it indicates a downward trend and no position should be held.

### Strategy Principle
1. Calculate two SMAs with different lengths, which can be customized.
2. Check if the current time is within the backtesting window. If not, do nothing.
3. If the fast MA crosses above the slow MA, enter a long position.
4. If the fast MA crosses below the slow MA, close all long positions.
5. In other cases, stay flat and do nothing.

### Advantage Analysis
1. Simple and easy to understand, with clear logic, suitable for beginners to learn and use.
2. Moving average is a widely used technical indicator, with obvious trend characteristics, which can well reflect the current market trend.
3. MA crossover is a classic trend-following signal that can quickly capture changes in trends.
4. The lengths of MAs and the backtesting window can be customized, providing good flexibility.
5. Suitable for instruments and timeframes with strong trending characteristics.

### Risk Analysis
1. Moving averages have a certain lag. When the market fluctuates greatly and the trend reverses frequently, there may be frequent crossover signals, resulting in excessive trading and increased transaction costs.
2. This strategy can only capture upward trends, and is powerless in range-bound markets and downward trends.
3. The selection of MA parameters needs to be optimized for different instruments and timeframes. Different parameters may have large differences in performance.
4. This strategy does not have any stop-loss measures, and may face greater drawdown risks when the market fluctuates dramatically.

### Optimization Directions
1. Consider adding appropriate stop-loss measures, such as ATR-based trailing stop, to control the maximum loss of a single trade.
2. Consider adding some filtering conditions, such as trading volume and volatility, to filter out some false signals.
3. Consider optimizing parameters, such as using genetic algorithms or other intelligent algorithms to find the optimal parameter combination.
4. Consider combining other technical indicators or trading signals with the MA crossover, such as MACD and RSI, to improve the reliability and effectiveness of the strategy.

### Conclusion
The SMA moving average crossover strategy is a simple, easy-to-understand, classic and practical trend-following strategy that is suitable for beginners to learn and use. It utilizes the trend characteristics of moving averages and the signal characteristics of MA crossovers to quickly capture changes in market trends. However, this strategy also has some limitations and risks, such as lag, frequent trading, and lack of stop-loss. Therefore, in practical applications, it needs to be appropriately optimized and improved according to specific conditions to enhance the stability and profitability of the strategy.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_int_1|true|From Day|
|v_input_int_2|true|From Month|
|v_input_int_3|2018|From Year|
|v_input_int_4|30|Thru Day|
|v_input_int_5|9|Thru Month|
|v_input_int_6|2024|Thru Year|
|v_input_int_7|100|Slow MA lenght|
|v_input_int_8|30|Fast MA lenght|


> Source (PineScript)

``` pinescript
/*backtest
start: 2023-03-22 00:00:00
end: 2024-03-27 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © j0secyn

//@version=5
strategy("MA Cross", overlay=true, margin_long=100, margin_short=100, default_qty_value=100, default_qty_type=strategy.percent_of_equity, initial_capital=10000)

// === INPUT BACKTEST RANGE ===
fromDay   = input.int(defval = 1, title = "From Day", minval = 1, maxval = 31)
fromMonth = input.int(defval = 1, title = "From Month", minval = 1, maxval = 12)
fromYear  = input.int(defval = 2018,title = "From Year", minval = 1970)
thruDay   = input.int(defval = 30, title = "Thru Day", minval = 1, maxval = 31)
thruMonth = input.int(defval = 9, title = "Thru Month", minval = 1, maxval = 12)
thruYear  = input.int(defval = 2024, title = "Thru Year", minval = 1970)

slow_ma_length = input.int(defval = 100, title = "Slow MA lenght")
fast_ma_length = input.int(defval = 30, title = "Fast MA lenght")

// === FUNCTION EXAMPLE ===
start     = timestamp(fromYear, fromMonth, fromDay, 00, 00)            // backtest start  window
finish    = timestamp(thruYear, thruMonth, thruDay, 23, 59)            // backtest finish window
window()  => true

// === LOGIC ===
crossOv = ta.crossover(ta.sma(close, fast_ma_length), ta.sma(close, slow_ma_length))
crossUn = ta.crossunder(ta.sma(close, fast_ma_length), ta.sma(close, slow_ma_length))

// === EXECUTION ===
// strategy.entry("L", strategy.long, when = window() and crossOv)        // enter long when "within window of time" AND crossover
// strategy.close("L", when = window() and crossUn)                       // exits long when "within window of time" AND crossunder         
strategy.entry("L", strategy.long, when = window() and crossOv)        // enter long when "within window of time" AND crossover
strategy.close("L", when = window() and crossUn)                       // exits long when "within window of time" AND crossunder         
```

> Detail

https://www.fmz.com/strategy/446457

> Last Modified

2024-03-28 17:50:00
