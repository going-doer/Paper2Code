
> Name

基于双移动平均线交叉策略-Dual-Moving-Average-Crossover-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/600348a9f53c14dc35.png)
[trans]
#### 概述
双移动平均线交叉策略是一种常见的量化交易策略。该策略使用两条不同周期的移动平均线作为买卖信号,当短期均线上穿长期均线时买入,当短期均线下穿长期均线时卖出。该策略代码支持多种常见的移动平均线类型,如简单移动平均线(SMA)、指数移动平均线(EMA)、双指数移动平均线(DEMA)、三指数移动平均线(TEMA)、加权移动平均线(WMA)和成交量加权移动平均线(VWMA),并且可以灵活设置短期均线和长期均线的周期。同时,该策略还支持选择不同的价格类型来计算均线,如收盘价、最高价、开盘价、最低价、典型价格和中间价格。

#### 策略原理 
该策略的核心原理是利用两条不同周期移动平均线的趋势特性和滞后性来捕捉价格趋势。一般来说,短期均线对价格变化更敏感,而长期均线则相对滞后。当价格处于上升趋势时,短期均线会先于长期均线向上运动并最终上穿长期均线,形成"金叉"买入信号;反之,当价格处于下降趋势时,短期均线会先于长期均线向下运动并最终下穿长期均线,形成"死叉"卖出信号。通过捕捉金叉和死叉信号,该策略可以顺应价格的主要趋势方向进行交易。

#### 策略优势
1. 简单易用:双移动平均线交叉策略是一种简单易懂、容易实现的量化交易策略,适合新手traders学习和使用。

2. 适用性广:该策略可以应用于各种金融市场和交易标的,如股票、期货、外汇、加密货币等,通用性强。

3. 参数灵活:该策略代码支持多种常见的移动平均线类型和价格类型,用户可以根据自己的需求灵活设置参数,以适应不同的市场环境和交易风格。

4. 趋势跟踪:通过两条不同周期均线的交叉信号,该策略可以较好地捕捉价格的主要趋势,有助于顺势而为、避免逆势交易。

#### 策略风险
1. 滞后性:移动平均线本质上是一种趋势跟踪指标,存在一定的滞后性,可能错过最佳的入场和出场时机。

2. 震荡市中失效:在震荡市或者横盘整理行情中,价格波动较大,均线交叉信号频繁,可能导致策略频繁交易,造成高昂的交易成本和资金损失。

3. 参数优化难:均线周期的选择对策略效果影响很大,但最优参数往往因市场状况不同而有所差异,很难找到放之四海而皆准的最优参数组合。

#### 策略优化方向
1. 引入趋势过滤:在均线交叉信号的基础上,可以结合其他趋势指标如MACD、ADX等进行趋势过滤,只在趋势明确的情况下进行交易,避免在震荡市中频繁交易。

2. 优化止盈止损:在策略中加入合理的止盈止损逻辑,如移动止损、波动率止损等,以控制单次交易风险,提高策略的风险收益比。

3. 动态参数优化:针对不同的市场环境,可以定期对均线周期等参数进行动态优化,使策略能够自适应市场变化,提高稳健性。

4. 多因子组合:将双移动平均线交叉信号与其他有效的量化因子(如动量、价值、成交量等)进行组合,形成更加稳健有效的多因子策略。

#### 总结
双移动平均线交叉策略是一种简单经典的趋势追踪策略,通过两条不同周期均线的交叉信号来捕捉价格趋势,适合趋势性市场。但该策略也存在滞后性和参数优化难等问题,需要结合其他方法进行优化和改进,如趋势过滤、动态参数优化、多因子组合等,以提高策略的适用性和稳健性。总的来说,双移动平均线交叉策略可以作为量化交易的基础策略之一,值得广大量化爱好者学习和研究。

|| 

#### Overview
The Dual Moving Average Crossover Strategy is a common quantitative trading strategy. This strategy uses two moving averages with different periods as buy and sell signals. It buys when the short-term moving average crosses above the long-term moving average and sells when the short-term moving average crosses below the long-term moving average. The strategy code supports various common types of moving averages, such as Simple Moving Average (SMA), Exponential Moving Average (EMA), Double Exponential Moving Average (DEMA), Triple Exponential Moving Average (TEMA), Weighted Moving Average (WMA), and Volume Weighted Moving Average (VWMA). It also allows flexible settings for the periods of the short-term and long-term moving averages. Moreover, the strategy supports selecting different price types for calculating the moving averages, including close, high, open, low, typical price, and median price.

#### Strategy Principle
The core principle of this strategy is to capture price trends by leveraging the trend characteristics and lag of two moving averages with different periods. Generally speaking, the short-term moving average is more sensitive to price changes, while the long-term moving average is relatively lagging. When the price is in an uptrend, the short-term moving average will move upward before the long-term moving average and eventually cross above it, forming a "golden cross" buy signal. Conversely, when the price is in a downtrend, the short-term moving average will move downward before the long-term moving average and eventually cross below it, forming a "death cross" sell signal. By capturing the golden cross and death cross signals, this strategy can trade in line with the main trend direction of the price.

#### Strategy Advantages
1. Simple and easy to use: The Dual Moving Average Crossover Strategy is a simple, easy-to-understand, and easy-to-implement quantitative trading strategy, suitable for novice traders to learn and use.

2. Wide applicability: This strategy can be applied to various financial markets and trading instruments, such as stocks, futures, forex, cryptocurrencies, etc., with strong versatility.

3. Flexible parameters: The strategy code supports multiple common types of moving averages and price types, allowing users to flexibly set parameters according to their needs to adapt to different market conditions and trading styles.

4. Trend tracking: By using the crossover signals of two moving averages with different periods, this strategy can effectively capture the main trend of the price, which helps to follow the trend and avoid counter-trend trading.

#### Strategy Risks
1. Lag: Moving averages are essentially trend-following indicators and have a certain lag, which may miss the best entry and exit timings.

2. Ineffectiveness in range-bound markets: In range-bound or sideways markets, price fluctuations are large, and moving average crossover signals occur frequently, which may lead to frequent trading and result in high trading costs and capital losses.

3. Difficulty in parameter optimization: The selection of moving average periods has a significant impact on the strategy performance, but the optimal parameters often vary depending on market conditions, making it challenging to find universally applicable optimal parameter combinations.

#### Strategy Optimization Directions
1. Introduce trend filters: In addition to the moving average crossover signals, other trend indicators such as MACD and ADX can be incorporated for trend filtering, trading only when the trend is clear to avoid frequent trading in range-bound markets.

2. Optimize take-profit and stop-loss: Incorporate reasonable take-profit and stop-loss logic into the strategy, such as trailing stop-loss and volatility-based stop-loss, to control single-trade risk and improve the strategy's risk-reward ratio.

3. Dynamic parameter optimization: For different market environments, periodically perform dynamic optimization on parameters such as moving average periods to enable the strategy to adapt to market changes and improve robustness.

4. Multi-factor combination: Combine the dual moving average crossover signals with other effective quantitative factors (such as momentum, value, volume, etc.) to form a more robust and effective multi-factor strategy.

#### Summary
The Dual Moving Average Crossover Strategy is a simple and classic trend-following strategy that captures price trends through the crossover signals of two moving averages with different periods, suitable for trending markets. However, this strategy also has issues such as lag and difficulty in parameter optimization, requiring combinations with other methods for optimization and improvement, such as trend filtering, dynamic parameter optimization, multi-factor combination, etc., to enhance the strategy's applicability and robustness. Overall, the Dual Moving Average Crossover Strategy can serve as one of the foundational strategies in quantitative trading, worthy of learning and research by quantitative enthusiasts.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-05-08 00:00:00
end: 2024-05-13 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © SustainableInvestment

//@version=5
strategy("Moving average strategy (이동평균선 전략)", overlay=true)

// === INPUTS ===

basisType   = input.string(defval = "EMA", title = "MA Type: SMA, EMA, DEMA, TEMA, WMA, VWMA ",options=["SMA", "EMA", "DEMA", "TEMA", "WMA", "VWMA"])
shortLen    = input.int(defval = 1, title = "Short MA Period", minval = 1)
longLen    = input.int(defval = 20, title = "Long MA Period", minval = 1)
price       = input.string(defval = "Typical", title = "Price Type : Close, High, Open, Low, Typical, Center ",options=["Close", "High", "Open", "Low", "Typical", "Center"])

// === BASE FUNCTIONS ===
// 가격 종류 설정
priceType(price) =>
    Typical = (high+low+close)/3
    Center  = (high+low) / 2
    price=="High"?high : price=="Low"?low : price=="Open"?open : price=="Typical"?Typical : price=="Center"?Center : close

// 이동평균선 종류 설정
variant(type, src, len) =>
    v1 = ta.sma(src, len)                                                  // Simple
    v2 = ta.ema(src, len)                                                  // Exponential
    v3 = 2 * v2 - ta.ema(v2, len)                                          // Double Exponential
    v4 = 3 * (v2 - ta.ema(v2, len)) + ta.ema(ta.ema(v2, len), len)         // Triple Exponential
    v5 = ta.wma(src, len)                                                  // Weighted
    v6 = ta.vwma(src, len)                                                 // Volume Weighted
    
    type=="EMA"?v2 : type=="DEMA"?v3 : type=="TEMA"?v4 : type=="WMA"?v5 : type=="VWMA"?v6 : v1

longCondition = ta.crossover(variant(basisType, priceType(price), shortLen), variant(basisType, priceType(price), longLen))
if (longCondition)
    strategy.entry("Long Entry", strategy.long)

exitCondition = ta.crossunder(variant(basisType, priceType(price), shortLen), variant(basisType, priceType(price), longLen))
if (exitCondition)
    strategy.close("Long Entry","Long Exit")

```

> Detail

https://www.fmz.com/strategy/451389

> Last Modified

2024-05-14 15:37:54
