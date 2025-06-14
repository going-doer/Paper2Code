
> Name

趋势过滤与异常退出的平滑移动平均止损止盈策略-Smooth-Moving-Average-Stop-Loss-Take-Profit-Strategy-with-Trend-Filter-and-Exception-Exit

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/f38fdf21a90bb8e63c.png)
[trans]
#### 概述
该策略通过使用平滑移动平均线(SMA)、相对强弱指数(RSI)、真实范围(TR)和成交量移动平均线(Volume MA)等指标,结合趋势过滤、交易量和波动率条件,在满足特定条件时进行交易。该策略的主要思路是在价格低于SMA200且处于下跌趋势、低交易量和低波动率的情况下进行买入,并设置止损和止盈位。同时,该策略还具有异常退出机制,即当RSI超过70或达到预设的止损止盈位时退出交易。

#### 策略原理
1. 计算SMA、RSI、交易量MA和TR MA等指标
2. 判断当前是否处于上升或下降趋势
3. 判断当前交易量和波动率是否处于低位
4. 当价格低于SMA200且满足低交易量和低波动率条件时,进行买入
5. 设置止损位为买入价的95%,止盈位为买入价的150%
6. 当RSI超过70或达到预设的止损止盈位时,退出交易
7. 当趋势发生变化且价格突破SMA时,强制平仓

#### 优势分析
1. 该策略结合了多个技术指标,可以更全面地分析市场状况
2. 通过趋势过滤和交易量、波动率条件,可以避免在不利的市场环境下进行交易
3. 设置明确的止损止盈位,可以有效控制风险
4. 异常退出机制可以在特定情况下及时平仓,防止进一步损失

#### 风险分析
1. 该策略依赖于多个参数的设置,参数的选择可能会影响策略表现
2. 在某些情况下,价格可能会在触发买入条件后快速反转,导致损失
3. 该策略没有考虑基本面因素,可能会受到重大事件的影响

#### 优化方向
1. 可以考虑引入更多的技术指标,如MACD、布林带等,以提高入场和出场的精确度
2. 可以优化止损止盈位的设置,如使用移动止损或动态止盈
3. 可以根据不同的市场状况,动态调整策略参数
4. 可以加入风险管理模块,如仓位管理、资金管理等

#### 总结
该策略通过综合使用多个技术指标,结合趋势过滤和交易量、波动率条件,在特定情况下进行交易。同时,设置明确的止损止盈位和异常退出机制,可以有效控制风险。但该策略也存在一定的局限性,如参数选择、市场异常等因素可能会影响策略表现。未来可以通过引入更多指标、优化参数设置、加入风险管理等方式来进一步改进该策略。

|| 

#### Overview
This strategy utilizes indicators such as the Smooth Moving Average (SMA), Relative Strength Index (RSI), True Range (TR), and Volume Moving Average (Volume MA) in combination with trend filters, volume, and volatility conditions to execute trades when specific criteria are met. The main idea behind this strategy is to enter a long position when the price is below the SMA200, the trend is downward, and both volume and volatility are low. Stop loss and take profit levels are set upon entry. Additionally, the strategy incorporates an exception exit mechanism, closing the position when the RSI exceeds 70 or when the preset stop loss or take profit levels are reached.

#### Strategy Principles
1. Calculate indicators such as SMA, RSI, Volume MA, and TR MA
2. Determine if the current trend is upward or downward
3. Check if the current volume and volatility are low
4. Enter a long position when the price is below the SMA200 and the low volume and volatility conditions are met
5. Set the stop loss at 95% and the take profit at 150% of the entry price
6. Exit the trade when the RSI exceeds 70 or when the preset stop loss or take profit levels are reached
7. Force close the position when the trend changes and the price breaks through the SMA

#### Advantage Analysis
1. This strategy combines multiple technical indicators for a more comprehensive analysis of market conditions
2. The trend filter and volume/volatility conditions help avoid trading in unfavorable market environments
3. Setting clear stop loss and take profit levels effectively manages risk
4. The exception exit mechanism allows for timely position closing in specific situations, preventing further losses

#### Risk Analysis
1. The strategy's performance may be affected by the choice of parameter settings
2. In some cases, the price may quickly reverse after triggering the entry condition, leading to losses
3. The strategy does not consider fundamental factors and may be influenced by significant events

#### Optimization Directions
1. Consider incorporating additional technical indicators such as MACD, Bollinger Bands, etc., to improve entry and exit accuracy
2. Optimize the stop loss and take profit level settings, such as using trailing stops or dynamic take profit
3. Dynamically adjust strategy parameters based on different market conditions
4. Introduce a risk management module, including position sizing and money management

#### Summary
This strategy combines multiple technical indicators with trend filters, volume, and volatility conditions to execute trades in specific situations. By setting clear stop loss and take profit levels and implementing an exception exit mechanism, the strategy effectively manages risk. However, the strategy has certain limitations, as factors such as parameter selection and market anomalies may impact its performance. Future improvements can be made by incorporating more indicators, optimizing parameter settings, and adding risk management components.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-05-01 00:00:00
end: 2024-05-31 23:59:59
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Strategia Stop Loss & Take Profit z Filtrem Trendu i Wyjątkiem", shorttitle="Smooth MA SL & TP with Exception", overlay=true)

// Parametry
tp_multiplier = input.float(1.5, title="Mnożnik Take Profit")
sl_percent = input.float(5, title="Procent Stop Loss")
wait_bars = input.int(3, title="Liczba Oczekiwanych Świec")
sma_period = input.int(200, title="Okres SMA")
rsi_period = input.int(14, title="Okres RSI")
vol_ma_period = input.int(20, title="Okres Średniej Wolumenu")
tr_ma_period = input.int(20, title="Okres Średniej Rzeczywistego Zakresu")

// Obliczenie Gładkiej Średniej Kroczącej
sma = ta.sma(close, sma_period)

// Obliczenie RSI
rsi = ta.rsi(close, rsi_period)

// Filtr Trendu
uptrend = close > sma
downtrend = close < sma

// Warunek konsolidacji: Niski wolumen i niska zmienność
niski_wolumen = volume < ta.sma(volume, vol_ma_period)
niska_zmienosc = ta.tr(true) < ta.sma(ta.tr(true), tr_ma_period)

// Warunek Wejścia (Long): Cena poniżej SMA 200 i filtr trendu w strefie czerwonej
warunek_wejscia = close < sma and niski_wolumen and niska_zmienosc and not uptrend

// Warunek Wyjścia ze strategii
warunek_wyjscia = downtrend and close > sma and ta.crossover(close, sma)

// Ustalanie Stop Loss i Take Profit
var float stop_loss = na
var float take_profit = na

var int indeks_wejscia = na

if (warunek_wejscia)
    stop_loss := close * (1 - sl_percent / 100)
    take_profit := close * (1 + tp_multiplier)
    indeks_wejscia := bar_index

// Handel
if (warunek_wejscia)
    strategy.entry("Long", strategy.long)

// Warunek Wyjścia: RSI w strefie wykupienia lub Stop Loss/Take Profit
if (strategy.opentrades != 0)
    if (rsi > 70)
        strategy.exit("Take Profit/Stop Loss", "Long", limit=take_profit)
    else if (bar_index - indeks_wejscia == wait_bars)
        strategy.exit("Take Profit/Stop Loss", "Long", stop=stop_loss, limit=take_profit)

// Wyjątek: Warunek Wyjścia z Longów na podstawie zmiany trendu
if (warunek_wyjscia)
    strategy.close("Long")

// Rysowanie RSI
rsi_plot = plot(rsi, title="RSI", color=color.blue)

// Rysowanie Gładkiej Średniej Kroczącej
sma_plot = plot(sma, color=color.gray, title="Smooth MA", linewidth=2)

// Rysowanie Filtru Trendu
fill(sma_plot, rsi_plot, color=downtrend ? color.new(color.red, 90) : na)

```

> Detail

https://www.fmz.com/strategy/453279

> Last Modified

2024-06-03 16:54:04
