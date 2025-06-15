
> Name

短中长期三重移动平均线趋势跟踪策略-Short-Medium-Long-Term-Triple-Moving-Average-Trend-Following-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/127746d729b4d14af8b.png)

[trans]
#### 概述
"短中长期三重移动平均线趋势跟踪策略"是一种利用不同周期移动平均线组合来捕捉市场趋势并进行交易的量化投资策略。该策略以3日最低价短期移动平均线、3日最高价短期移动平均线和30日收盘价中期移动平均线为基础,通过比较收盘价与这三条均线的相对位置来判断趋势方向并发出交易信号。当收盘价跌破3日最低价均线且高于30日收盘价均线时做多,而当收盘价突破3日最高价均线时平仓。

#### 策略原理
该策略的核心原理是利用移动平均线的趋势特性和不同周期均线的交叉关系来捕捉市场趋势。短期的3日最低价和最高价移动平均线能够快速反应价格的短期波动,而中期的30日收盘价移动平均线则能体现更大级别的趋势方向。

当收盘价跌破3日最低价均线且高于30日收盘价均线时,说明短期出现回撤但中期趋势依然看涨,此时入场做多。而当收盘价突破3日最高价均线时,短期上涨动能衰竭,此时平仓了结。通过短中期均线的配合使用,策略可以在趋势产生初期介入,并在趋势结束前及时退出。

#### 策略优势
1. 趋势捕捉能力强。策略利用短中期不同周期均线的配合,能较好地捕捉市场中长期趋势,顺势而为。
2. 及时止盈。通过30日中期均线判断趋势方向,并利用3日短期均线及时兑现利润,避免过度持仓。 
3. 参数简单,易于理解和优化。策略仅使用了三条均线,逻辑清晰,参数也容易进行优化测试。
4. 适应性强。短中期均线组合能适应不同波动周期的市场,对趋势和震荡行情都有一定适应性。

#### 策略风险
1. 频繁交易。策略在震荡行情中可能出现频繁交易信号,增加了交易成本。
2. 突发事件风险。市场若出现剧烈异常波动,均线系统可能失效,造成较大回撤。
3. 参数失效风险。市场趋势节奏若发生变化,原有参数可能失去效力,需要重新优化。
4. 仓位管理欠缺。策略未设置仓位管理和资金管理规则,风险控制能力有限。

#### 策略优化方向  
1. 增加仓位管理。可以根据趋势强度、波动率等指标动态调节仓位,提高收益风险比。
2. 结合其他趋势指标。可以引入MACD、DMI等其他趋势类指标作为辅助,提高趋势判断准确度。
3. 优化参数。针对不同标的和周期,对均线参数进行优化,找出最佳参数组合。
4. 加入止损。设置合理止损位,控制单次交易最大亏损,提高策略稳健性。
5. 适当过滤。减少在震荡行情下的交易频率,可以考虑加入ATR等波动率过滤机制。

#### 总结
"短中长期三重移动平均线趋势跟踪策略"是一个利用不同周期均线捕捉趋势的量化交易策略。它通过比较价格与3日最低价均线、3日最高价均线和30日均线的位置关系,在趋势产生初期介入并在结束前退出。策略逻辑简单易懂,适应性较强,但也存在频繁交易、仓位管理欠缺等风险。未来可从仓位管理、止损止盈、参数优化等方面对策略进行完善,以期获得更稳健的收益。

|| 

#### Overview
The "Short-Medium-Long Term Triple Moving Average Trend Following Strategy" is a quantitative investment strategy that utilizes a combination of moving averages with different periods to capture market trends and generate trading signals. The strategy is based on a 3-day low price short-term moving average, a 3-day high price short-term moving average, and a 30-day closing price medium-term moving average. By comparing the closing price's relative position to these three moving averages, the strategy determines the trend direction and issues trading signals. When the closing price falls below the 3-day low price moving average and is above the 30-day closing price moving average, a long position is entered. When the closing price breaks above the 3-day high price moving average, the position is closed.

#### Strategy Principle
The core principle of this strategy is to utilize the trend characteristics of moving averages and the crossover relationships between different period moving averages to capture market trends. The short-term 3-day low and high price moving averages can quickly react to short-term price fluctuations, while the medium-term 30-day closing price moving average reflects the trend direction at a higher level.

When the closing price falls below the 3-day low price moving average and is above the 30-day closing price moving average, it indicates a short-term pullback but a bullish medium-term trend, signaling a long entry. When the closing price breaks above the 3-day high price moving average, the short-term upward momentum is exhausted, prompting a position exit. By combining short and medium-term moving averages, the strategy can enter at the early stage of a trend and exit before the trend ends.

#### Strategy Advantages
1. Strong trend-capturing ability. The strategy utilizes the combination of short and medium-term moving averages with different periods to effectively capture medium to long-term market trends and follow the trend.
2. Timely profit-taking. By using the 30-day medium-term moving average to determine the trend direction and the 3-day short-term moving average to timely realize profits, excessive holding is avoided.
3. Simple parameters, easy to understand and optimize. The strategy only uses three moving averages, with a clear logic and parameters that are easy to optimize and test.
4. Strong adaptability. The combination of short and medium-term moving averages can adapt to markets with different fluctuation cycles and has a certain adaptability to both trending and ranging markets.

#### Strategy Risks
1. Frequent trading. The strategy may generate frequent trading signals in ranging markets, increasing trading costs.
2. Sudden event risk. If the market experiences severe abnormal fluctuations, the moving average system may fail, causing significant drawdowns.
3. Parameter invalidation risk. If the rhythm of market trends changes, the original parameters may lose effectiveness and require re-optimization.
4. Lack of position management. The strategy does not set position management and capital management rules, limiting its risk control capability.

#### Strategy Optimization Direction
1. Add position management. Dynamic position adjustment can be implemented based on trend strength, volatility, and other indicators to improve the risk-return ratio.
2. Combine with other trend indicators. Other trend-based indicators such as MACD and DMI can be introduced as supplements to improve the accuracy of trend judgment.
3. Optimize parameters. Optimize moving average parameters for different underlying assets and timeframes to find the optimal parameter combination.
4. Incorporate stop-loss. Set reasonable stop-loss levels to control the maximum loss per trade and enhance the strategy's robustness.
5. Appropriate filtering. Reduce trading frequency in ranging markets by considering the addition of volatility filtering mechanisms such as ATR.

#### Summary
The "Short-Medium-Long Term Triple Moving Average Trend Following Strategy" is a quantitative trading strategy that captures trends using moving averages with different periods. It enters at the early stage of a trend and exits before its end by comparing the price's position relative to the 3-day low price moving average, 3-day high price moving average, and 30-day moving average. The strategy's logic is simple and easy to understand, with strong adaptability. However, it also has risks such as frequent trading and lack of position management. Future improvements can be made in areas such as position management, stop-loss and profit-taking, and parameter optimization to obtain more robust returns.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-05-05 00:00:00
end: 2024-05-10 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy(title="Estratégia de Médias Móveis - Entrada/Saída Simples", shorttitle="MM3", overlay=true)

// Parâmetros de entrada para a data de início e final do backtest
var start_date_input = input(title="Data de Início", defval=timestamp("01 Jan 2020 00:00 +0000"))
var end_date_input = input(title="Data Final", defval=timestamp("01 Jan 2021 00:00 +0000"))

// Convertendo as datas de entrada para formato de tempo
start_date = timestamp(year(start_date_input), month(start_date_input), dayofmonth(start_date_input), 0, 0)
end_date = timestamp(year(end_date_input), month(end_date_input), dayofmonth(end_date_input), 23, 59)

// Definindo as Médias Móveis
min_ma_3 = ta.sma(low, 3)
max_ma_3 = ta.sma(high, 3)
close_ma_30 = ta.sma(close, 30)

// Condição de Entrada: Fechamento abaixo da Média de 3 Mínimas e acima da Média de 30 Fechamentos
entry_condition = close < min_ma_3 and close > close_ma_30

// Condição de Saída: Fechamento acima da Média de 3 Máximas
exit_condition = close > max_ma_3

// Sinal de Compra: Entrada na próxima vela após a condição de entrada ser verdadeira
if (entry_condition )
    strategy.entry("Buy", strategy.long)

// Sinal de Venda: Saída na próxima vela após a condição de saída ser verdadeira
if (exit_condition)
    strategy.close("Buy")

// Plotando as Médias Móveis e os Sinais de Entrada/Saída
plot(min_ma_3, color=color.red, linewidth=2, title="Média de 3 Mínimas")
plot(max_ma_3, color=color.blue, linewidth=2, title="Média de 3 Máximas")
plot(close_ma_30, color=color.orange, linewidth=2, title="Média de 30 Fechamentos")

```

> Detail

https://www.fmz.com/strategy/451032

> Last Modified

2024-05-11 12:04:27
