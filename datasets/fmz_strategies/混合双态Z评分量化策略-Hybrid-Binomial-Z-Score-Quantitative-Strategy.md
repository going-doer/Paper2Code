
> Name

混合双态Z评分量化策略-Hybrid-Binomial-Z-Score-Quantitative-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/12be8e880763cfb4b1b.png)

[trans]
#### 概述
该策略采用了一种混合的量化分析方法,结合了二项分布模型和回归分析,以识别不同的市场状态。策略首先计算简单移动平均线(SMA)和布林带(BB)指标,然后基于历史回报的均值和标准差计算Z评分。当Z评分低于下阈值且价格低于下轨时,策略开仓做多;当Z评分高于上阈值且价格高于上轨时,策略平仓。

#### 策略原理
该策略的核心原理是利用Z评分来衡量当前回报相对于历史回报分布的位置。Z评分的计算公式为:(当前回报 - 历史回报均值) / 历史回报标准差。Z评分越高,表示当前回报越极端,超买可能性越大;Z评分越低,表示当前回报越极端,超卖可能性越大。同时,策略还结合了布林带指标,以价格突破上下轨作为二次确认。只有在Z评分和布林带信号同时满足条件时,策略才会产生交易信号。这种组合方式可以有效降低假信号的出现。

#### 策略优势
1. 量化分析:该策略完全基于量化指标,规则明确,易于实现和回测。
2. 双重确认:策略同时采用了Z评分和布林带两个指标,形成双重过滤机制,提高信号准确度。
3. 统计学基础:Z评分源于统计学中的正态分布理论,有坚实的理论基础,可以客观衡量当前回报的极端程度。
4. 参数灵活:用户可以根据需要调整SMA周期、布林带倍数、Z评分阈值等参数,灵活适应不同市场。

#### 策略风险
1. 参数敏感:不同参数设置可能导致策略表现差异较大,需要进行充分的参数优化和稳定性测试。
2. 趋势风险:当市场出现强趋势时,Z评分可能长期处于极端区域,导致策略信号稀少或完全缺失。
3. 过拟合风险:若对策略参数优化过度,可能导致过拟合,在样本外表现不佳。
4. 黑天鹅风险:极端行情下,历史统计规律可能失效,策略面临较大回撤风险。

#### 策略优化方向
1. 动态参数:考虑根据市场波动率、趋势强度等指标,动态调整Z评分阈值和布林带倍数,提高适应性。
2. 加入趋势过滤:在现有机制上叠加趋势判断指标,如MA交叉、DMI等,避免在强趋势下出现过多无效信号。
3. 组合优化:将该策略与其他量化策略(如动量、均值回归等)进行组合,发挥各自优势,提高稳健性。
4. 止损止盈:引入合理的止损止盈机制,控制单次交易风险敞口,提高风险调整后收益。

#### 总结
混合双态Z评分量化策略是一个基于统计学原理的量化交易策略,通过比较当前回报与历史回报分布,识别潜在的超买超卖机会。同时,策略采用布林带指标进行二次确认,提高信号可靠性。策略规则明确,易于实现和优化,但同时也面临参数敏感、趋势风险、过拟合风险等挑战。未来可以从动态参数、趋势过滤、组合优化、止损止盈等方面对策略进行优化,以提升其适应性和稳健性。总的来说,该策略为量化交易提供了一种简单而有效的思路,值得进一步探索和改进。

|| 

#### Overview
This strategy employs a hybrid quantitative analysis approach, combining binomial distribution model and regression analysis, to identify different market regimes. The strategy first calculates the Simple Moving Average (SMA) and Bollinger Bands (BB) indicators, then computes the Z-score based on the mean and standard deviation of historical returns. When the Z-score is below the lower threshold and the price is below the lower band, the strategy enters a long position; when the Z-score is above the upper threshold and the price is above the upper band, the strategy closes the position.

#### Strategy Principle
The core principle of this strategy is to use the Z-score to measure the position of current returns relative to the distribution of historical returns. The formula for calculating the Z-score is: (Current Return - Historical Return Mean) / Historical Return Standard Deviation. A higher Z-score indicates that the current return is more extreme and the probability of overbought is higher; a lower Z-score indicates that the current return is more extreme and the probability of oversold is higher. At the same time, the strategy also incorporates the Bollinger Bands indicator, using price breakouts above or below the bands as a secondary confirmation. The strategy generates trading signals only when both the Z-score and Bollinger Bands conditions are met simultaneously. This combination approach can effectively reduce the occurrence of false signals.

#### Strategy Advantages
1. Quantitative Analysis: The strategy is entirely based on quantitative indicators, with clear rules that are easy to implement and backtest.
2. Dual Confirmation: The strategy employs both the Z-score and Bollinger Bands indicators, forming a dual filtering mechanism to improve signal accuracy.
3. Statistical Foundation: The Z-score originates from the normal distribution theory in statistics, with a solid theoretical foundation, and can objectively measure the extremity of current returns.
4. Parameter Flexibility: Users can adjust parameters such as the SMA period, Bollinger Bands multiplier, and Z-score thresholds according to their needs, flexibly adapting to different markets.

#### Strategy Risks
1. Parameter Sensitivity: Different parameter settings may lead to significant differences in strategy performance, requiring extensive parameter optimization and stability testing.
2. Trend Risk: When the market exhibits strong trends, the Z-score may remain in extreme regions for an extended period, resulting in infrequent or completely absent strategy signals.
3. Overfitting Risk: If the strategy parameters are over-optimized, it may lead to overfitting and poor out-of-sample performance.
4. Black Swan Risk: Under extreme market conditions, historical statistical patterns may fail, exposing the strategy to significant drawdown risks.

#### Strategy Optimization Directions
1. Dynamic Parameters: Consider dynamically adjusting the Z-score thresholds and Bollinger Bands multiplier based on indicators such as market volatility and trend strength to improve adaptability.
2. Trend Filtering: Overlay trend determination indicators, such as MA crossover or DMI, on top of the existing mechanism to avoid excessive invalid signals during strong trends.
3. Portfolio Optimization: Combine this strategy with other quantitative strategies (such as momentum, mean reversion, etc.) to leverage their respective strengths and improve robustness.
4. Stop-Loss and Take-Profit: Introduce reasonable stop-loss and take-profit mechanisms to control risk exposure per trade and improve risk-adjusted returns.

#### Summary
The Hybrid Binomial Z-Score Quantitative Strategy is a quantitative trading strategy based on statistical principles, identifying potential overbought and oversold opportunities by comparing current returns with the distribution of historical returns. Additionally, the strategy employs the Bollinger Bands indicator for secondary confirmation, enhancing signal reliability. The strategy rules are clear and easy to implement and optimize, but it also faces challenges such as parameter sensitivity, trend risk, overfitting risk, etc. In the future, the strategy can be optimized in terms of dynamic parameters, trend filtering, portfolio optimization, stop-loss and take-profit mechanisms, etc., to improve its adaptability and robustness. Overall, this strategy provides a simple yet effective approach for quantitative trading, worthy of further exploration and refinement.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-05-22 00:00:00
end: 2024-05-27 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("Estratégia Híbrida Quantitativa", overlay=true)

// Definição de parâmetros
sma_length = input.int(20, title="Período da SMA")
threshold_high = input.float(1.5, title="Threshold Alto")
threshold_low = input.float(-1.5, title="Threshold Baixo")
lookback_period = input.int(252, title="Período de Retorno Histórico (dias)")

// Funções auxiliares
f_sma(source, length) =>
    ta.sma(source, length)

f_bollinger_band(source, length, mult) =>
    basis = ta.sma(source, length)
    dev = mult * ta.stdev(source, length)
    [basis + dev, basis - dev]

// Cálculo dos indicadores
sma = f_sma(close, sma_length)
[upper_band, lower_band] = f_bollinger_band(close, sma_length, 2)

// Regime de Mercado: Binomial
retornos = ta.change(close, 1)
media_retornos = ta.sma(retornos, lookback_period)
desvio_padrao_retornos = ta.stdev(retornos, lookback_period)

// Indicador de Regime: Z-Score
z_score = (retornos - media_retornos) / desvio_padrao_retornos

// Sinal de Compra e Venda
sinal_compra = z_score < threshold_low and close < lower_band
sinal_venda = z_score > threshold_high and close > upper_band

// Execução de Ordem
if (sinal_compra)
    strategy.entry("Long", strategy.long)
if (sinal_venda)
    strategy.close("Long")

// Plotagem dos Indicadores
plot(sma, title="SMA", color=color.blue)
plot(upper_band, title="Upper Bollinger Band", color=color.red)
plot(lower_band, title="Lower Bollinger Band", color=color.green)
hline(threshold_high, "Threshold Alto", color=color.red, linestyle=hline.style_dashed)
hline(threshold_low, "Threshold Baixo", color=color.green, linestyle=hline.style_dashed)
plot(z_score, title="Z-Score", color=color.purple)

```

> Detail

https://www.fmz.com/strategy/452742

> Last Modified

2024-05-28 17:38:08
