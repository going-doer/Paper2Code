
> Name

Laguerre-RSI与ADX滤波交易信号策略-Laguerre-RSI-with-ADX-Filtered-Trading-Signals-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/c42bce585af4bffd71.png)

[trans]
#### 概述
该策略使用Laguerre RSI指标生成买卖信号,并结合ADX指标对信号进行过滤。当Laguerre RSI越过预设的买卖级别,且ADX高于设定的阈值时,策略会产生买卖信号。这种结合快速和慢速指标的方法,可以在趋势强度足够时及时捕捉交易机会,同时避免在趋势不明朗的情况下交易。

#### 策略原理
Laguerre RSI是一种动量指标,用于衡量价格变化的速度和强度。它基于Laguerre滤波器,相比传统RSI对价格变化的反应更加灵敏。策略通过比较Laguerre RSI与预设的买卖级别,产生对应的信号。

ADX指标衡量价格趋势的强度,数值越大表明趋势越强。策略通过设定ADX阈值,在趋势强度达标时开仓,而在趋势不明显时保持观望。这有助于提高信号的可靠性,避免频繁交易。

策略使用Laguerre RSI的交叉来触发买卖信号,当指标上穿买入水平时开多仓,下穿卖出水平时开空仓。同时,ADX需高于预设的阈值,以确认趋势强度。这种双重条件的设计,旨在捕捉强势趋势中的交易机会。

#### 策略优势
1. Laguerre RSI灵敏捕捉价格变化,能够及时产生交易信号。
2. ADX过滤器确保在趋势明确时交易,提高了信号的可靠性。
3. 参数可调,用户能够根据自己的偏好设置买卖级别和ADX阈值。
4. 代码简洁高效,易于理解和实现。
5. 适用于多种市场和时间框架,具有较好的通用性。

#### 策略风险
1. Laguerre RSI在震荡市会产生较多虚假信号,导致频繁交易。
2. ADX滤波可能延迟信号产生,错失部分交易机会。
3. 固定的买卖级别无法适应市场的动态变化。
4. 策略未设置止损,面临单次交易风险无法控制的问题。
5. 缺乏仓位管理和资金管理,难以控制整体风险。

#### 策略优化方向
1. 引入自适应买卖级别,根据价格波动幅度动态调整。这有助于适应不同市场状态,减少虚假信号。
2. 优化ADX过滤器,设置更加动态的阈值,在趋势初期就开始交易。这可以提早捕捉趋势,增加收益。
3. 加入止损和止盈机制,控制单次交易风险。避免持仓亏损过大,同时及时锁定利润。
4. 结合其他辅助指标,如交易量、波动率等,提高信号的可靠性。
5. 引入仓位管理和资金管理,控制总体风险敞口。根据市场趋势强度和账户净值,动态调整每次交易的资金比例。

#### 总结
Laguerre RSI结合ADX过滤的交易策略,是一种趋势跟踪方法。它利用快速指标捕捉价格变化,同时通过慢速指标确认趋势强度。这种组合可以在趋势明确时及时交易,又能在趋势不明朗时保持观望。策略优势在于逻辑简单,适用范围广,但也存在频繁交易和风险控制不足的问题。未来可以从信号优化、风控完善、仓位管理等方面对策略进行提升,以期获得更加稳健的收益。

|| 

#### Overview
This strategy generates buy and sell signals using the Laguerre RSI indicator and filters the signals using the ADX indicator. When the Laguerre RSI crosses above or below predefined buy and sell levels, and the ADX is above a set threshold, the strategy produces buy or sell signals. This approach of combining a fast and a slow indicator allows for timely capture of trading opportunities when the trend strength is sufficient while avoiding trading when the trend is unclear.

#### Strategy Principles
The Laguerre RSI is a momentum indicator used to measure the speed and strength of price changes. It is based on the Laguerre filter and is more responsive to price changes compared to the traditional RSI. The strategy generates signals by comparing the Laguerre RSI with predefined buy and sell levels.

The ADX indicator measures the strength of a price trend, with higher values indicating a stronger trend. The strategy sets an ADX threshold to enter trades only when the trend strength is sufficient and to stay on the sidelines when the trend is not clear. This helps to improve the reliability of the signals and avoid frequent trading.

The strategy uses crossovers of the Laguerre RSI to trigger buy and sell signals. It enters a long position when the indicator crosses above the buy level and a short position when it crosses below the sell level. At the same time, the ADX must be above the preset threshold to confirm the trend strength. This dual-condition design aims to capture trading opportunities in strong trends.

#### Strategy Advantages
1. The Laguerre RSI captures price changes responsively, enabling timely generation of trading signals.
2. The ADX filter ensures trading only when the trend is clear, improving the reliability of signals.
3. The parameters are adjustable, allowing users to set buy and sell levels and the ADX threshold according to their preferences.
4. The code is concise and efficient, easy to understand and implement.
5. The strategy is applicable to various markets and time frames, offering good versatility.

#### Strategy Risks
1. The Laguerre RSI may generate frequent false signals in choppy markets, leading to overtrading.
2. The ADX filter may delay signal generation, missing some trading opportunities.
3. Fixed buy and sell levels cannot adapt to dynamic changes in the market.
4. The strategy does not include stop-loss, exposing it to the risk of uncontrolled single-trade losses.
5. It lacks position sizing and money management, making it difficult to control overall risk.

#### Strategy Optimization Directions
1. Introduce adaptive buy and sell levels that adjust dynamically based on the magnitude of price fluctuations. This helps adapt to different market states and reduce false signals.
2. Optimize the ADX filter by setting more dynamic thresholds to start trading early in the trend. This can help capture trends earlier and increase profits.
3. Incorporate stop-loss and take-profit mechanisms to control single-trade risk. Avoid excessive losses on holdings while timely locking in profits.
4. Combine other auxiliary indicators, such as trading volume and volatility, to improve signal reliability.
5. Introduce position sizing and money management to control overall risk exposure. Dynamically adjust the percentage of funds for each trade based on the strength of the market trend and account equity.

#### Summary
The Laguerre RSI with ADX filtered trading strategy is a trend-following approach. It utilizes a fast indicator to capture price changes while confirming trend strength with a slow indicator. This combination allows for timely trading when the trend is clear while staying on the sidelines when the trend is uncertain. The strategy's advantages lie in its simplicity and wide applicability, but it also has issues such as frequent trading and insufficient risk control. Future enhancements can focus on signal optimization, risk management improvements, and position sizing to achieve more robust returns.

[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-05-11 00:00:00
end: 2024-05-16 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy('Laguerre RSI with Buy/Sell Signals and ADX Filter', shorttitle='LaRSI_ADX Signals', overlay=false)

// Kullanıcı girdileri
src = input(title='Source', defval=close)
alpha = input.float(title='Alpha', minval=0, maxval=1, step=0.1, defval=0.2)
buyLevel = input(20, title='Buy Level')
sellLevel = input(80, title='Sell Level')
adxLength = input(14, title='ADX Length')
adxSmoothing = input(14, title='ADX Smoothing')
adxLevel = input(20, title='ADX Level') // adxLevel tanımlamasını ekledik

// ADX hesaplaması
[diPlus, diMinus, adx] = ta.dmi(adxLength, adxSmoothing)

// Laguerre RSI hesaplamaları
gamma = 1 - alpha
L0 = 0.0
L0 := (1 - gamma) * src + gamma * nz(L0[1])
L1 = 0.0
L1 := -gamma * L0 + nz(L0[1]) + gamma * nz(L1[1])
L2 = 0.0
L2 := -gamma * L1 + nz(L1[1]) + gamma * nz(L2[1])
L3 = 0.0
L3 := -gamma * L2 + nz(L2[1]) + gamma * nz(L3[1])
cu = (L0 > L1 ? L0 - L1 : 0) + (L1 > L2 ? L1 - L2 : 0) + (L2 > L3 ? L2 - L3 : 0)
cd = (L0 < L1 ? L1 - L0 : 0) + (L1 < L2 ? L2 - L1 : 0) + (L2 < L3 ? L3 - L2 : 0)
temp = cu + cd == 0 ? -1 : cu + cd
LaRSI = temp == -1 ? 0 : cu / temp

// Alım ve satım sinyalleri
longCondition = ta.crossover(100 * LaRSI, buyLevel) and adx > adxLevel
shortCondition = ta.crossunder(100 * LaRSI, sellLevel) and adx > adxLevel

// Strateji giriş ve çıkışları
strategy.entry('Long', strategy.long, when=longCondition)
strategy.entry('Short', strategy.short, when=shortCondition)

// Göstergeleri çizme
plot(100 * LaRSI, title='LaRSI', linewidth=2, color=color.new(color.blue, 0))
hline(buyLevel, title='Buy Level', color=color.new(color.green, 0), linestyle=hline.style_dotted)
hline(sellLevel, title='Sell Level', color=color.new(color.red, 0), linestyle=hline.style_dotted)
plot(adx, title='ADX', color=color.new(color.orange, 0))

```

> Detail

https://www.fmz.com/strategy/451723

> Last Modified

2024-05-17 15:01:17
