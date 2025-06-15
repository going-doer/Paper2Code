
> Name

Alligator-长期趋势跟踪交易策略-Alligator-Long-Term-Trend-Following-Trading-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/185e1ee0fa1c043947c.png)

[trans]
#### 概述
Alligator 长期趋势跟踪交易策略是一种基于 Williams Alligator 指标的量化交易策略。该策略利用不同周期的移动平均线组合来捕捉市场的主要趋势,适用于中长期的趋势跟踪交易。策略的主要思路是通过 Alligator 指标的开口方向和价格与 Alligator 指标的相对位置来判断趋势的方向和强度,从而做出交易决策。

#### 策略原理
Alligator 长期趋势跟踪交易策略使用了三条不同周期的移动平均线来构建 Alligator 指标,分别是:

1. Jaw 线:13 周期 SMMA,向未来平移 8 根 K 线
2. Teeth 线:8 周期 SMMA,向未来平移 5 根 K 线
3. Lips 线:5 周期 SMMA,向未来平移 3 根 K 线

当 Alligator 指标的开口方向向上,即 Jaw 线在最下方,Teeth 线在中间,Lips 线在最上方,同时价格在 Alligator 指标上方时,策略会开仓做多。这种情况表明一个向上的趋势浪已经确认,我们希望持有该仓位直到趋势结束。

当价格跌破 Jaw 线时,策略会平掉多单。这可以保证我们不会在熊市中继续持有头寸。

#### 策略优势
1. 适用于中长期交易:该策略基于 Alligator 指标,可以有效捕捉市场的主要趋势,非常适合中长期的趋势跟踪交易。
2. 交易频率低:策略只在确认趋势形成时开仓,趋势结束时平仓,交易频率相对较低,可以有效降低交易成本。
3. 适用范围广:该策略可以应用于各种金融市场,如外汇、加密货币等,具有很强的适应性和灵活性。
4. 无需优化参数:策略完全跟随市场趋势,无需对参数进行优化,简单易用。

#### 策略风险
1. 潜在的滑点风险:在市场剧烈波动或流动性不足的情况下,交易订单可能无法以预期价格成交,导致滑点风险。
2. 缺乏固定的风险管理:该策略没有固定的风险管理设置,需要根据自身的风险偏好来调整每笔交易的仓位大小。
3. 可能错过短期机会:由于策略专注于捕捉中长期趋势,可能会错过一些短期的交易机会。

#### 策略优化方向
1. 加入风险管理模块:可以考虑加入一些风险管理措施,如止损、动态仓位调整等,以更好地控制风险。
2. 结合其他技术指标:可以尝试将 Alligator 指标与其他技术指标相结合,如 RSI、MACD 等,以提高策略的准确性和可靠性。
3. 优化参数设置:虽然该策略无需优化参数,但可以尝试对不同的时间周期和交易标的进行回测,以找到最佳的参数组合。

#### 总结
Alligator 长期趋势跟踪交易策略是一种简单易用、适用范围广的量化交易策略。通过利用 Alligator 指标捕捉市场主要趋势,该策略可以在中长期内获得稳定的收益。虽然策略存在一些潜在的风险,但通过加入风险管理模块、结合其他技术指标以及优化参数设置等方法,可以进一步提高策略的性能和稳定性。对于偏好中长期趋势跟踪交易的投资者来说,Alligator 长期趋势跟踪交易策略是一个值得考虑的选择。

|| 

#### Overview
The Alligator Long-Term Trend Following Trading Strategy is a quantitative trading strategy based on the Williams Alligator indicator. The strategy utilizes a combination of moving averages with different periods to capture the main trends in the market, suitable for medium to long-term trend following trades. The main idea of the strategy is to determine the direction and strength of the trend by the opening direction of the Alligator indicator and the relative position of the price to the Alligator indicator, and make trading decisions accordingly.

#### Strategy Principles
The Alligator Long-Term Trend Following Trading Strategy uses three moving averages with different periods to construct the Alligator indicator, namely:

1. Jaw line: 13-period SMMA, shifted 8 bars into the future
2. Teeth line: 8-period SMMA, shifted 5 bars into the future
3. Lips line: 5-period SMMA, shifted 3 bars into the future

When the Alligator indicator's mouth opens upward, i.e., the Jaw line is at the bottom, the Teeth line is in the middle, and the Lips line is at the top, and the price is above the Alligator indicator, the strategy will open a long position. This situation indicates that an upward trend wave has been confirmed, and we want to hold the position until the trend ends.

When the price breaks below the Jaw line, the strategy will close the long position. This ensures that we will not continue to hold positions in a bear market.

#### Strategy Advantages
1. Suitable for medium to long-term trading: The strategy is based on the Alligator indicator, which can effectively capture the main trends in the market, making it very suitable for medium to long-term trend following trades.
2. Low trading frequency: The strategy only opens positions when a trend is confirmed and closes positions when the trend ends, resulting in a relatively low trading frequency, which can effectively reduce trading costs.
3. Wide range of applications: The strategy can be applied to various financial markets, such as forex, cryptocurrencies, etc., with strong adaptability and flexibility.
4. No need for parameter optimization: The strategy completely follows market trends and does not require parameter optimization, making it simple and easy to use.

#### Strategy Risks
1. Potential slippage risk: In cases of severe market fluctuations or insufficient liquidity, trading orders may not be executed at the expected price, leading to slippage risk.
2. Lack of fixed risk management: The strategy does not have fixed risk management settings, and the position size of each trade needs to be adjusted according to one's risk preference.
3. Possible missed short-term opportunities: As the strategy focuses on capturing medium to long-term trends, it may miss some short-term trading opportunities.

#### Strategy Optimization Directions
1. Add risk management module: Consider adding some risk management measures, such as stop-loss, dynamic position adjustment, etc., to better control risks.
2. Combine with other technical indicators: Try combining the Alligator indicator with other technical indicators, such as RSI, MACD, etc., to improve the accuracy and reliability of the strategy.
3. Optimize parameter settings: Although the strategy does not require parameter optimization, you can try backtesting different time periods and trading targets to find the optimal parameter combination.

#### Summary
The Alligator Long-Term Trend Following Trading Strategy is a simple, easy-to-use, and widely applicable quantitative trading strategy. By utilizing the Alligator indicator to capture the main trends in the market, the strategy can achieve stable returns over the medium to long term. Although the strategy has some potential risks, by adding risk management modules, combining with other technical indicators, and optimizing parameter settings, the strategy's performance and stability can be further improved. For investors who prefer medium to long-term trend following trades, the Alligator Long-Term Trend Following Trading Strategy is a choice worth considering.
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

//_______ <licence>
// This Pine Script™ code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © Skyrex

//_______ <version>
//@version=5

//_______ <declaration_statement>
strategy(title = "Alligator Long Term Trend Following Strategy [Skyrex.io]", 
         shorttitle = "Alligator Strategy [Skyrex.io]", 
         overlay = true, 
         format = format.inherit, 
         pyramiding = 1, 
         calc_on_order_fills = false, 
         calc_on_every_tick = true, 
         default_qty_type = strategy.percent_of_equity, 
         default_qty_value = 100, 
         initial_capital = 10000, 
         currency = currency.NONE,  
         commission_type = strategy.commission.percent, 
         commission_value = 0.1,
         slippage = 5)


//_______ <constant_declarations>
var color skyrexGreen = color.new(#2ECD99, 0)
var color skyrexGray = color.new(#F2F2F2, 0)
var color skyrexWhite = color.new(#FFFFFF, 0)

var color barcolor = na


//_______ <inputs>
// Trading bot settings
sourceUuid = input.string(title = "sourceUuid:", defval = "yourBotSourceUuid", group = "Trading Bot Settings")
secretToken = input.string(title = "secretToken:", defval = "yourBotSecretToken", group = "Trading Bot Settings")

// Trading Period Settings
lookBackPeriodStart = input(title = "Trade Start Date/Time", defval = timestamp('2023-01-01T00:00:00'), group = "Trading Period Settings")
lookBackPeriodStop = input(title = "Trade Stop Date/Time", defval = timestamp('2025-01-01T00:00:00'), group = "Trading Period Settings")

//_______ <function_declarations>
//@function       Used to calculate Simple moving average for Alligator
//@param src      Sourse for smma Calculations
//@param length   Number of bars to calculate smma
//@returns        The calculated smma value 
smma(src, length) =>
    smma =  0.0
    smma := na(smma[1]) ? ta.sma(src, length) : (smma[1] * (length - 1) + src) / length
    smma


//@function       Used to decide if current candle above the Alligator
//@param jaw      Jaw line of an Alligator
//@param teeth    Teeth line of an Alligator
//@param lips     Lips line of an Alligator
//@returns        Bool value  
is_LowAboveAlligator(jaw, teeth, lips) =>
    result = low > jaw and low > lips and low > teeth 
    result


//@function       Used to decide if current candle below the Alligator
//@param jaw      Jaw line of an Alligator
//@param teeth    Teeth line of an Alligator
//@param lips     Lips line of an Alligator
//@returns        Bool value  
is_HighBelowAlligator(jaw, teeth, lips) =>
    result = high < jaw and high < lips and high < teeth 
    result


//@function       Used to decide if Alligator's mouth is open
//@param jaw      Jaw line of an Alligator
//@param teeth    Teeth line of an Alligator
//@param lips     Lips line of an Alligator
//@returns        Bool value 
is_AlligatorHungry(jaw, teeth, lips) =>
    result = lips > jaw[5] and lips > teeth[2] and teeth > jaw[3]
    result


//_______ <calculations>
jaw = smma(hl2, 13)[8]
teeth = smma(hl2, 8)[5]
lips = smma(hl2, 5)[3]


jaw_o = smma(hl2, 13)
teeth_o = smma(hl2, 8)
lips_o = smma(hl2, 5)


//_______ <strategy_calls>
longCondition = is_LowAboveAlligator(jaw, teeth, lips) and is_AlligatorHungry(jaw_o, teeth_o, lips_o) 
if (longCondition)
    strategy.entry(id = "entry1", direction = strategy.long, alert_message = '{\n"base": "' + syminfo.basecurrency + '",\n"quote": "' + syminfo.currency + '",\n"position": "entry1",\n"price": "' + str.tostring(close) + '",\n"sourceUuid": "' + sourceUuid + '",\n"secretToken": "' + secretToken + '",\n"timestamp": "' + str.tostring(timenow) + '"\n}')

if close < jaw
    strategy.close(id = "entry1", alert_message = '{\n"base": "' + syminfo.basecurrency + '",\n"quote": "' + syminfo.currency + '",\n"position": "close",\n"price": "' + str.tostring(close) + '",\n"sourceUuid": "' + sourceUuid + '",\n"secretToken": "' + secretToken + '",\n"timestamp": "' + str.tostring(timenow) + '"\n}')



//_______ <visuals>
if strategy.opentrades > 0
    barcolor := skyrexGreen
else 
    barcolor := skyrexGray

barcolor(barcolor)
//_______ <alerts>
```

> Detail

https://www.fmz.com/strategy/451731

> Last Modified

2024-05-17 15:40:13
