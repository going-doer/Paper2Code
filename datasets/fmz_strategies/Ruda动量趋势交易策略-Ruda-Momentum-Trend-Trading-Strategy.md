
> Name

Ruda动量趋势交易策略-Ruda-Momentum-Trend-Trading-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/d56483cba8a45b2649.png)
[trans]
#### 概述
Ruda动量趋势交易策略是一个基于动量和趋势指标的量化交易策略。该策略使用OBV(On Balance Volume)、EMA(Exponential Moving Average)和K线实体比例等指标来判断买入和卖出时机。当短期EMA上穿长期EMA,OBV创新高,且K线实体比例大于设定阈值时,策略会在次日开盘价买入;当价格跌破止损价或收盘价跌破短期EMA时,策略会平仓。

#### 策略原理
1. 计算两条EMA线,短期EMA参数为5,长期EMA参数为21。当短期EMA上穿长期EMA时,认为趋势向上,反之则趋势向下。
2. 计算OBV指标,当OBV创10日新高时,认为多头动能强劲。
3. 计算K线实体占比,当实体占比大于设定阈值(默认50%)时,认为趋势确立。
4. 当趋势向上、多头动能强劲且趋势确立时,策略在次日开盘价买入,止损价为当日最低价和开盘价-1%的最小值。
5. 当价格跌破止损价或收盘价跌破短期EMA时,策略平仓。

#### 优势分析
1. 结合趋势和动量指标,能够捕捉强势品种。
2. 使用次日开盘价买入和动态止损,可以避免部分假突破。
3. 止损和止盈条件明确,风险可控。

#### 风险分析
1. 趋势和动量指标存在滞后性,可能出现追高买入和止损过早的情况。
2. 参数固定,缺乏自适应性,不同市场状态下表现可能差异较大。
3. 单一市场和品种回测,策略稳定性和适用性有待进一步验证。

#### 优化方向
1. 对趋势和动量指标的参数进行优化,提高指标灵敏度和有效性。
2. 引入市场状态判断,根据当前市场特征动态调整参数。
3. 扩大回测范围,增加不同市场和品种的测试,提高策略稳健性。
4. 考虑引入仓位管理和风险控制模块,提高收益风险比。

#### 总结
Ruda动量趋势交易策略是一个简单易用的量化交易策略,通过趋势和动量指标的结合,能够捕捉强势品种和趋势机会。但该策略也存在一定局限性,如指标滞后、参数固定等问题。未来可以从优化指标参数、引入自适应机制、扩大回测范围和加强风险管理等方面对策略进行优化和改进,以提高策略的稳健性和盈利能力。

|| 

#### Overview
The Ruda Momentum Trend Trading Strategy is a quantitative trading strategy based on momentum and trend indicators. The strategy uses indicators such as OBV (On Balance Volume), EMA (Exponential Moving Average), and candlestick body ratio to determine buy and sell signals. When the short-term EMA crosses above the long-term EMA, OBV reaches a new high, and the candlestick body ratio is greater than the set threshold, the strategy buys at the next day's opening price; when the price falls below the stop-loss price or the closing price falls below the short-term EMA, the strategy closes the position.

#### Strategy Principle
1. Calculate two EMA lines with parameters of 5 for the short-term EMA and 21 for the long-term EMA. When the short-term EMA crosses above the long-term EMA, the trend is considered upward, and vice versa.
2. Calculate the OBV indicator. When OBV reaches a 10-day high, the bullish momentum is considered strong.
3. Calculate the candlestick body ratio. When the body ratio is greater than the set threshold (default 50%), the trend is considered established.
4. When the trend is upward, bullish momentum is strong, and the trend is established, the strategy buys at the next day's opening price with a stop-loss price set at the minimum of the current day's low and the opening price minus 1%.
5. When the price falls below the stop-loss price or the closing price falls below the short-term EMA, the strategy closes the position.

#### Advantage Analysis
1. By combining trend and momentum indicators, the strategy can capture strong instruments.
2. Using the next day's opening price for buying and dynamic stop-loss can avoid some false breakouts.
3. The stop-loss and take-profit conditions are clear, and the risk is controllable.

#### Risk Analysis
1. Trend and momentum indicators have a lag, which may lead to buying at high prices and premature stop-losses.
2. Fixed parameters lack adaptability, and performance may vary significantly under different market conditions.
3. Backtesting on a single market and instrument requires further verification of the strategy's stability and applicability.

#### Optimization Direction
1. Optimize the parameters of trend and momentum indicators to improve indicator sensitivity and effectiveness.
2. Introduce market state judgment and dynamically adjust parameters according to current market characteristics.
3. Expand the backtesting scope, increase testing on different markets and instruments to improve strategy robustness.
4. Consider introducing position management and risk control modules to improve the risk-reward ratio.

#### Summary
The Ruda Momentum Trend Trading Strategy is a simple and easy-to-use quantitative trading strategy that captures strong instruments and trend opportunities by combining trend and momentum indicators. However, the strategy also has certain limitations, such as indicator lag and fixed parameters. In the future, the strategy can be optimized and improved by optimizing indicator parameters, introducing adaptive mechanisms, expanding the backtesting scope, and strengthening risk management to enhance the strategy's robustness and profitability.
[/trans]

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|v_input_int_1|true|(?Backtest)Nº Anos |
|v_input_int_2|5|(?RUDA)Momentum |
|v_input_int_3|21|Trend |
|v_input_int_4|50|CORPO |


> Source (PineScript)

``` pinescript
/*backtest
start: 2024-03-01 00:00:00
end: 2024-03-31 23:59:59
period: 1h
basePeriod: 15m
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © lhcbenac

//@version=5
strategy('Ruda_Strategy', overlay=true , initial_capital=5000 , pyramiding = 3, commission_type =  strategy.commission.cash_per_contract , commission_value =  1 )

//
// 
////////////////////////////////////////////////////////
//                                                    //
//                                                    //
//                    Otimizações                     //
//                                                    //
//                                                    //
////////////////////////////////////////////////////////
//
// 

////////////////////////////////////////////////////////
//                                                    //
//                                                    //
//                 Codigo Operacional                 //
//                                                    //
//                                                    //
////////////////////////////////////////////////////////
//
//
// Indica situação de Compra ou Venda

// Condição True or False 
YEAR_BT= input.int(1,title="Nº Anos ", group = "Backtest")

INPUT_ME1 = input.int(5,title="Momentum ", group = "RUDA")
INPUT_ME2 = input.int(21,title="Trend ", group = "RUDA")
INPUT_CORPO = input.int(50,title="CORPO ", group = "RUDA")/100



v_obv = ta.obv
v_med1 = ta.ema(close , INPUT_ME1)
v_med2 = ta.ema(close , INPUT_ME2)
valid_1 = v_med1 > v_med2 
valid_2 = v_obv >= ta.highest(v_obv[1], 10)
valid_3 = math.abs(close - open) / (high-low) > INPUT_CORPO
plot(v_med1)
plot(v_med2)

compra = valid_1 and valid_2 and  strategy.position_size == 0 and valid_3


var float v_minima_ref = na

dataInicio = timestamp(year(timenow) - YEAR_BT, month(timenow), dayofmonth(timenow), 00, 00)

// Variáveis globais
var float preco_entrada = na
var float preco_stop = na

if compra and time >= dataInicio and ta.change(time("D")) != 0 and ta.change(compra)  
    v_minima_ref := low
    preco_entrada := open
    preco_stop := math.min(low, open - 0.01 * open)
    strategy.entry("Compra", strategy.long , stop = preco_stop )
    if (not na(preco_entrada) and not na(preco_stop))
        label.new(x=bar_index, y= low * 0.9, text= "Dia: " + str.tostring(dayofmonth) + "\nPreço de Entrada: " + str.tostring(preco_entrada) + "\nPreço de Stop Loss: " + str.tostring(preco_stop), style=label.style_label_up, color=color.green)

    
    
// Lógica de saída
// Saída no stop loss
if (not na(preco_stop) and low < preco_stop and ta.change(low) < 0)
    strategy.close("Compra", comment="Saída no Stop")

// Saída no lucro
if (close < v_med1 and ta.change(close) < 0)
    strategy.close("Compra", comment="Saída na Media")

venda =( (not na(preco_stop) and low < preco_stop and ta.change(low) < 0) or (close < v_med1 and ta.change(close) < 0) ) and strategy.position_size > 0
codiff = compra ? 1 : venda ? -1 : na 
plotarrow(codiff, colorup=#00c3ff, colordown=#ff0062,title="Compra", maxheight=20, offset=0)





```

> Detail

https://www.fmz.com/strategy/446965

> Last Modified

2024-04-03 15:16:47
