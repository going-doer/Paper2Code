
> Name

双均线过滤的ATR自适应趋势跟踪策略-Dual-EMA-Filtered-ATR-Adaptive-Trend-Following-Strategy

> Author

ianzeng123

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/2d82952526444e8ba0f21.png)
![IMG](https://www.fmz.com/upload/asset/2d84926837c27050efb6d.png)



[trans]  
#### 概述  
该策略结合了双均线过滤系统和ATR自适应跟踪止损机制，通过Heikin Ashi蜡烛图平滑价格波动，实现高胜率的趋势跟踪。策略核心在于利用快速EMA和慢速EMA的金叉死叉作为趋势方向过滤器，同时采用基于ATR的动态止损来保护利润。历史回测显示该策略胜率超过90%，适合中短线趋势交易。  

#### 策略原理  
1. **信号生成层**：  
   - 使用Heikin Ashi转换后的价格作为基础数据源（可切换原始价格）  
   - 计算ATR通道：通过ATR长度(20)和倍数(1.0)确定动态通道宽度  
   - 实现自适应跟踪止损：当价格突破通道时触发反向信号  

2. **趋势过滤层**：  
   - 采用双EMA系统（10周期快线/50周期慢线）  
   - 仅当快线高于慢线时允许做多，反之允许做空  

3. **风险管理层**：  
   - 动态追踪止损：通过trail_step和trail_offset参数控制止损移动粒度  
   - 固定点数止盈：take_profit_points设定绝对盈利目标  

4. **执行逻辑**：  
   - 当价格突破ATR通道且符合EMA方向时开仓  
   - 出现反向信号或触及止损/止盈时平仓  

#### 优势分析  
1. **高胜率设计**：三重过滤机制（Heikin Ashi平滑+ATR通道+EMA交叉）有效减少假信号  
2. **自适应风控**：ATR动态调整止损位置，在市场波动增大时自动扩大容错空间  
3. **趋势延续性**：EMA过滤确保只交易符合大趋势方向的机会  
4. **多时间框架兼容**：参数可调整适用于不同波动性品种  
5. **可视化辅助**：内置买卖信号标记和均线展示便于人工验证  

#### 风险分析  
1. **趋势反转风险**：在剧烈反转行情中，ATR通道可能滞后导致超额损失  
   - 优化方案：增加最大回撤硬止损  
2. **参数过拟合**：90%胜率可能在特定历史数据中优化得出  
   - 优化方案：进行多周期Walk-Forward检验  
3. **横盘磨损**：EMA交叉在震荡市中产生连续假信号  
   - 优化方案：引入ADX过滤器或波动率阈值  
4. **滑点影响**：追踪止损在快速行情中可能以不利价格执行  
   - 优化方案：设置最小滑点容忍值  

#### 优化方向  
1. **动态参数调整**：  
   - 根据市场波动率（如VIX指数）自动调节ATR倍数  
   - 实现原理：通过标准差或历史波动率百分位计算  

2. **复合过滤系统**：  
   - 加入成交量加权确认：要求突破时伴随成交量放大  
   - 增加时间过滤器：避开重要经济数据发布时间  

3. **机器学习优化**：  
   - 使用强化学习动态调整EMA周期组合  
   - 通过LSTM预测最佳止盈点位  

4. **多维度验证**：  
   - 引入周线级别趋势确认  
   - 添加RSI背离作为辅助出场信号  

#### 总结  
该策略通过Heikin Ashi-ATR-EMA三重架构实现了高概率趋势捕获，动态止损机制有效保护利润。核心优势在于将趋势方向判断（EMA）、波动率适应（ATR）和噪声过滤（Heikin Ashi）有机整合。进一步优化应着重于参数自适应性和多因子验证，建议在实际应用中配合硬性风控规则使用。  

||  

#### Overview  
This strategy combines a dual EMA filter system with ATR adaptive trailing stop mechanism, utilizing Heikin Ashi candles to smooth price fluctuations for high-probability trend following. The core logic employs fast/slow EMA crossovers as trend direction filters while implementing dynamic ATR-based stops. Historical backtests show over 90% win rate, suitable for medium-short term trend trading.  

#### Strategy Logic  
1. **Signal Generation Layer**:  
   - Uses Heikin Ashi converted price as base data source (switchable to raw price)  
   - Calculates ATR channel: Determines dynamic width via ATR length(20) and multiplier(1.0)  
   - Implements adaptive trailing stop: Triggers reversal signals when price breaks channel  

2. **Trend Filter Layer**:  
   - Dual EMA system (10-period fast / 50-period slow)  
   - Only allows long when fast EMA > slow EMA, vice versa for shorts  

3. **Risk Management Layer**:  
   - Dynamic trailing stop: Controlled by trail_step and trail_offset parameters  
   - Fixed-point take profit: take_profit_points sets absolute profit target  

4. **Execution Logic**:  
   - Enters when price breaks ATR channel with EMA confirmation  
   - Exits on reverse signals or stop/target triggers  

#### Advantages  
1. **High Win Rate Design**: Triple filtering (Heikin Ashi + ATR channel + EMA crossover) effectively reduces false signals  
2. **Adaptive Risk Control**: ATR dynamically adjusts stop levels, expanding tolerance during high volatility  
3. **Trend Continuity**: EMA filter ensures trading only in dominant trend direction  
4. **Multi-Timeframe Compatibility**: Adjustable parameters suit various instruments  
5. **Visual Assistance**: Built-in signal markers and EMA plots facilitate manual verification  

#### Risks  
1. **Trend Reversal Risk**: ATR channel may lag during sharp reversals causing excess loss  
   - Solution: Add hard maximum drawdown stop  
2. **Overfitting Risk**: 90% win rate might be optimized for specific historical data  
   - Solution: Conduct multi-period Walk-Forward testing  
3. **Range Whipsaws**: EMA crosses generate consecutive false signals in sideways markets  
   - Solution: Introduce ADX filter or volatility threshold  
4. **Slippage Impact**: Trailing stops may execute at unfavorable prices during fast markets  
   - Solution: Set minimum slippage tolerance  

#### Optimization Directions  
1. **Dynamic Parameter Adjustment**:  
   - Auto-adjust ATR multiplier based on market volatility (e.g. VIX index)  
   - Implementation: Calculate via standard deviation or historical volatility percentile  

2. **Composite Filter System**:  
   - Add volume-weighted confirmation: Require breakout with volume expansion  
   - Time filter: Avoid major economic data release times  

3. **Machine Learning Enhancement**:  
   - Use reinforcement learning to optimize EMA period combinations  
   - Predict optimal take-profit levels via LSTM  

4. **Multi-Dimensional Validation**:  
   - Incorporate weekly trend confirmation  
   - Add RSI divergence as auxiliary exit signal  

#### Conclusion  
The strategy achieves high-probability trend capture through Heikin Ashi-ATR-EMA triple architecture, with dynamic stops effectively protecting profits. Its core strength lies in the organic integration of trend direction (EMA), volatility adaptation (ATR) and noise filtration (Heikin Ashi). Further optimization should focus on parameter adaptability and multi-factor confirmation, recommending complementary hard risk rules in live trading.  
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2025-01-01 00:00:00
end: 2025-04-23 08:00:00
period: 1h
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"DOGE_USDT"}]
*/

//@version=5
strategy("UTBot + EMA Filter (HA + ATR Logic)", overlay = true, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// === INPUTS ===
bandwidth = input.float(8., 'Bandwidth')
atr_mult = input.float(1.0, 'ATR Multiplier')
atr_len = input.int(20, 'ATR Length')
ema_fast_len = input.int(10, 'EMA Fast Length')
ema_slow_len = input.int(50, 'EMA Slow Length')
use_heikin = input.bool(true, title='Use Heikin Ashi Candle')
trail_step = input.float(10.0, title='Trailing Step (Points)', minval=0.1)
trail_offset = input.float(10.0, title='Trailing Offset (Points)', minval=0.1)
take_profit_points = input.float(100.0, title='Take Profit (Points)', minval=0.1)

// === SOURCE ===
sr = use_heikin ? request.security(ticker.heikinashi(syminfo.tickerid), timeframe.period, close) : close

// === ATR Trailing Stop ===
atr = ta.atr(atr_len)
nLoss = atr_mult * atr

var float trail = na
iff_1 = sr > nz(trail[1]) ? sr - nLoss : sr + nLoss
iff_2 = sr < nz(trail[1]) and sr[1] < nz(trail[1]) ? math.min(nz(trail[1]), sr + nLoss) : iff_1
trail := sr > nz(trail[1]) and sr[1] > nz(trail[1]) ? math.max(nz(trail[1]), sr - nLoss) : iff_2

// === EMA FILTER ===
ema_fast = ta.ema(sr, ema_fast_len)
ema_slow = ta.ema(sr, ema_slow_len)

// === ENTRY & EXIT CONDITIONS ===
buy = sr[1] < trail[1] and sr > trail and ema_fast > ema_slow
sell = sr[1] > trail[1] and sr < trail and ema_fast < ema_slow

// === EXIT on opposite signal ===
exit_buy = sell
exit_sell = buy

// === STRATEGY EXECUTION ===
if buy
    strategy.entry("Buy", strategy.long)
if sell
    strategy.entry("Sell", strategy.short)

if exit_buy and strategy.position_size > 0
    strategy.close("Buy")
if exit_sell and strategy.position_size < 0
    strategy.close("Sell")

// === TRAILING STOP + TAKE PROFIT ===
// Long
if strategy.position_size > 0
    strategy.exit("Exit Long", from_entry="Buy", trail_points=trail_step, trail_offset=trail_offset, limit=sr + take_profit_points)

// Short
if strategy.position_size < 0
    strategy.exit("Exit Short", from_entry="Sell", trail_points=trail_step, trail_offset=trail_offset, limit=sr - take_profit_points)

// === PLOTS ===
plotshape(buy, title='Buy Signal', text='Buy', location=location.belowbar, color=color.green, style=shape.labelup, textcolor=color.white, size=size.tiny)
plotshape(sell, title='Sell Signal', text='Sell', location=location.abovebar, color=color.red, style=shape.labeldown, textcolor=color.white, size=size.tiny)

plot(ema_fast, color=color.teal, title='EMA Fast')
plot(ema_slow, color=color.purple, title='EMA Slow')

// === ALERTS ===
alertcondition(buy, title='UTBot Buy', message='UTBot Buy Signal')
alertcondition(sell, title='UTBot Sell', message='UTBot Sell Signal')
```

> Detail

https://www.fmz.com/strategy/492009

> Last Modified

2025-04-25 15:01:18
