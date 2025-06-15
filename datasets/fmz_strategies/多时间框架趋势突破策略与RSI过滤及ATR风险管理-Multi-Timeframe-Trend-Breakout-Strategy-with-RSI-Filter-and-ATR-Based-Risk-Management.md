
> Name

多时间框架趋势突破策略与RSI过滤及ATR风险管理-Multi-Timeframe-Trend-Breakout-Strategy-with-RSI-Filter-and-ATR-Based-Risk-Management

> Author

ianzeng123

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/2d89859b64130b84ace89.png)
![IMG](https://www.fmz.com/upload/asset/2d8effa4a29888266ca3f.png)



[trans]
#### 概述  
该策略是一个结合趋势跟踪和突破交易的多重时间框架策略，使用EMA交叉作为趋势过滤器，RSI作为动量确认指标，ATR进行动态风险管理。策略通过分离的警报系统实现精确的入场和出场信号管理，并采用基于百分比的资金管理方法来控制风险。  

#### 策略原理  
1. **趋势判定**：使用快速EMA(9)和慢速EMA(21)的交叉关系来判定市场趋势方向。当EMA9上穿EMA21时判定为上升趋势，反之为下降趋势。  
2. **动量确认**：通过RSI指标(周期14)确认趋势强度，多头交易要求RSI>50，空头交易要求RSI<50。  
3. **突破信号**：在趋势方向确认后，价格突破前一根K线的高低点时产生交易信号。  
4. **风险管理**：使用ATR(周期14)计算动态止损位，固定风险比例为账户权益的2%。止盈设置为止损距离的3倍，并在达到50%盈利后启动追踪止损。  
5. **仓位计算**：根据止损距离和风险比例动态计算仓位大小，确保每笔交易风险一致。  

#### 优势分析  
1. **多因子验证**：结合趋势、动量和价格行为三个维度的确认，提高信号质量。  
2. **动态风险管理**：ATR-based止损能适应市场波动性变化，追踪止损保护浮动盈利。  
3. **科学的资金管理**：固定百分比风险控制避免过度交易，仓位计算精确匹配风险偏好。  
4. **清晰的视觉信号**：通过plotshape函数直观显示交易信号，便于监控。  
5. **分离警报系统**：独立的开仓/平仓警报便于实现自动化交易对接。  

#### 风险分析  
1. **震荡市场风险**：在趋势不明显的盘整行情中可能产生连续假突破信号，解决方案是增加ADX等趋势强度过滤器。  
2. **参数敏感风险**：固定参数在不同品种或市场环境下可能失效，建议进行参数优化或自适应参数设置。  
3. **跳空风险**：价格跳空可能导致滑点扩大，实际止损执行价与预期不符，解决方法是在重要数据发布前降低仓位或暂停交易。  
4. **过度拟合风险**：基于历史数据优化的参数可能在未来失效，应进行充分的前向测试。  

#### 优化方向  
1. **自适应参数**：将固定参数改为基于波动率或市场状态的自适应参数，如使用ATR百分比设置EMA周期。  
2. **复合趋势过滤**：加入更高时间框架的趋势确认，例如同时满足日线趋势和小时线信号才交易。  
3. **动态止盈**：将固定TP比例改为基于支撑阻力位或斐波那契扩展位的动态止盈。  
4. **机器学习优化**：使用强化学习动态调整RSI阈值和TP/SL比例。  
5. **事件过滤**：整合经济日历数据，在重要事件前后自动调整风险参数或暂停交易。  

#### 总结  
这是一个结构严谨的趋势跟踪策略，通过多重技术指标验证提高信号可靠性，科学的资金管理体系有效控制下行风险。策略特别适合趋势明确的市场环境，在波动性适中的品种上表现最佳。通过进一步优化参数自适应机制和增加市场状态识别模块，可以显著提升策略的稳健性和适应能力。  

||  

#### Overview  
This strategy combines trend-following and breakout trading across multiple timeframes, using EMA crossovers as trend filters, RSI for momentum confirmation, and ATR for dynamic risk management. It features a separated alert system for precise entry/exit signal management and employs percentage-based money management to control risk.  

#### Strategy Logic  
1. **Trend Identification**: Uses the crossover relationship between fast EMA(9) and slow EMA(21) to determine market trend direction.  
2. **Momentum Confirmation**: RSI indicator (period 14) confirms trend strength, requiring RSI>50 for long trades and RSI<50 for short trades.  
3. **Breakout Signals**: Generates trading signals when price breaks the previous bar's high/low points after trend confirmation.  
4. **Risk Management**: Uses ATR (period 14) to calculate dynamic stop-loss levels with fixed 2% account risk per trade. Take-profit is set at 3 times the stop-loss distance, with trailing stop activated after 50% profit.  
5. **Position Sizing**: Dynamically calculates position size based on stop-loss distance and risk percentage to ensure consistent risk exposure.  

#### Advantages  
1. **Multi-factor Verification**: Combines trend, momentum and price action confirmation for higher signal quality.  
2. **Dynamic Risk Management**: ATR-based stops adapt to market volatility changes, with trailing stops protecting floating profits.  
3. **Scientific Money Management**: Fixed-percentage risk control prevents overtrading, with precise position sizing.  
4. **Clear Visual Signals**: plotshape function provides intuitive visual cues for monitoring.  
5. **Separated Alert System**: Independent entry/exit alerts facilitate automated trading integration.  

#### Risks  
1. **Range-bound Market Risk**: May generate false breakout signals during trendless conditions. Solution: Add ADX filter.  
2. **Parameter Sensitivity**: Fixed parameters may fail in different instruments/market regimes. Solution: Parameter optimization or adaptive settings.  
3. **Gap Risk**: Price gaps may cause slippage. Solution: Reduce position before major news.  
4. **Overfitting Risk**: Historical optimization may not work forward. Solution: Robust out-of-sample testing.  

#### Optimization Directions  
1. **Adaptive Parameters**: Replace fixed parameters with volatility-based adaptive settings.  
2. **Composite Trend Filter**: Add higher timeframe trend confirmation.  
3. **Dynamic Take-profit**: Replace fixed TP ratio with support/resistance or Fibonacci extensions.  
4. **Machine Learning**: Use reinforcement learning to dynamically adjust RSI thresholds and TP/SL ratios.  
5. **Event Filtering**: Integrate economic calendar to adjust risk parameters around major events.  

#### Conclusion  
This is a well-structured trend-following strategy that improves signal reliability through multi-indicator confirmation and controls downside risk with scientific money management. It performs best in trending markets with moderate volatility. Further enhancements in parameter adaptation and market regime detection could significantly improve robustness and adaptability.  
[/trans]



> Source (PineScript)

``` pinescript
// @version=5
strategy("Trend Breakout Strategy with Separated Alerts", overlay=true, initial_capital=10, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// --- Parameters ---
var float risk_per_trade = 0.02 // 2% risk per trade
var int ema_fast = 9
var int ema_slow = 21
var int rsi_length = 14
var int atr_length = 14
var float atr_multiplier_sl = 2.0 // ATR multiplier for SL
var float tp_ratio = 3.0 // TP to SL ratio = 3:1
var float trail_trigger_ratio = 0.5 // Trailing stop triggers at 50% of TP

// --- Indicators ---
ema9 = ta.ema(close, ema_fast)
ema21 = ta.ema(close, ema_slow)
rsi = ta.rsi(close, rsi_length)
atr = ta.atr(atr_length)

// --- Trend Filter ---
bull_trend = ta.crossover(ema9, ema21) or (ema9 > ema21)
bear_trend = ta.crossunder(ema9, ema21) or (ema9 < ema21)

// --- Entry Conditions ---
long_entry = bull_trend and rsi > 50 and close > high[1]
short_entry = bear_trend and rsi < 50 and close < low[1]

// --- Position Size Calculation ---
equity = strategy.equity
stop_loss_distance = atr * atr_multiplier_sl
risk_amount = equity * risk_per_trade
position_size = risk_amount / stop_loss_distance

// --- SL and TP Levels ---
long_sl = close - stop_loss_distance
long_tp = close + stop_loss_distance * tp_ratio
short_sl = close + stop_loss_distance
short_tp = close - stop_loss_distance * tp_ratio

// --- Trailing Stop (activated after 50% of TP) ---
trail_points = atr * atr_multiplier_sl * tp_ratio * trail_trigger_ratio
trail_offset = atr * atr_multiplier_sl

// --- Entries ---
if long_entry
    strategy.entry("Long", strategy.long, qty=position_size)
    strategy.exit("Long Exit", "Long", stop=long_sl, limit=long_tp, trail_points=trail_points, trail_offset=trail_offset)

if short_entry
    strategy.entry("Short", strategy.short, qty=position_size)
    strategy.exit("Short Exit", "Short", stop=short_sl, limit=short_tp, trail_points=trail_points, trail_offset=trail_offset)

// --- Alert Conditions ---
var bool long_opened = false
var bool short_opened = false

// Track position opening
long_open_alert = long_entry and not long_opened
short_open_alert = short_entry and not short_opened

// Track position closing
long_close_alert = long_opened and strategy.position_size == 0
short_close_alert = short_opened and strategy.position_size == 0

// Update position states
if long_entry
    long_opened := true
if short_entry
    short_opened := true
if strategy.position_size == 0
    long_opened := false
    short_opened := false

// --- Alerts ---
alertcondition(long_open_alert, title="Open Long", message='{"action":"buy","symbol":"{{ticker}}","price":{{close}}}')
alertcondition(long_close_alert, title="Close Long", message='{"action":"close_long","symbol":"{{ticker}}","price":{{close}}}')
alertcondition(short_open_alert, title="Open Short", message='{"action":"sell","symbol":"{{ticker}}","price":{{close}}}')
alertcondition(short_close_alert, title="Close Short", message='{"action":"close_short","symbol":"{{ticker}}","price":{{close}}}')

// --- Visualization ---
plot(ema9, color=color.blue, title="EMA9")
plot(ema21, color=color.red, title="EMA21")
plotshape(long_open_alert, title="Long Entry", location=location.belowbar, color=color.green, style=shape.triangleup, size=size.small)
plotshape(short_open_alert, title="Short Entry", location=location.abovebar, color=color.red, style=shape.triangledown, size=size.small)

```

> Detail

https://www.fmz.com/strategy/491891

> Last Modified

2025-04-24 16:55:42
