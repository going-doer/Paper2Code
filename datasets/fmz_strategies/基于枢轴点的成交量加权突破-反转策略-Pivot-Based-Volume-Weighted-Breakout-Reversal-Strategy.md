
> Name

基于枢轴点的成交量加权突破-反转策略-Pivot-Based-Volume-Weighted-Breakout-Reversal-Strategy

> Author

ianzeng123

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/2d87e45522d5ef9112c83.png)
![IMG](https://www.fmz.com/upload/asset/2d80d09add8daa10daa4e.png)




[trans]  
#### 概述  
该策略结合了支撑/阻力(S/R)突破/反转、成交量过滤和警报系统，旨在捕捉市场中的关键转折点。策略通过识别价格突破或反转信号，并结合异常成交量确认，以提高交易信号的可靠性。策略采用固定2%的止损和可调节的止盈比例(默认3%)来管理风险。  

#### 策略原理  
1. **支撑/阻力识别**：使用`ta.pivothigh()`和`ta.pivotlow()`函数在指定周期(pivotLen)内识别关键价格水平。当价格突破阻力位(上破1%)或从支撑位反弹(下探后收回)时触发信号。  
2. **成交量过滤**：计算成交量的SMA(volSmaLength周期)，当当前成交量超过SMA的volMultiplier倍(默认1.5倍)时视为有效确认。  
3. **多空逻辑**：  
   - **多头条件**：价格突破阻力区(close > resZone*1.01)且伴随高成交量，或价格接近支撑区(±1%范围内)出现"假跌破"(low ≤ supZone但收盘收回)且成交量放大。  
   - **空头条件**：价格跌破支撑区(close < supZone*0.99)且伴随高成交量，或价格接近阻力区(±1%范围内)出现"假突破"(high ≥ resZone但收盘回落)且成交量放大。  
4. **风险管理**：固定2%的止损和可调止盈(默认3%)通过`strategy.exit()`实现。  

#### 优势分析  
1. **多因子验证**：结合价格结构(S/R)、成交量和市场行为(假突破/假跌破)，显著降低假信号概率。  
2. **动态适应**：自动更新支撑/阻力位，适应市场变化。  
3. **风险控制严格**：固定止损防止单笔交易过度亏损，止盈比例可调以适应不同波动性市场。  
4. **可视化强**：实时绘制支撑/阻力线，交易信号标注清晰。  
5. **警报集成**：可对接自动化交易系统，适合不同交易场景。  

#### 风险分析  
1. **震荡市风险**：在无趋势市场中频繁触发假突破，导致多次止损。解决方法：增加趋势过滤指标如ADX或EMA。  
2. **参数敏感**：pivotLen和volMultiplier需根据市场调整。解决方法：进行参数优化和Walk-Forward测试。  
3. **成交量滞后**：异常成交量可能出现在价格波动之后。解决方法：结合盘口数据或缩短volSmaLength。  
4. **跳空风险**：开盘跳空可能跳过止损位。解决方法：使用限价单或避开高波动时段。  

#### 优化方向  
1. **趋势过滤**：加入ADX>25条件或200EMA方向过滤，避免逆势交易。  
2. **动态参数**：根据市场波动率(如ATR)自动调整pivotLen和volMultiplier。  
3. **分级止盈**：设置两档止盈(如2%平半仓，剩余追踪止损)，提升盈亏比。  
4. **机器学习优化**：使用历史数据训练模型优化volMultiplier和tpPerc参数。  
5. **跨周期验证**：引入更高时间框架的S/R确认，增强信号质量。  

#### 总结  
该策略通过三重验证(价格位置、成交量、价格行为)设计了一个高概率交易框架，特别适合趋势初期的捕捉。核心优势在于逻辑透明、风险可控，但需注意其在震荡市中的局限性。未来优化可聚焦于参数自适应和趋势过滤，以进一步提升稳定性。  

||  

#### Overview  
This strategy combines support/resistance (S/R) breakout/reversal, volume filtering, and alert systems to capture key market turning points. It identifies price breakout/reversal signals validated by abnormal volume activity to improve reliability. The strategy employs a fixed 2% stop loss and adjustable take profit (default 3%) for risk management.  

#### Strategy Logic  
1. **S/R Identification**: Uses `ta.pivothigh()` and `ta.pivotlow()` to detect key price levels within a specified period (pivotLen). Signals trigger when price breaks resistance (upward >1%) or bounces from support (false breakdown recovery).  
2. **Volume Filter**: Calculates volume SMA (volSmaLength periods). A valid signal requires current volume exceeding SMA by volMultiplier (default 1.5x).  
3. **Long/Short Logic**:  
   - **Long Condition**: Price breaks resistance (close > resZone*1.01) with high volume, or shows false breakdown near support (±1% range) with volume confirmation.  
   - **Short Condition**: Price breaks support (close < supZone*0.99) with high volume, or shows false breakout near resistance (±1% range) with volume confirmation.  
4. **Risk Management**: Fixed 2% stop loss and adjustable take profit (default 3%) via `strategy.exit()`.  

#### Advantages  
1. **Multi-Factor Validation**: Combines price structure (S/R), volume, and market behavior (false breaks), significantly reducing false signals.  
2. **Dynamic Adaptation**: Auto-updates S/R levels to adapt to market changes.  
3. **Strict Risk Control**: Fixed stop loss prevents excessive losses; adjustable take profit suits varying market conditions.  
4. **High Visibility**: Real-time S/R plotting and clear signal labels.  
5. **Alert Integration**: Compatible with automated trading systems.  

#### Risks  
1. **Range-Bound Risk**: Frequent false breakouts in choppy markets. Solution: Add trend filters like ADX or EMA.  
2. **Parameter Sensitivity**: pivotLen and volMultiplier require market-specific tuning. Solution: Parameter optimization and Walk-Forward testing.  
3. **Volume Lag**: Abnormal volume may follow price moves. Solution: Incorporate order book data or reduce volSmaLength.  
4. **Gap Risk**: Opening gaps may skip stop levels. Solution: Use limit orders or avoid high-volatility sessions.  

#### Optimization Directions  
1. **Trend Filtering**: Add ADX>25 or 200EMA direction filters to avoid counter-trend trades.  
2. **Dynamic Parameters**: Auto-adjust pivotLen and volMultiplier based on volatility (e.g., ATR).  
3. **Scaled Take Profit**: Implement two-tier exits (e.g., close 50% at 2%, trail remainder).  
4. **Machine Learning**: Train models to optimize volMultiplier and tpPerc historically.  
5. **Multi-Timeframe Confirmation**: Validate signals with higher timeframe S/R levels.  

#### Conclusion  
This strategy establishes a high-probability framework through triple validation (price position, volume, price action), ideal for capturing early trend phases. Its transparency and controlled risk are key strengths, though range-bound performance requires caution. Future enhancements should focus on parameter self-adaptation and trend filtering for greater robustness.  
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2024-04-24 00:00:00
end: 2024-12-31 00:00:00
period: 1d
basePeriod: 1d
exchanges: [{"eid":"Futures_Binance","currency":"DOGE_USDT"}]
*/

//@version=5
strategy("S/R Breakout/Reversal + Volume + Alerts", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// === INPUTS ===
pivotLen       = input.int(10, "Pivot Lookback for S/R")
volSmaLength   = input.int(20, "Volume SMA Length")
volMultiplier  = input.float(1.5, "Volume Multiplier")
tpPerc         = input.float(3.0, "Take Profit %", step=0.1)
slPerc         = 2.0  // Stop Loss fixed at 2%

// === S/R ZONES ===
pivotHigh = ta.pivothigh(high, pivotLen, pivotLen)
pivotLow  = ta.pivotlow(low, pivotLen, pivotLen)

var float resZone = na
var float supZone = na
if not na(pivotHigh)
    resZone := pivotHigh
if not na(pivotLow)
    supZone := pivotLow

plot(supZone, title="Support", color=color.green, linewidth=2, style=plot.style_linebr)
plot(resZone, title="Resistance", color=color.red,   linewidth=2, style=plot.style_linebr)

// === VOLUME FILTER ===
volSma     = ta.sma(volume, volSmaLength)
highVolume = volume > volSma * volMultiplier

// === LONG LOGIC ===
priceAboveRes     = close > resZone * 1.01
nearSupport       = close >= supZone * 0.99 and close <= supZone * 1.01
rejectSupport     = low <= supZone and close > supZone
longBreakoutCond  = priceAboveRes and highVolume
longReversalCond  = nearSupport and rejectSupport and highVolume
longCondition     = longBreakoutCond or longReversalCond

// === SHORT LOGIC ===
priceBelowSup     = close < supZone * 0.99
nearResistance    = close >= resZone * 0.99 and close <= resZone * 1.01
rejectResistance  = high >= resZone and close < resZone
shortBreakoutCond = priceBelowSup and highVolume
shortReversalCond = nearResistance and rejectResistance and highVolume
shortCondition    = shortBreakoutCond or shortReversalCond

// === ENTRIES WITH LABELS ===
if (longCondition)
    strategy.entry("Long", strategy.long)
    label.new(bar_index, low * 0.995, "BUY", style=label.style_label_up, color=color.green, textcolor=color.white)

if (shortCondition)
    strategy.entry("Short", strategy.short)
    label.new(bar_index, high * 1.005, "SELL", style=label.style_label_down, color=color.red, textcolor=color.white)

// === TP/SL ===
longTP  = close * (1 + tpPerc / 100)
longSL  = close * (1 - slPerc / 100)
shortTP = close * (1 - tpPerc / 100)
shortSL = close * (1 + slPerc / 100)

strategy.exit("Long TP/SL",  from_entry="Long",  limit=longTP,  stop=longSL)
strategy.exit("Short TP/SL", from_entry="Short", limit=shortTP, stop=shortSL)

// === ALERT CONDITIONS ===
alertcondition(longCondition,  title="Buy Alert",  message="? BUY signal: S/R + Volume breakout/reversal")
alertcondition(shortCondition, title="Sell Alert", message="? SELL signal: S/R + Volume breakout/reversal")

```

> Detail

https://www.fmz.com/strategy/491895

> Last Modified

2025-04-24 17:08:39
