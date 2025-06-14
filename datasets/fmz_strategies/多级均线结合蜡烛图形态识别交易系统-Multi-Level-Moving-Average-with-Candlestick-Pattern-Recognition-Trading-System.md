
> Name

多级均线结合蜡烛图形态识别交易系统-Multi-Level-Moving-Average-with-Candlestick-Pattern-Recognition-Trading-System

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/1b0b3f1b0c4343a73d5.png)

[trans]
#### 概述
该策略是一个综合性的技术分析交易系统,将经典的蜡烛图形态识别与趋势分析相结合。系统主要依托于多种经典蜡烛图形态的识别,包括十多种不同的蜡烛图形态,同时结合短期和长期移动平均线来确认市场趋势,从而产生买卖信号。该策略适用于不同的时间周期,既可用于短线交易也可用于中长期持仓。

#### 策略原理
策略采用多层次的信号确认机制:
1. 使用6周期指数移动平均线(EMA)作为短期趋势指标
2. 通过50和200周期简单移动平均线(SMA)判断长期趋势
3. 识别多种蜡烛图形态:
   - 十字星家族(普通十字星、墓碑十字、蜻蜓十字)
   - 锤子形态(锤子线、上吊线、倒锤子、流星)
   - 吞没形态
   - 孕线形态
   - 启明星/黄昏星形态
   - 三兵/三鸦形态
4. 结合趋势和形态生成交易信号

#### 策略优势
1. 信号多维度确认:通过移动平均线和蜡烛图形态的双重确认,提高交易信号的可靠性
2. 适应性强:可以适应不同市场环境,既能捕捉趋势也能把握反转机会
3. 风险控制完善:通过形态识别的严格条件,降低虚假信号
4. 操作逻辑清晰:每个交易信号都有明确的入场条件
5. 可扩展性强:策略框架容易根据需求添加新的形态识别模块

#### 策略风险
1. 形态识别延迟:蜡烛图形态确认需要多根K线完成,可能错过最佳入场时机
2. 信号重叠:多个形态同时出现可能导致信号冲突
3. 市场噪音:震荡市中可能产生过多虚假信号
4. 参数敏感性:移动平均线周期选择对策略表现影响较大
5. 计算复杂度:多个形态的实时计算可能影响执行效率

#### 策略优化方向
1. 信号权重系统:
   - 为不同形态设置可调整的权重
   - 根据市场环境动态调整权重
2. 市场环境识别:
   - 添加波动率指标以识别市场状态
   - 根据不同市场状态调整策略参数
3. 止损优化:
   - 基于形态特征设计动态止损
   - 添加跟踪止损机制
4. 信号过滤:
   - 增加成交量确认机制
   - 添加趋势强度过滤器
5. 优化计算效率:
   - 简化形态识别算法
   - 优化数据结构

#### 总结
该策略通过整合多种技术分析工具,构建了一个完整的交易系统。策略的核心优势在于多维度的信号确认机制,但同时也面临着信号延迟和过度拟合的风险。通过添加市场环境识别和动态参数调整机制,策略有望获得更好的表现。在实际应用中,建议通过回测优化参数,并结合风险管理系统使用。

|| 

#### Overview
This strategy is a comprehensive technical analysis trading system that combines classical candlestick pattern recognition with trend analysis. The system primarily relies on identifying multiple classic candlestick patterns, including more than ten different formations, while incorporating both short-term and long-term moving averages to confirm market trends and generate buy/sell signals. The strategy is adaptable to different timeframes and suitable for both short-term trading and medium to long-term position holding.

#### Strategy Principles
The strategy employs a multi-layered signal confirmation mechanism:
1. Uses 6-period Exponential Moving Average (EMA) as short-term trend indicator
2. Utilizes 50 and 200-period Simple Moving Averages (SMA) for long-term trend judgment
3. Identifies multiple candlestick patterns:
   - Doji family (regular Doji, Gravestone Doji, Dragonfly Doji)
   - Hammer patterns (Hammer, Hanging Man, Inverted Hammer, Shooting Star)
   - Absorption patterns
   - Inside patterns
   - Morning Star/Evening Star patterns
   - Three Soldiers/Three Crows patterns
4. Generates trading signals by combining trend and pattern analysis

#### Strategy Advantages
1. Multi-dimensional confirmation: Improves signal reliability through dual confirmation of moving averages and candlestick patterns
2. High adaptability: Suits different market environments, capturing both trends and reversals
3. Comprehensive risk control: Reduces false signals through strict pattern recognition criteria
4. Clear operational logic: Each trading signal has specific entry conditions
5. High scalability: Strategy framework easily accommodates new pattern recognition modules

#### Strategy Risks
1. Pattern recognition delay: Multiple candlesticks required for confirmation may miss optimal entry points
2. Signal overlap: Simultaneous patterns may cause conflicting signals
3. Market noise: May generate excessive false signals in choppy markets
4. Parameter sensitivity: Moving average period selection significantly impacts strategy performance
5. Computational complexity: Real-time calculation of multiple patterns may affect execution efficiency

#### Strategy Optimization Directions
1. Signal weighting system:
   - Implement adjustable weights for different patterns
   - Dynamically adjust weights based on market conditions
2. Market environment recognition:
   - Add volatility indicators to identify market states
   - Adjust strategy parameters based on market conditions
3. Stop-loss optimization:
   - Design dynamic stop-loss based on pattern characteristics
   - Add trailing stop mechanism
4. Signal filtering:
   - Incorporate volume confirmation mechanism
   - Add trend strength filters
5. Computational efficiency optimization:
   - Simplify pattern recognition algorithms
   - Optimize data structures

#### Summary
This strategy integrates multiple technical analysis tools to build a complete trading system. Its core strength lies in the multi-dimensional signal confirmation mechanism, though it faces challenges of signal delay and potential overfitting. The strategy's performance can be enhanced by adding market environment recognition and dynamic parameter adjustment mechanisms. In practical application, it's recommended to optimize parameters through backtesting and implement alongside a risk management system.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-11-12 00:00:00
end: 2024-11-11 00:00:00
period: 1d
basePeriod: 1d
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
strategy("hazed candles", shorttitle="hazed candles", overlay=true)

// Inputs
ema_input = input.int(6, title="EMA value to detect trend")

show_doji = input.bool(true, title="Doji star")
show_doji_grave = input.bool(true, title="Doji grave")
show_doji_dragonfly = input.bool(true, title="Doji dragonfly")
show_hammer = input.bool(true, title="Hammer")
show_hanginman = input.bool(true, title="Hanging man")
show_rhammer = input.bool(true, title="Reversed hammer")
show_falling_star = input.bool(true, title="Falling star")
show_absorption = input.bool(true, title="Absorptions")
show_tweezers = input.bool(true, title="Tweezers")
show_triple_inside = input.bool(true, title="Triple inside")
show_three_soldiers = input.bool(true, title="Three soldiers")
show_three_crows = input.bool(true, title="Three crows")
show_morning_evening_stars = input.bool(true, title="Morning / evening stars")
show_golden_death_cross = input.bool(true, title="Golden / Death cross")

// EMA calculation
prev_p_1 = ta.ema(close, ema_input)

// Variables
lowhigh_long_prop = 10
body_prop_size = 9

bar_size_h = high - close
bar_size_l = math.max(open, close) - math.min(close, open)
body_size_h = high - low

low_body_prop = close - low
high_body_prop = high - close

low_half_eq = (low_body_prop > body_size_h / 2.5 and low_body_prop < body_size_h / 1.65)
high_half_eq = (high_body_prop > body_size_h / 2.5 and high_body_prop < body_size_h / 1.65)
open_close_eq = (bar_size_l < body_size_h / body_prop_size)

///////////////// Doji star ///////////////
doji_star_up = show_doji and close <= prev_p_1 and open_close_eq and high_body_prop and low_half_eq
doji_star_down = show_doji and close > prev_p_1 and open_close_eq and high_body_prop and low_half_eq

plotshape(doji_star_up, style=shape.labelup, color=color.green, location=location.belowbar, size=size.large, text="Doji star")
plotshape(doji_star_down, style=shape.labeldown, color=color.red, location=location.abovebar, size=size.large, text="Doji star")

// Strategy entries for Doji star
if (doji_star_up)
    strategy.entry("Buy Doji Star", strategy.long)
if (doji_star_down)
    strategy.entry("Sell Doji Star", strategy.short)

///////////////// Doji grave ///////////////
long_high_body = (high_body_prop > bar_size_l * lowhigh_long_prop)
open_low_eq = ((close - low) < body_size_h / body_prop_size)

doji_grave = show_doji_grave and close > prev_p_1 and open_close_eq and open_low_eq and long_high_body
plotshape(doji_grave, style=shape.labeldown, color=color.red, location=location.abovebar, size=size.large, text="Doji grave")

// Strategy entries for Doji grave
if (doji_grave)
    strategy.entry("Sell Doji Grave", strategy.short)

///////////////// Doji dragonfly ///////////////
long_low_body = (low_body_prop > bar_size_l * lowhigh_long_prop)
open_high_eq = ((high - close) < body_size_h / body_prop_size)

doji_dragonfly = show_doji_dragonfly and close <= prev_p_1 and open_close_eq and open_high_eq and long_low_body
plotshape(doji_dragonfly, style=shape.labelup, color=color.green, location=location.belowbar, size=size.large, text="Doji dragonfly")

// Strategy entries for Doji dragonfly
if (doji_dragonfly)
    strategy.entry("Buy Doji Dragonfly", strategy.long)

///////////////// Hammer ///////////////
bottom_low = close - bar_size_h * 15
bottom_high = close - bar_size_h * 1.5
top_low = open + bar_size_l * 1.5
top_high = open + bar_size_l * 15

h_down = show_hammer and prev_p_1 > close and open == high and low > bottom_low and low < bottom_high
plotshape(h_down, style=shape.labelup, color=color.green, location=location.belowbar, size=size.large, text="Hammer")

// Strategy entries for Hammer
if (h_down)
    strategy.entry("Buy Hammer", strategy.long)

///////////////// Hanging man ///////////////
hm_down = show_hanginman and prev_p_1 < close and open == high and low > bottom_low and low < bottom_high
plotshape(hm_down, style=shape.labeldown, color=color.red, location=location.abovebar, size=size.large, text="Hanging man")

// Strategy entries for Hanging man
if (hm_down)
    strategy.entry("Sell Hanging Man", strategy.short)

///////////////// Reversed hammer ///////////////
rh_down = show_rhammer and prev_p_1 > open and low == close and high > top_low and high < top_high
plotshape(rh_down, style=shape.labelup, color=color.green, location=location.belowbar, size=size.large, text="Reversed hammer")

// Strategy entries for Reversed hammer
if (rh_down)
    strategy.entry("Buy Reversed Hammer", strategy.long)

///////////////// Fallling star ///////////////
fs_down = show_falling_star and prev_p_1 < close and low == close and high > top_low and high < top_high
plotshape(fs_down, style=shape.labeldown, color=color.red, location=location.abovebar, size=size.large, text="Falling star")

// Strategy entries for Falling star
if (fs_down)
    strategy.entry("Sell Falling Star", strategy.short)

///////////////// Absorption ///////////////
open_1 = open[1]
close_1 = close[1]
high_1 = high[1]
low_1 = low[1]

open_2 = open[2]
close_2 = close[2]
high_2 = high[2]
low_2 = low[2]

open_3 = open[3]
close_3 = close[3]
high_3 = high[3]
low_3 = low[3]

bar_1 = math.max(open_1, close_1) - math.min(open_1, close_1)
bar_2 = math.max(open_2, close_2) - math.min(open_2, close_2)
bar_3 = math.max(open_3, close_3) - math.min(open_3, close_3)
bar_h = math.max(open, close) - math.min(open, close)

bar_size_min = bar_1 * 1.2
bar_size_f = (bar_h > bar_size_min)

absorption_up = show_absorption and bar_size_f and open_1 > close_1 and open_1 != open and open_3 > open_2 and open_2 > open_1 and open_1 > open and close > open
absorption_down = show_absorption and bar_size_f and open_1 < close_1 and open_1 != open and open_3 < open_2 and open_2 < open_1 and open_1 < open and close < open

plotshape(absorption_up, style=shape.labelup, color=color.green, location=location.belowbar, size=size.large, text="Absorption")
plotshape(absorption_down, style=shape.labeldown, color=color.red, location=location.abovebar, size=size.large, text="Absorption")

// Strategy entries for Absorption
if (absorption_up)
    strategy.entry("Buy Absorption", strategy.long)
if (absorption_down)
    strategy.entry("Sell Absorption", strategy.short)

///////////////// Tweezer ///////////////
match_lows = (low_1 == low or (low_2 == low and open_2 == open_1))
sprici_up = show_tweezers and prev_p_1 > open and match_lows and open_3 > open_2 and open_2 > open_1 and open_1 > open and low != open and close_1 != low_1
plotshape(sprici_up, style=shape.labelup, color=color.green, location=location.belowbar, size=size.large, text="Tweezer")

match_highs = (high_1 == high or (high_2 == high and open_2 == open_1))
sprici_down = show_tweezers and prev_p_1 <= open and match_highs and open_3 < open_2 and open_2 < open_1 and open_1 < open and high != open and close_1 != high_1
plotshape(sprici_down, style=shape.labeldown, color=color.red, location=location.abovebar, size=size.large, text="Tweezer")

// Strategy entries for Tweezer
if (sprici_up)
    strategy.entry("Buy Tweezer", strategy.long)
if (sprici_down)
    strategy.entry("Sell Tweezer", strategy.short)

///////////////// Triple inside up/down ///////////////
open_close_min = math.min(close, open)
open_close_max = math.max(close, open)
bar = open_close_max - open_close_min
open_close_min_1 = math.min(close[1], open[1])
open_close_max_1 = math.max(close[1], open[1])
open_close_min_2 = math.min(close[2], open[2])
open_close_max_2 = math.max(close[2], open[2])

body_top_1 = math.max(close[1], open[1])
body_low_1 = math.min(close[1], open[1])

triple_inside_up = show_triple_inside and open_close_min_2 == open_close_min_1 and bar_1 > bar_2 * 0.4 and bar_1 < bar_2 * 0.6 and close > open_2 and bar > bar_1 and bar + bar_1 < bar_2 * 2
plotshape(triple_inside_up, style=shape.labelup, color=color.green, location=location.belowbar, size=size.large, text="Three inside")

triple_inside_down = show_triple_inside and open_close_max_2 == open_close_max_1 and bar_1 > bar_2 * 0.4 and bar_1 < bar_2 * 0.6 and close < open_2 and bar > bar_1 and bar + bar_1 < bar_2 * 2
plotshape(triple_inside_down, style=shape.labeldown, color=color.red, location=location.abovebar, size=size.large, text="Three inside")

// Strategy entries for Triple inside
if (triple_inside_up)
    strategy.entry("Buy Triple Inside", strategy.long)
if (triple_inside_down)
    strategy.entry("Sell Triple Inside", strategy.short)

///////////////// Triple soldiers / crows ///////////////
triple_solders = show_three_soldiers and prev_p_1 > open_2 and bar > bar_1 * 0.8 and bar < bar_1 * 1.2 and bar > bar_2 * 0.8 and bar < bar_2 * 1.2 and close > close_1 and close_1 > close_2 and open_2 < close_2 and open_1 < close_1
plotshape(triple_solders, style=shape.labelup, color=color.green, location=location.belowbar, size=size.large, text="Three soldiers")

triple_crows = show_three_crows and prev_p_1 < open_2 and bar > bar_1 * 0.8 and bar < bar_1 * 1.2 and bar > bar_2 * 0.8 and bar < bar_2 * 1.2 and close < close_1 and close_1 < close_2 and open_2 > close_2 and open_1 > close_1
plotshape(triple_crows, style=shape.labeldown, color=color.red, location=location.abovebar, size=size.large, text="Three crows")

// Strategy entries for Three soldiers and Three crows
if (triple_solders)
    strategy.entry("Buy Three Soldiers", strategy.long)
if (triple_crows)
    strategy.entry("Sell Three Crows", strategy.short)

///////////////// Golden death cross ///////////////
ma_50 = ta.sma(close, 50)
ma_200 = ta.sma(close, 200)

ma_50_200_cross = ta.crossover(ma_50, ma_200) or ta.crossunder(ma_50, ma_200)

golden_cross_up = show_golden_death_cross and ma_50_200_cross and ma_50 > ma_200
plotshape(golden_cross_up, style=shape.labelup, color=color.green, location=location.belowbar, size=size.large, text="Golden cross")

death_cross_down = show_golden_death_cross and ma_50_200_cross and ma_50 < ma_200
plotshape(death_cross_down, style=shape.labeldown, color=color.red, location=location.abovebar, size=size.large, text="Death cross")

// Strategy entries for Golden cross and Death cross
if (golden_cross_up)
    strategy.entry("Buy Golden Cross", strategy.long)
if (death_cross_down)
    strategy.entry("Sell Death Cross", strategy.short)

///////////////// Morning evening stars ///////////////
morning_star = show_morning_evening_stars and bar > bar_1 and bar_2 > bar_1 and bar > (bar_2 * 0.5) and open_close_min_2 > open_close_min_1 and open_close_min > open_close_min_1 and prev_p_1 > close_2 and prev_p_1 > close_1 and close > close_1 and close_3 > close_2 and close_2 > close_1 and close > body_top_1 and close_2 != close_1 and open != close and open_2 != close_2
plotshape(morning_star, style=shape.labelup, color=color.green, location=location.belowbar, size=size.large, text="Morning star")

evening_star = show_morning_evening_stars and bar > bar_1 and bar_2 > bar_1 and bar > (bar_2 * 0.5) and open_close_max_2 < open_close_max_1 and open_close_max < open_close_max_1 and prev_p_1 < close_2 and prev_p_1 < close_1 and close < close_1 and close_3 < close_2 and close_2 < close_1 and close < body_low_1 and close_2 != close_1 and open != close and open_2 != close_2
plotshape(evening_star, style=shape.labeldown, color=color.red, location=location.abovebar, size=size.large, text="Evening star")

// Strategy entries for Morning star and Evening star
if (morning_star)
    strategy.entry("Buy Morning Star", strategy.long)
if (evening_star)
    strategy.entry("Sell Evening Star", strategy.short)

```

> Detail

https://www.fmz.com/strategy/471720

> Last Modified

2024-11-12 16:39:22
