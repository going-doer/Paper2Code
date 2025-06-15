
> Name

高频翻转百分比追踪动量策略-High-Frequency-Flip-Percentage-Tracking-Momentum-Strategy

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/122c7f4588175c3de33.png)
[trans]

#### 概述

高频翻转百分比追踪动量策略是一种基于Kaufman自适应移动平均线(KAMA)的高频交易策略。该策略在1小时时间框架上使用KAMA指标作为主要参考,并在更短的时间框架(如15分钟)上进行交易。策略的核心思想是在价格穿越KAMA线时快速翻转多空头寸,同时设定1%的获利目标来锁定小额但频繁的利润。这种方法旨在利用市场的短期波动,同时通过快速止盈来控制风险。

#### 策略原理

1. 使用1小时时间框架的KAMA线作为主要趋势指标。
2. 在价格上穿KAMA线时开立多头仓位,下穿时开立空头仓位。
3. 当持有多头仓位时,如果价格下穿KAMA线,则平仓多头并开立空头;反之亦然。
4. 设定1%的获利目标。一旦达到目标,立即平仓并重置账户余额。
5. 使用账户余额的90%作为每次交易的仓位大小。
6. 在较短的时间框架(如15分钟)上执行交易,以捕捉更多的交易机会。

策略的核心在于利用KAMA线捕捉短期趋势,并通过频繁的仓位翻转来适应市场波动。1%的获利目标确保了快速锁定利润,减少了持仓时间和潜在风险。

#### 策略优势

1. 高频交易特性:策略能够捕捉短期市场波动,提高交易频率和潜在盈利机会。

2. 风险控制:通过设定1%的获利目标,策略能够快速锁定小额利润,降低单次交易的风险暴露。

3. 适应性强:KAMA指标具有自适应特性,能够在不同市场条件下调整灵敏度,提高策略的适应性。

4. 资金效率高:策略使用90%的账户余额作为仓位大小,充分利用了可用资金。

5. drawdown控制:频繁的小额获利有助于控制最大回撤,使策略更加稳定。

6. 杠杆潜力:由于drawdown较低,策略有潜力使用更高的杠杆来放大收益。

7. 全自动化:策略逻辑清晰,易于实现全自动化交易,减少人为干预。

#### 策略风险

1. 过度交易:高频翻转可能导致过度交易,增加交易成本和滑点损失。

2. 震荡市不利:在横盘震荡市场中,频繁的多空翻转可能导致连续小额亏损累积。

3. 趋势错失:1%的获利目标可能导致在强趋势市场中过早平仓,错失更大的盈利机会。

4. 假突破风险:价格在KAMA线附近频繁穿越可能触发多次假突破交易。

5. 资金管理风险:使用90%的账户余额作为仓位可能在连续亏损时迅速侵蚀资金。

6. 适用性限制:策略可能仅适用于波动性较高的市场,在低波动市场中效果欠佳。

7. 技术依赖:策略高度依赖KAMA指标,如果指标失效可能导致严重亏损。

#### 策略优化方向

1. 动态止盈:考虑将固定的1%获利目标改为基于ATR或波动率的动态止盈,以适应不同的市场条件。

2. 入场过滤:引入额外的过滤条件(如RSI、成交量)来减少假突破交易。

3. 趋势强度评估:在开仓前评估趋势强度,仅在趋势明确时进行交易,避免在震荡市频繁交易。

4. 仓位管理优化:实施更灵活的仓位管理策略,如基于账户盈亏或市场波动调整仓位大小。

5. 多时间框架分析:结合更长期的时间框架分析,提高交易方向的准确性。

6. 止损机制:引入适当的止损机制,防止单笔交易的过度亏损。

7. 参数优化:对KAMA的参数进行优化,寻找最佳的快慢线周期组合。

8. 市场适应性:开发市场状态识别机制,在不同市场条件下自动调整策略参数或暂停交易。

#### 总结

高频翻转百分比追踪动量策略是一种基于KAMA指标的创新性高频交易方法。通过快速捕捉短期趋势变化并设定固定的获利目标,该策略旨在实现频繁的小额盈利。其优势在于高适应性、低drawdown和潜在的高资金效率,但同时也面临过度交易和震荡市场风险等挑战。

通过优化入场条件、引入动态止盈和改进仓位管理等方式,该策略有潜力进一步提升其性能和稳定性。然而,交易者在使用此策略时应当充分认识其风险,并根据个人风险偏好和市场条件进行适当调整。总的来说,这是一个富有潜力的量化交易策略,特别适合那些追求高频、低风险交易机会的投资者。

||

#### Overview

The High-Frequency Flip Percentage Tracking Momentum Strategy is a high-frequency trading approach based on the Kaufman Adaptive Moving Average (KAMA). This strategy uses the KAMA indicator on a 1-hour timeframe as its primary reference while executing trades on shorter timeframes, such as 15 minutes. The core concept involves rapidly flipping between long and short positions as the price crosses the KAMA line, with a 1% profit target to secure small but frequent gains. This method aims to capitalize on short-term market fluctuations while controlling risk through quick profit-taking.

#### Strategy Principles

1. Utilize the KAMA line on a 1-hour timeframe as the main trend indicator.
2. Open a long position when the price crosses above the KAMA line, and a short position when it crosses below.
3. When holding a long position, if the price crosses below the KAMA line, close the long and open a short; vice versa for short positions.
4. Set a 1% profit target. Once reached, immediately close the position and reset the account balance.
5. Use 90% of the account balance as the position size for each trade.
6. Execute trades on a shorter timeframe (e.g., 15 minutes) to capture more trading opportunities.

The strategy's core lies in using the KAMA line to capture short-term trends and adapting to market fluctuations through frequent position flipping. The 1% profit target ensures quick profit locking, reducing holding time and potential risk.

#### Strategy Advantages

1. High-frequency trading characteristics: The strategy can capture short-term market volatility, increasing trading frequency and potential profit opportunities.

2. Risk control: By setting a 1% profit target, the strategy can quickly lock in small profits, reducing risk exposure per trade.

3. High adaptability: The KAMA indicator has adaptive features, allowing it to adjust sensitivity under different market conditions, enhancing the strategy's adaptability.

4. High capital efficiency: The strategy uses 90% of the account balance as position size, fully utilizing available funds.

5. Drawdown control: Frequent small profits help control maximum drawdown, making the strategy more stable.

6. Leverage potential: Due to lower drawdowns, the strategy has the potential to use higher leverage to amplify returns.

7. Full automation: The strategy logic is clear and easy to implement for fully automated trading, reducing human intervention.

#### Strategy Risks

1. Over-trading: High-frequency flipping may lead to excessive trading, increasing transaction costs and slippage losses.

2. Unfavorable in choppy markets: In sideways, choppy markets, frequent long-short flips may result in accumulated small losses.

3. Missing trends: The 1% profit target may cause early exits in strong trend markets, missing out on larger profit opportunities.

4. False breakout risk: Frequent price crosses around the KAMA line may trigger multiple false breakout trades.

5. Money management risk: Using 90% of the account balance as position size may rapidly erode capital during consecutive losses.

6. Limited applicability: The strategy may only be suitable for highly volatile markets, underperforming in low volatility environments.

7. Technical dependency: The strategy heavily relies on the KAMA indicator; if the indicator fails, it may lead to significant losses.

#### Strategy Optimization Directions

1. Dynamic take-profit: Consider changing the fixed 1% profit target to a dynamic take-profit based on ATR or volatility to adapt to different market conditions.

2. Entry filtering: Introduce additional filtering conditions (such as RSI, volume) to reduce false breakout trades.

3. Trend strength assessment: Evaluate trend strength before entry, only trading when trends are clear to avoid frequent trading in choppy markets.

4. Position management optimization: Implement a more flexible position sizing strategy, adjusting position size based on account performance or market volatility.

5. Multi-timeframe analysis: Incorporate analysis from longer timeframes to improve trade direction accuracy.

6. Stop-loss mechanism: Introduce appropriate stop-loss mechanisms to prevent excessive losses on individual trades.

7. Parameter optimization: Optimize KAMA parameters to find the best combination of fast and slow periods.

8. Market adaptability: Develop a market state recognition mechanism to automatically adjust strategy parameters or pause trading under different market conditions.

#### Conclusion

The High-Frequency Flip Percentage Tracking Momentum Strategy is an innovative high-frequency trading method based on the KAMA indicator. By quickly capturing short-term trend changes and setting fixed profit targets, this strategy aims to achieve frequent small profits. Its advantages lie in high adaptability, low drawdown, and potential high capital efficiency, but it also faces challenges such as over-trading and risks in choppy markets.

By optimizing entry conditions, introducing dynamic take-profits, and improving position management, this strategy has the potential to further enhance its performance and stability. However, traders should fully recognize its risks when using this strategy and make appropriate adjustments based on personal risk preferences and market conditions. Overall, this is a promising quantitative trading strategy, particularly suitable for investors seeking high-frequency, low-risk trading opportunities.

[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-07-23 00:00:00
end: 2024-07-28 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/

//@version=5
// indicator('TeeLek Flip 1 Percent', shorttitle='TeeLek Flip 1 Percent', overlay=true)
strategy("TeeLek Flip 1 Percent", shorttitle="TeeLek Flip 1 Percent", overlay=true)

// ----------------------------------------
// Input
// ----------------------------------------
BALANCE_USDT = input.float(1000, title="Start Balance (USDT)", minval=100)
PERCENT_POSITION_SIZE = input.float(90, title="Position Size (%USDT)", minval=1, maxval=100)
PERCENT_TAKE_PROFIT = input.float(10, title="Take Profit (%)", minval=0.1, maxval=100)
// KAMA Setup
KAMA_PERIOD = int(10)
KMA_FAST_LEN = input.int(5, "KMA Fast Legnth", minval=1,group="KAMA Setup")
KMA_SLOW_LEN = input.int(50, "KMA Slow Legnth", minval=1,group="KAMA Setup")

// ----------------------------------------
// Function
// ----------------------------------------
pine_kama(source) =>
    price_change = math.abs(source - source[KAMA_PERIOD])
    sum_price_change = math.sum(math.abs(source - source[1]), KAMA_PERIOD)
    fastest = 2/(KMA_FAST_LEN + 1)
    slowest = 2/(KMA_SLOW_LEN + 1)
    ER = price_change / sum_price_change
    SC =  math.pow((ER * (fastest-slowest) + slowest), 2)
    alpha = SC
    sum = 0.0
    sum := na(sum[1]) ? source : sum[1] + SC * (source - nz(sum[1]))

// ----------------------------------------
// Variable
// ----------------------------------------
var CurrentBalance_USDT = float(0)
var Accom_USDT = float(0)
var PositionSize_USDT = float(0)
var PositionSize_BTC = float(0)
var PositionTarget_USDT = float(0)
var TargetPrice = float(0)

var Long_BTC = float(0)
var Long_AvgPrice = float(0)
var Short_BTC = float(0)
var Short_AvgPrice = float(0)

var Long_Profit = float(0)
var Short_Profit = float(0)
// เริ่มต้นจากจำนวน Balanace ที่กำหนดมาให้
if CurrentBalance_USDT==0
    CurrentBalance_USDT:=BALANCE_USDT

// ----------------------------------------
// Signal
// ----------------------------------------
// kama line
kama_1h = request.security(syminfo.tickerid, "60",pine_kama(close))

// ----------------------------------------
// Strategy Preparing
// ----------------------------------------
// คำนวณ Position Size เตรียมเอาไว้
PositionSize_USDT:=CurrentBalance_USDT*PERCENT_POSITION_SIZE/100
PositionSize_BTC:=PositionSize_USDT/close
// คำนวณหามูลค่าเป้าหมาย ถ้าถึงก็จะขายเลย
PositionTarget_USDT:=CurrentBalance_USDT+(CurrentBalance_USDT*PERCENT_TAKE_PROFIT/100)

// ถ้ายังไม่ได้เปิด Order // ให้รอ ราคาตัดเส้น KAMA 1H ก่อน
if Long_BTC==0 and Short_BTC==0
    // ตัดขึ้น ให้ซื้อขึ้น Long
    if close>kama_1h and close[1]<=kama_1h[1]
        strategy.entry("L", strategy.long)
        Long_BTC:=PositionSize_BTC
        Long_AvgPrice:=close
    // ตัดลง ให้ซื้อลง  Short
    else if close<kama_1h and close[1]>=kama_1h[1]
        strategy.entry("S", strategy.short)
        Short_BTC:=PositionSize_BTC
        Short_AvgPrice:=close

// ----------------------------------------
// Strategy Switch Side
// ----------------------------------------
// ถ้าเปิด Long อยู่
if Long_BTC>0 
    // ถ้าตัดลง ให้ปิด Long แล้วซื้อลง Short
    if close<kama_1h and close[1]>=kama_1h[1]
        strategy.close_all("X")
        strategy.entry("S", strategy.short)
        Accom_USDT:=Accom_USDT+(close*Long_BTC)-(Long_AvgPrice*Long_BTC)
        Long_AvgPrice:=0
        Long_BTC:=0
        Short_AvgPrice:=close
        Short_BTC:=PositionSize_BTC
// ถ้าเปิด Short อยู่
if Short_BTC>0
    // ตัดขึ้น ให้ปิด Short แล้วซื้อขึ้น Long
    if close>kama_1h and close[1]<=kama_1h[1]
        strategy.close_all("X")
        strategy.entry("L", strategy.long)
        Accom_USDT:=Accom_USDT+(Short_AvgPrice*Short_BTC)-(close*Short_BTC)
        Short_AvgPrice:=0
        Short_BTC:=0
        Long_AvgPrice:=close
        Long_BTC:=PositionSize_BTC

// ----------------------------------------
// Strategy Take Profit
// ----------------------------------------
// ถ้าเปิด Long อยู่
if Long_BTC>0
    // คำนวณหาราคา Target price
    TargetPrice:=(PositionTarget_USDT+(Long_AvgPrice*Long_BTC)-(CurrentBalance_USDT+Accom_USDT))/Long_BTC
    // ถ้าราคามากกว่าราคาเป้าหมายก็ปิดทำกำไรได้เลย
    if close>=TargetPrice
        strategy.close_all("Take Profit")
        // เก็บกำไรเป็นทุน ไปเริ่มรอบใหม่
        CurrentBalance_USDT:=CurrentBalance_USDT+(close*Long_BTC)-(Long_AvgPrice*Long_BTC)
        Long_BTC:=0
        Long_AvgPrice:=0
        Accom_USDT:=0

// ถ้าเปิด Short อยู่
if Short_BTC>0
    // คำนวณหาราคา Target price
    TargetPrice:=((CurrentBalance_USDT+Accom_USDT)+(Short_AvgPrice*Short_BTC)-PositionTarget_USDT)/Short_BTC
    // ถ้าราคามากกว่าราคาเป้าหมายก็ปิดทำกำไรได้เลย
    if close<=TargetPrice
        strategy.close_all("Take Profit")
        // เก็บกำไรเป็นทุน ไปเริ่มรอบใหม่
        CurrentBalance_USDT:=CurrentBalance_USDT+(Short_AvgPrice*Short_BTC)-(close*Short_BTC)
        Short_BTC:=0
        Short_AvgPrice:=0
        Accom_USDT:=0

// ----------------------------------------
// Draw
// ----------------------------------------
// KAMA
plot(kama_1h,"KAMA 1H", #f18a23 , linewidth = 2)

// ----------------------------------------
// Alert
// ----------------------------------------

// ----------------------------------------
// Info Table
// ----------------------------------------

```

> Detail

https://www.fmz.com/strategy/458037

> Last Modified

2024-07-29 14:12:08
