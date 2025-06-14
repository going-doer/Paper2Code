
> Name

KNN-机器学习策略基于-K-近邻算法的趋势预测交易系统-KNN-Machine-Learning-Strategy-Trend-Prediction-Trading-System-Based-on-K-Nearest-Neighbors-Algorithm

> Author

ChaoZhang

> Strategy Description

![IMG](https://www.fmz.com/upload/asset/16a58f5895909d878ea.png)

[trans]
#### 概述
该策略采用 K 最近邻(KNN)机器学习算法对价格趋势进行预测。通过选择不同的价格计算方法(如 HL2、VWAP、SMA 等)作为输入值,以及选择不同的目标值(如价格走势、VWAP、波动率等)作为评估对象,利用 KNN 算法找出最接近当前市场状态的 K 个历史数据点,并基于这 K 个数据点的趋势方向进行多空预测。同时,策略还采用了移动平均线对预测结果进行平滑处理,以提高预测的稳定性。最后,根据预测结果进行交易决策,并以背景颜色的变化直观地展示当前的市场趋势预测。

#### 策略原理
1. 选择价格计算方法(如 HL2、VWAP、SMA 等)作为 KNN 算法的输入值。
2. 选择评估目标(如价格走势、VWAP、波动率等)作为 KNN 算法的目标值。
3. 设定最近邻数量 K 和平滑周期,以调整 KNN 算法的敏感度和预测结果的平滑程度。
4. 对于每个新的价格数据点,通过 KNN 算法在历史数据中找出最接近当前市场状态的 K 个数据点。
5. 根据这 K 个数据点的趋势方向(多头或空头)进行投票,得出当前市场的趋势预测。
6. 将预测结果通过移动平均线进行平滑处理,以提高预测的稳定性。
7. 根据平滑后的预测结果生成交易信号(多头或空头),并以背景颜色的变化直观地展示当前的市场趋势预测。

#### 优势分析
1. 采用机器学习算法,能够从历史数据中学习和预测价格趋势,具有一定的适应性和灵活性。
2. 可以通过调整输入值、目标值、最近邻数量和平滑周期等参数,来优化策略的表现和适应不同的市场状况。
3. 将预测结果与移动平均线相结合,提高了预测的稳定性和可靠性。
4. 通过背景颜色的变化直观地展示当前的市场趋势预测,便于交易者快速判断市场状态并做出决策。

#### 风险分析
1. KNN 算法的预测性能依赖于历史数据的质量和代表性,如果历史数据不足或不具代表性,可能导致预测结果不准确。
2. 策略的表现可能受到参数设置的影响,不恰当的参数组合可能导致策略表现不佳或产生过拟合。
3. 在市场趋势发生急剧变化或出现黑天鹅事件时,基于历史数据的预测可能失效,导致策略产生错误的交易信号。

#### 优化方向
1. 引入更多的技术指标或市场情绪数据作为 KNN 算法的输入,以提高预测的准确性和稳健性。
2. 采用自适应机制动态调整策略参数,以适应不同的市场状况和波动水平。
3. 结合其他技术分析方法或风险管理措施,以减少策略的风险敞口并提高收益稳定性。

#### 总结
该策略通过将 KNN 机器学习算法应用于价格趋势预测,展示了如何利用历史数据和统计方法来把握市场趋势并生成交易信号。策略的优势在于其适应性和灵活性,能够通过参数调整来优化表现并适应不同市场状况。然而,策略的风险主要来自历史数据的质量和代表性,以及参数设置的合理性。未来可以考虑引入更多指标、自适应机制和风险管理措施,以进一步提升策略的稳健性和盈利能力。

|| 

#### Overview
This strategy employs the K-Nearest Neighbors (KNN) machine learning algorithm to predict price trends. By selecting different price computation methods (such as HL2, VWAP, SMA, etc.) as input values and various target values (such as price action, VWAP, volatility, etc.) for evaluation, the KNN algorithm identifies the K historical data points closest to the current market state. The strategy then makes long or short predictions based on the trend direction of these K data points. Additionally, the strategy applies a moving average to smooth the prediction results and improve stability. Finally, trading decisions are made according to the predicted results, and the current market trend prediction is visually demonstrated through changes in the background color.

#### Strategy Principles
1. Select a price computation method (e.g., HL2, VWAP, SMA) as the input value for the KNN algorithm.
2. Choose an evaluation target (e.g., price action, VWAP, volatility) as the target value for the KNN algorithm.
3. Set the number of nearest neighbors (K) and smoothing period to adjust the sensitivity of the KNN algorithm and the smoothness of the prediction results.
4. For each new price data point, use the KNN algorithm to find the K historical data points closest to the current market state.
5. Based on the trend direction (bullish or bearish) of these K data points, vote to obtain the current market trend prediction.
6. Smooth the prediction results using a moving average to improve stability.
7. Generate trading signals (long or short) based on the smoothed prediction results and visually demonstrate the current market trend prediction through changes in the background color.

#### Advantages
1. By utilizing a machine learning algorithm, the strategy can learn from historical data and predict price trends, providing adaptability and flexibility.
2. The strategy's performance can be optimized to suit different market conditions by adjusting parameters such as input values, target values, the number of nearest neighbors, and smoothing period.
3. Combining prediction results with a moving average enhances the stability and reliability of the predictions.
4. The current market trend prediction is visually demonstrated through changes in the background color, allowing traders to quickly assess market conditions and make decisions.

#### Risks
1. The predictive performance of the KNN algorithm relies on the quality and representativeness of the historical data. Insufficient or unrepresentative data may lead to inaccurate predictions.
2. The strategy's performance may be influenced by parameter settings, and inappropriate parameter combinations can result in poor performance or overfitting.
3. When the market trend undergoes rapid changes or black swan events occur, predictions based on historical data may become ineffective, causing the strategy to generate incorrect trading signals.

#### Optimization Directions
1. Incorporate more technical indicators or market sentiment data as inputs for the KNN algorithm to improve prediction accuracy and robustness.
2. Implement an adaptive mechanism to dynamically adjust strategy parameters based on different market conditions and volatility levels.
3. Combine other technical analysis methods or risk management measures to reduce the strategy's risk exposure and enhance the stability of returns.

#### Summary
By applying the KNN machine learning algorithm to price trend prediction, this strategy demonstrates how to capture market trends and generate trading signals using historical data and statistical methods. The strategy's strengths lie in its adaptability and flexibility, as it can be optimized through parameter adjustments to suit different market conditions. However, the strategy's risks primarily stem from the quality and representativeness of historical data, as well as the reasonableness of parameter settings. Future improvements could involve incorporating more indicators, adaptive mechanisms, and risk management measures to further enhance the strategy's robustness and profitability.
[/trans]



> Source (PineScript)

``` pinescript
/*backtest
start: 2023-05-09 00:00:00
end: 2024-05-14 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_Binance","currency":"BTC_USDT"}]
*/


// This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/
// © Blake_22 {

//@version=5
strategy('money printer part 1', overlay=true)

// ~~ Tooltips {
t1 ="PriceValue selects the method of price computation. \n\nSets the smoothing period for the PriceValue. \n\nAdjusting these settings will change the input values for the K-Nearest Neighbors algorithm, influencing how the trend is calculated."
t2 = "TargetValue specifies the target to evaluate. \n\nSets the smoothing period for the TargetValue."
t3 ="numberOfClosestValues sets the number of closest values that are considered when calculating the KNN Moving Average. Adjusting this number will affect the sensitivity of the trend line, with a higher value leading to a smoother line and a lower value resulting in a line that is more responsive to recent price changes."
t4 ="smoothingPeriod sets the period for the moving average applied to the KNN classifier. Adjusting the smoothing period will affect how rapidly the trend line responds to price changes, with a larger smoothing period leading to a smoother line that may lag recent price movements, and a smaller smoothing period resulting in a line that more closely tracks recent changes."
t5 ="This option controls the background color for the trend prediction. Enabling it will change the background color based on the prediction, providing visual cues on the direction of the trend. A green color indicates a positive prediction, while red indicates a negative prediction."
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~}

// ~~ Inputs {
PriceValue   = input.string("hl2", options = ["hl2","VWAP", "sma", "wma", "ema", "hma"], group="", inline="Value")
maLen        = input.int(5, minval=2, maxval=200, title="", group="", inline="Value", tooltip=t1)
TargetValue  = input.string("Price Action", options = ["Price Action","VWAP", "Volatility", "sma", "wma", "ema", "hma"], group="", inline="Target")
maLen_       = input.int(5, minval=2, maxval=200, title="", group="", inline="Target", tooltip=t2)
// Input parameters for the KNN Moving Average
numberOfClosestValues = input.int(3, "Number of Closest Values", 2, 200, tooltip=t3) 
smoothingPeriod       = input.int(50, "Smoothing Period", 2, 500, tooltip=t4) 
windowSize            = math.max(numberOfClosestValues, 30) 

// knn Color
Upknn_col   = input.color(color.lime, title="", group="KNN Color", inline="knn col")
Dnknn_col   = input.color(color.red, title="", group="KNN Color", inline="knn col")
Neuknn_col  = input.color(color.orange, title="", group="KNN Color", inline="knn col")
// MA knn Color
Maknn_col   = input.color(color.teal, title="", group="MA KNN Color", inline="MA knn col")
// BG Color
bgcolor = input.bool(false, title="Trend Prediction Color", group="BG Color", inline="bg", tooltip=t5)
Up_col  = input.color(color.lime, title="", group="BG Color", inline="bg")
Dn_col  = input.color(color.red, title="", group="BG Color", inline="bg")
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~}

// ~~ kNN Classifier {
value_in = switch PriceValue
    "hl2"  => ta.sma(hl2,maLen)
    "VWAP" => ta.vwap(close[maLen])
    "sma" => ta.sma(close,maLen)
    "wma" => ta.wma(close,maLen)
    "ema" => ta.ema(close,maLen)
    "hma" => ta.hma(close,maLen)

meanOfKClosest(value_,target_) => 
    closestDistances = array.new_float(numberOfClosestValues, 1e10) 
    closestValues    = array.new_float(numberOfClosestValues, 0.0) 
    for i = 1 to windowSize 
        value = value_[i] 
        distance = math.abs(target_ - value) 
        maxDistIndex = 0 
        maxDistValue = closestDistances.get(0) 
        for j = 1 to numberOfClosestValues - 1 
            if closestDistances.get(j) > maxDistValue
                maxDistIndex := j
                maxDistValue := closestDistances.get(j)
        if distance < maxDistValue 
            closestDistances.set(maxDistIndex, distance)
            closestValues.set(maxDistIndex, value)
    closestValues.sum() / numberOfClosestValues 

// Choose the target input based on user selection
target_in = switch TargetValue
    "Price Action"  => ta.rma(close,maLen_) 
    "VWAP"          => ta.vwap(close[maLen_])
    "Volatility"    => ta.atr(14)
    "sma" => ta.sma(close,maLen_)
    "wma" => ta.wma(close,maLen_)
    "ema" => ta.ema(close,maLen_)
    "hma" => ta.hma(close,maLen_)

knnMA = meanOfKClosest(value_in,target_in)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~}

// ~~ kNN Prediction {
// Function to calculate KNN Classifier
price = math.avg(knnMA, close)
c     = ta.rma(knnMA[1], smoothingPeriod) 
o     = ta.rma(knnMA, smoothingPeriod)

// Defines KNN function to perform classification
knn(price) => 
    Pos_count = 0 
    Neg_count = 0 
    min_distance = 10e10 
    nearest_index = 0 
    for j = 1 to 10 
        distance = math.sqrt(math.pow(price[j] - price, 2)) 
        if distance < min_distance 
            min_distance := distance
            nearest_index := j
            Neg = c[nearest_index] > o[nearest_index] 
            Pos = c[nearest_index] < o[nearest_index] 
            if Pos 
                Pos_count += 1
            if Neg 
                Neg_count += 1
    output = Pos_count>Neg_count?1:-1 

// Calls KNN function and smooths the prediction
knn_prediction_raw = knn(price) 
knn_prediction     = ta.wma(knn_prediction_raw, 3)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~}

// ~~ Plots {
// Plots for display on the chart
knnMA_          = ta.wma(knnMA,5)
knnMA_col       = knnMA_>knnMA_[1]?Upknn_col:knnMA_<knnMA_[1]?Dnknn_col:Neuknn_col
Classifier_Line = plot(knnMA_,"Knn Classifier Line", knnMA_col)
MAknn_          = ta.rma(knnMA, smoothingPeriod)
plot(MAknn_,"Average Knn Classifier Line" ,Maknn_col) 
green = knn_prediction < 0.5
red   = knn_prediction > -0.5 
bgcolor( green and bgcolor? color.new(Dn_col,80) : 
 red and bgcolor ? color.new(Up_col,80) : na) 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~}

// ~~ Alerts {
knnMA_cross_Over_Ma      = ta.crossover(knnMA_,MAknn_)
knnMA_cross_Under_Ma     = ta.crossunder(knnMA_,MAknn_)
knnMA_cross_Over_Close   = ta.crossover(knnMA_,close)
knnMA_cross_Under_Close  = ta.crossunder(knnMA_,close)
knnMA_Switch_Up          = knnMA_[1]<knnMA_ and knnMA_[1]<=knnMA_[2]
knnMA_Switch_Dn          = knnMA_[1]>knnMA_ and knnMA_[1]>=knnMA_[2]
knnMA_Neutral            = knnMA_col==Neuknn_col and knnMA_col[1]!=Neuknn_col
greenBG                  = green and not green[1]
redBG                    = red and not red[1]

alertcondition(knnMA_cross_Over_Ma,  title = "Knn Crossover Average Knn",  message = "Knn Crossover Average Knn")
alertcondition(knnMA_cross_Under_Ma, title = "Knn Crossunder Average Knn", message = "Knn Crossunder Average Knn")
alertcondition(knnMA_cross_Over_Close,  title = "Knn Crossover Close",  message = "Knn Crossover Close")
alertcondition(knnMA_cross_Under_Close, title = "Knn Crossunder Close", message = "Knn Crossunder Close")
alertcondition(knnMA_Switch_Up,  title = "Knn Switch Up",  message = "Knn Switch Up")
alertcondition(knnMA_Switch_Dn, title = "Knn Switch Dn", message = "Knn Switch Dn")
alertcondition(knnMA_Neutral, title = "Knn is Neutral", message = "Knn is Neutral")
alertcondition(greenBG, title = "Positive Prediction", message = "Positive Prediction")
alertcondition(redBG, title = "Negative Prediction", message = "Negative Prediction")
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~}

//~~Trenddilo {



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~}



//~~ strategy { 1

LongCondtion = knnMA_[1]<knnMA_ and knnMA_[1]<=knnMA_[2]
ShortCondtion = knnMA_[1]>knnMA_ and knnMA_[1]>=knnMA_[2]


//SecondaryLongCondtion = col == color.lime
//SecondaryShortCondtion = col == color.red

strategy.entry("Long", strategy.long, when = LongCondtion)
strategy.close("Long", when =ShortCondtion)

strategy.entry("Short", strategy.short, when =ShortCondtion)
strategy.close("short", when =LongCondtion)


plot(strategy.equity, title="equity", color=color.red, linewidth=2, style=plot.style_areabr)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~}

    

```

> Detail

https://www.fmz.com/strategy/451532

> Last Modified

2024-05-15 17:09:34
