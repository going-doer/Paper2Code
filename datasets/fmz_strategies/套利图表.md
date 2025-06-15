
> Name

套利图表

> Author

ChaoZhang



> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|SYMBOL1||交易对1|
|SYMBOL2||交易对2|
|BASESYBMOL||基准交易对|
|PERIOD||周期|


> Source (javascript)

``` javascript
// 这个chart在JavaScript语言中是对象，在使用Chart函数之前我们需要声明一个配置图表的对象变量chart
let PERIODARR = [PERIOD_M1, PERIOD_M3, PERIOD_M5, PERIOD_M15, PERIOD_M30, PERIOD_H1, PERIOD_H2, PERIOD_H4, PERIOD_H6, PERIOD_H12, PERIOD_D1, PERIOD_D3, PERIOD_W1];

/**
 * 计算简单移动平均线 (SMA)
 * @param {Array<number>} data - 输入数据数组
 * @param {number} period - 移动平均线的周期
 * @returns {Array<number>} - 包含SMA值的数组
 */
function calculateSMA(data, period) {
    let sma = [];
    let sum = 0;

    for (let i = 0; i < data.length; i++) {
        sum += data[i];

        if (i >= period - 1) {
            sma.push(sum / period);
            sum -= data[i - period + 1];
        } else {
            sma.push(null); // 如果数据点不足以计算SMA，返回null
        }
    }

    // 填充null值
    let firstNonNull = sma.find(value => value !== null);
    for (let i = 0; i < sma.length; i++) {
        if (sma[i] === null) {
            sma[i] = firstNonNull;
        }
    }

    return sma;
}

/**
 * 计算标准差 (Standard Deviation)
 * @param {Array<number>} data - 输入数据数组
 * @param {number} period - 标准差计算的周期
 * @returns {Array<number>} - 包含标准差值的数组
 */
function calculateStdev(data, period) {
    let stdev = [];

    for (let i = 0; i < data.length; i++) {
        if (i >= period - 1) {
            let subset = data.slice(i - period + 1, i + 1);
            let mean = subset.reduce((a, b) => a + b, 0) / period;
            let variance = subset.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / period;
            stdev.push(Math.sqrt(variance));
        } else {
            stdev.push(null); // 如果数据点不足以计算标准差，返回null
        }
    }

    // 填充null值
    let firstNonNull = stdev.find(value => value !== null);
    for (let i = 0; i < stdev.length; i++) {
        if (stdev[i] === null) {
            stdev[i] = firstNonNull;
        }
    }

    return stdev;
}

function main() {
    exchange.SetContractType("swap");
    // exchange.SetMaxBarLen(200);
    let chart = {
        // 该字段标记图表是否为一般图表，有兴趣的可以改成false运行看看
        __isStock: false, // 缩放工具
        tooltip: {
            xDateFormat: "%Y-%m-%d %H:%M:%S, %A",
            shared: true,
            crosshairs: true, // 启用十字线
            formatter: function () {
                let s = "<b>" + Highcharts.dateFormat("%Y-%m-%d %H:%M:%S, %A", this.x) + "</b>";

                this.points.forEach(function (point) {
                    s += "<br/>" + point.series.name + ": " + point.y.toFixed(2);
                });

                return s;
            }
        }, // 标题
        title: { text: "价差分析图" }, // 选择范围
        rangeSelector: {
            buttons: [
                { type: "hour", count: 1, text: "1h" },
                { type: "hour", count: 3, text: "3h" },
                {
                    type: "hour",
                    count: 8,
                    text: "8h"
                },
                { type: "all", text: "All" }
            ],
            selected: 0,
            inputEnabled: false
        }, // 坐标轴横轴即：x轴，当前设置的类型是：时间
        xAxis: {
            type: "datetime",
            crosshair: {
                color: "#cccccc", // 十字线颜色
                dashStyle: "shortdash", // 虚线样式
                width: 1, // 线宽
                zIndex: 5 // 确保线显示在其他元素之上
            }
        }, // 坐标轴纵轴即：y轴，默认数值随数据大小调整
        yAxis: {
            // 标题
            title: { text: "差价" }, // 是否启用右边纵轴
            opposite: false
        }, // 数据系列，该属性保存的是各个数据系列（线，K线图，标签等...）
        series: [
            // 索引为0，data数组内存放的是该索引系列的数据
            //指定颜色为blue
            { name: "spread", id: "spread", data: [], color: "blue" }, // 索引为1，设置了dashStyle:'shortdash'即：设置虚线
            { name: "basis", id: "basis", data: [], color: "orange" },
            {
                name: "upper",
                id: "upper",
                data: [],
                color: "red"
            },
            { name: "lower", id: "lower", data: [], color: "green" }
        ]
    };
    let symbol1Trades = exchange.GetRecords(SYMBOL1 , PERIODARR[PERIOD]);
    let symbol2Trades = exchange.GetRecords(SYMBOL2 , PERIODARR[PERIOD]);
    let baseTrades = exchange.GetRecords(BASESYBMOL , PERIODARR[PERIOD]);
    Log(symbol1Trades.length, symbol2Trades.length, baseTrades.length);
    Log(symbol1Trades[0], symbol2Trades[0], baseTrades[0]);
    //取到最小的长度做为循环次数
    let len = Math.min(symbol1Trades.length, symbol2Trades.length, baseTrades.length);
    let spreadRecords = [];
    //循环遍历
    for (let i = 0; i < len; i++) {
        //取到每个交易对的价格
        let symbol1Price = symbol1Trades[i].Close;
        let symbol2Price = symbol2Trades[i].Close;
        let basePrice = baseTrades[i].Close;
        //计算差价
        let diff1 = symbol1Price / basePrice;
        let diff2 = symbol2Price / basePrice;
        let spread = diff1 / diff2;
        spreadRecords.push(spread);
        let basis = calculateSMA(spreadRecords, 20);
        let calculateStdevs = calculateStdev(spreadRecords, 20);
        let dev = 2 * calculateStdevs[i];
        let upper = basis[i] + dev;
        let lower = basis[i] - dev;
        //将差价数据添加到图表中
        chart.series[0].data.push([symbol1Trades[i].Time, spread]);
        chart.series[1].data.push([symbol1Trades[i].Time, basis[i]]);
        chart.series[2].data.push([symbol1Trades[i].Time, upper]);
        chart.series[3].data.push([symbol1Trades[i].Time, lower]);
    }
    return chart;
}

```

> Detail

https://www.fmz.com/strategy/455637

> Last Modified

2024-07-11 18:33:26
