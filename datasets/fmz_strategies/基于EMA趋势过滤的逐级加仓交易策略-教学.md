
> Name

基于EMA趋势过滤的逐级加仓交易策略-教学

> Author

发明者量化-小小梦



> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|targetProfit|目标利润|目标利润|
|amount|下单量|下单量|
|totalEq|初始总权益|初始总权益|
|isReset|重置|重置|
|pricePrecision|价格精度|价格精度|
|amountPrecision|下单量精度|下单量精度|
|isSimulate|如果添加的交易所对象是OKEX_V5模拟盘的API KEY，勾选此选项切换为模拟盘|OKEX_V5模拟盘选项|
|ratio|加仓平仓距离比|加仓平仓距离比|
|contractType|合约代码|合约代码|
|isAmountForUSDT|按U下单，写100表示开仓头寸价值100U|按U下单|
|oneCtValue|一张合约价值，币安一张合约价值一个币，OKX ETH举例一张合约价值0.1个币。|一张合约价值|
|setAddCounter|强制设置加仓次数，用于迁移设备时恢复|强制设置加仓次数|
|reserve|保证金预留百分比|保证金预留百分比|
|marginLevel||杠杆值|
|maxAddCounter|最大加仓次数|最大加仓次数|
|isDoubling|是否翻倍加仓，勾选翻倍加仓|是否翻倍加仓|
|interval|轮询间隔|轮询间隔|
|isMaxAddCounterClear|达到最大加仓次数清仓|是否达到最大加仓次数清仓|


> Source (javascript)

``` javascript
/*backtest
start: 2024-10-01 00:00:00
end: 2025-04-23 00:00:00
period: 1h
basePeriod: 1m
exchanges: [{"eid":"Futures_Binance","currency":"ETH_USDT"}]
args: [["targetProfit",20],["amount",20],["amountPrecision",3],["isAmountForUSDT",true]]
*/

function getTotalEquity_OKEX_V5() {
    var totalEquity = null 
    var ret = exchange.IO("api", "GET", "/api/v5/account/balance", "ccy=USDT")
    if (ret) {
        try {
            totalEquity = parseFloat(ret.data[0].details[0].eq)
        } catch(e) {
            Log("获取账户总权益失败！")
            return null
        }
    }
    return totalEquity
}

function getTotalEquity_Binance() {
    var totalEquity = null 
    var ret = exchange.GetAccount()
    if (ret) {
        try {
            totalEquity = parseFloat(ret.Info.totalWalletBalance)
        } catch(e) {
            Log("获取账户总权益失败！")
            return null
        }
    }
    return totalEquity
}

function getTotalEquity() {
    var exName = exchange.GetName()
    if (exName == "Futures_OKCoin") {
        return getTotalEquity_OKEX_V5()
    } else if (exName == "Futures_Binance") {
        return getTotalEquity_Binance()
    } else {
        throw "不支持该交易所"
    }
}

function ceilToDecimals(value, decimals) {
    const factor = Math.pow(10, decimals);
    return Math.ceil(value * factor) / factor;
}

function cancelAll() {
    while (1) {
        var orders = _C(exchange.GetOrders)
        if (orders.length == 0) {
            break
        }
        for (var i = 0 ; i < orders.length ; i++) {
            exchange.CancelOrder(orders[i].Id, orders[i])
            Sleep(interval)
        }
        Sleep(interval)
    }
}

function trade(distance, price, amount) {
    var tradeFunc = null 
    if (distance == "buy") {
        tradeFunc = exchange.Buy
    } else if (distance == "sell") {
        tradeFunc = exchange.Sell
    } else if (distance == "closebuy") {
        tradeFunc = exchange.Sell
    } else {
        tradeFunc = exchange.Buy
    }
    exchange.SetDirection(distance)
    return tradeFunc(price, amount)
}

function openLong(price, amount) {
    return trade("buy", price, amount)
}

function openShort(price, amount) {
    return trade("sell", price, amount)
}

function coverLong(price, amount) {
    return trade("closebuy", price, amount)
}

function coverShort(price, amount) {
    return trade("closesell", price, amount)
}

function plotRecords(c, buyOrder, sellOrder, pos) {
    var bars = _C(exchange.GetRecords)
    if (bars.length == 0) {
        return  
    }

    bars.forEach(function(bar, index) {
        c.begin(bar)
        if (index == bars.length - 1) {
            if (buyOrder) {
                c.hline(buyOrder.Price, "buy", "rgba(255, 0, 0, 0.2)", "dotted")
            }
            if (sellOrder) {
                c.hline(sellOrder.Price, "sell", "rgba(0, 255, 0, 0.2)", "dotted")
            }
            if (pos && pos.length == 1) {
                c.hline(pos[0].Price, "pos", "rgba(0, 0, 255, 0.2)", "dashed")
            }
        }
        c.close()
    })
}

var buyOrderId = null
var sellOrderId = null
var logStatusMsgBuff = ""

function main() {
    var exName = exchange.GetName()    
    if (isSimulate && exName == "Futures_OKCoin") {
        exchange.IO("simulate", true)
    }

    if (isReset) {
        _G(null)
        LogReset(1)
        LogProfitReset()
        LogVacuum()
        Log("重置所有数据", "#FF0000")
    }

    exchange.SetContractType(contractType)
    exchange.SetPrecision(pricePrecision, amountPrecision)
    Log("设置精度", pricePrecision, amountPrecision)
    exchange.SetMarginLevel(marginLevel)

    if (totalEq == -1 && !IsVirtual()) {
        var recoverTotalEq = _G("totalEq")
        if (!recoverTotalEq) {
            var currTotalEq = getTotalEquity()
            if (currTotalEq) {
                totalEq = currTotalEq
                _G("totalEq", currTotalEq)
            } else {
                throw "获取初始权益失败"
            }
        } else {
            totalEq = recoverTotalEq
        }
    }

    var addCounter = _G("addCounter")
    if (!addCounter) {
        addCounter = 1
        if (setAddCounter != -1) {
            addCounter = setAddCounter
        }
        _G("addCounter", addCounter)
    } else {
        addCounter -= 1
    }

    let c = KLineChart({
        overlay: true
    })
    
    var isLock = false 
    while (true) {
        var ticker = _C(exchange.GetTicker)        
        var pos = _C(exchange.GetPosition)
        if (pos.length > 1) {
            Log(pos)
            throw "同时有多空持仓"
        }

        var r = _C(exchange.GetRecords, 60 * 60)
        var ema = TA.EMA(r, 60)
        if (Math.abs(ticker.Last - ema[ema.length - 2]) / ema[ema.length - 2] > 0.03) {
            cancelAll()
            isLock = true 
        }
        if (Math.abs(ticker.Last - ema[ema.length - 2]) / ema[ema.length - 2] < 0.02) {
            isLock = false 
        }
        if (isLock) {
            LogStatus(_D(), "暂停, 检测阈值：", _N(Math.abs(ticker.Last - ema[ema.length - 2]) / ema[ema.length - 2], 3), logStatusMsgBuff)
            plotRecords(c, null, null, pos)
            Sleep(interval)
            continue 
        }

        var currentAcc = _C(exchange.GetAccount)
        if (currentAcc.Balance < totalEq * reserve) {
            throw "no money, stop"
        }

        if (addCounter > maxAddCounter) {
            LogStatus(_D(), "加仓已达到上限", logStatusMsgBuff)
            if (isMaxAddCounterClear && pos.length >= 1) {
                Log("加仓已达到上限，撤单，清仓")
                cancelAll()
                if (pos[0].Type == PD_LONG) {
                    var coverId = coverLong(-1, pos[0].Amount)
                } else if (pos[0].Type == PD_SHORT) {
                    var coverId = coverShort(-1, pos[0].Amount)
                }
                addCounter = 1
            }
            continue
        }

        if (pos.length == 0) {
            if (!IsVirtual()) {
                var currTotalEq = getTotalEquity()
                if (currTotalEq) {
                    LogProfit(currTotalEq - totalEq, "当前总权益：", currTotalEq)
                }
            }

            var tradeAmountLong = amount
            var tradeAmountShort = amount
            if (isAmountForUSDT) {
                tradeAmountLong = ceilToDecimals(tradeAmountLong * 1.01 / (ticker.Last - targetProfit / 5) / oneCtValue, amountPrecision)
                tradeAmountShort = ceilToDecimals(tradeAmountShort * 1.01 / (ticker.Last + targetProfit / 5) / oneCtValue, amountPrecision)
            }

            buyOrderId = openLong(ticker.Last - targetProfit / 5, tradeAmountLong)
            sellOrderId = openShort(ticker.Last + targetProfit / 5, tradeAmountShort)

            addCounter = 1
            _G("addCounter", addCounter)
        } else if (pos[0].Type == PD_LONG) {
            var n = ratio
            var price = ticker.Last
            var addAmount = isDoubling ? pos[0].Amount : (isAmountForUSDT ? (ceilToDecimals(amount * 1.01 / (price - targetProfit * n) / oneCtValue, amountPrecision)) : amount)
            buyOrderId = openLong(price - targetProfit * n, addAmount)
            sellOrderId = coverLong(pos[0].Price + targetProfit, pos[0].Amount)

            addCounter++
            _G("addCounter", addCounter)
        } else if (pos[0].Type == PD_SHORT) {
            var n = ratio
            var price = ticker.Last
            var addAmount = isDoubling ? pos[0].Amount : (isAmountForUSDT ? (ceilToDecimals(amount * 1.01 / (price + targetProfit * n) / oneCtValue, amountPrecision)) : amount)
            buyOrderId = coverShort(pos[0].Price - targetProfit, pos[0].Amount)
            sellOrderId = openShort(price + targetProfit * n, addAmount)

            addCounter++
            _G("addCounter", addCounter)
        }

        if (!sellOrderId || !buyOrderId) {
            cancelAll()
            buyOrderId = null 
            sellOrderId = null
            continue
        } 

        while (1) { 
            var isFindBuyId = false 
            var isFindSellId = false
            var orders = _C(exchange.GetOrders)
            var buyOrder = null 
            var sellOrder = null 
            for (var i = 0 ; i < orders.length ; i++) {
                if (buyOrderId == orders[i].Id) {
                    isFindBuyId = true 
                    buyOrder = orders[i]
                }
                if (sellOrderId == orders[i].Id) {
                    isFindSellId = true 
                    sellOrder = orders[i]
                }               
            }
            if (!isFindSellId && !isFindBuyId) {    
                cancelAll()
                break
            } else if (!isFindBuyId) {   
                Log("买单成交")
                cancelAll()
                break
            } else if (!isFindSellId) {  
                Log("卖单成交")
                cancelAll()
                break
            }

            var acc = _C(exchange.GetAccount)
            var tbl = {"type": "table", "title": "data", "cols": ["data", "symbol", "type", "price", "amount"], "rows": []}
            if (buyOrder) {
                tbl.rows.push(["订单", buyOrder.Symbol, buyOrder.Type == ORDER_TYPE_BUY ? "买入" : "卖出", buyOrder.Price, buyOrder.Amount])
            }
            if (sellOrder) {
                tbl.rows.push(["订单", sellOrder.Symbol, sellOrder.Type == ORDER_TYPE_BUY ? "买入" : "卖出", sellOrder.Price, sellOrder.Amount])
            }
            if (pos && pos.length == 1) {
                tbl.rows.push(["持仓", pos[0].Symbol, pos[0].Type == PD_LONG ? "多" : "空", pos[0].Price, pos[0].Amount])
            }

            logStatusMsgBuff = "当前权益：" + acc.Equity + ", 初始权益：" + totalEq + (!IsVirtual() ? ", 浮动盈亏：" + (acc.Equity - totalEq) : "") + ", 加仓次数：" + addCounter + "\n`" + JSON.stringify(tbl) + "`"
            LogStatus(_D(), "当前权益：", acc.Equity, ", 初始权益：", totalEq, !IsVirtual() ? ", 浮动盈亏：" + (acc.Equity - totalEq) : "", ", 加仓次数：" + addCounter, 
                "\n`" + JSON.stringify(tbl) + "`")

            plotRecords(c, buyOrder, sellOrder, pos)            
            Sleep(interval)
        }
        Sleep(interval)
    }
}

function onexit() {
    Log("扫尾，撤销所有挂单")
    cancelAll()
}
```

> Detail

https://www.fmz.com/strategy/492116

> Last Modified

2025-04-27 13:32:36
