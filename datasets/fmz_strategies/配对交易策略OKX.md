
> Name

配对交易策略OKX

> Author

小草

> Strategy Description

策略原理：https://www.fmz.com/digest-topic/10457
源码解释：https://www.fmz.com/digest-topic/10461

需要OKX期货账户，单向持仓模式

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|Pair_A||交易币种A|
|Pair_B||交易币种B|
|Quote||基础货币|
|Pct||网格大小|
|Trade_Value||交易价值|
|Ice_Value||冰山委托价值|
|Max_Value||最大持仓|
|N||均价参数|
|Interval||休眠时间(秒)|


> Source (javascript)

``` javascript
function GetPosition(pair){
    let pos = _C(exchange.GetPosition, pair)
    if(pos.length == 0){
        return {amount:0, price:0, profit:0}
    }else if(pos.length > 1){
        throw '不支持双向持仓'
    }else{
        return {amount:pos[0].Type == 0 ? pos[0].Amount : -pos[0].Amount, price:pos[0].Price, profit:pos[0].Profit}
    }
}

function GetRatio(){
    let kline_A = exchange.GetRecords(Pair_A+"_"+Quote+".swap", 60*60, N)
    let kline_B = exchange.GetRecords(Pair_B+"_"+Quote+".swap", 60*60, N)
    let total = 0
    for(let i= Math.min(kline_A.length,kline_B.length)-1; i >= 0; i--){
        total += kline_A[i].Close / kline_B[i].Close
    }
    return total / Math.min(kline_A.length,kline_B.length)
}

function GetAccount(){
    let account = _C(exchange.GetAccount)
    let total_eq = 0
    if(exchange.GetName() == 'Futures_OKCoin'){
        total_eq = parseFloat(account.Info.data[0].totalEq)
    }else{
        total_eq = account.Balance
    }
    let init_eq = 0
    if(!_G('init_eq')){
        init_eq = total_eq
        _G('init_eq', total_eq)
    }else{
        init_eq = _G('init_eq')
    }
    LogProfit(total_eq - init_eq)
    return total_eq
}

function main(){
    var precision = exchange.GetMarkets()
    var last_get_ratio_time = Date.now()
    var ratio = GetRatio()
    var total_eq = GetAccount()
    while(true){
        let start_loop_time = Date.now()
        if(Date.now() - last_get_ratio_time > 10*60*1000){
            ratio = GetRatio()
            total_eq = GetAccount()
            last_get_ratio_time = Date.now()
        }
        let pair_a = Pair_A+"_"+Quote+".swap"
        let pair_b = Pair_B+"_"+Quote+".swap"
        let CtVal_a = "CtVal" in precision[pair_a] ? precision[pair_a].CtVal : 1
        let CtVal_b = "CtVal" in precision[pair_b] ? precision[pair_b].CtVal : 1
        let position_A = GetPosition(pair_a)
        let position_B = GetPosition(pair_b)
        let ticker_A = exchange.GetTicker(pair_a)
        let ticker_B = exchange.GetTicker(pair_b)
        if(!ticker_A || !ticker_B){
            continue
        }
        let diff = (ticker_A.Last / ticker_B.Last - ratio) / ratio
        let aim_value = - Trade_Value * diff / Pct
        let id_A = null
        let id_B = null
        if( -aim_value + position_A.amount*CtVal_a*ticker_A.Last > Trade_Value && position_A.amount*CtVal_a*ticker_A.Last > -Max_Value){
            id_A = exchange.CreateOrder(pair_a, "sell", _N(ticker_A.Buy*0.99, precision[pair_a].PricePrecision), _N(Ice_Value / (ticker_A.Buy * CtVal_a), precision[pair_a].AmountPrecision))
        }
        if( -aim_value - position_B.amount*CtVal_b*ticker_B.Last > Trade_Value && position_B.amount*CtVal_b*ticker_B.Last < Max_Value){
            id_B = exchange.CreateOrder(pair_b, "buy", _N(ticker_B.Sell*1.01, precision[pair_b].PricePrecision), _N(Ice_Value / (ticker_B.Sell * CtVal_b), precision[pair_b].AmountPrecision))
        }
        if( aim_value - position_A.amount*CtVal_a*ticker_A.Last > Trade_Value && position_A.amount*CtVal_a*ticker_A.Last < Max_Value){
            id_A = exchange.CreateOrder(pair_a, "buy", _N(ticker_A.Sell*1.01, precision[pair_a].PricePrecision), _N(Ice_Value / (ticker_A.Sell * CtVal_a), precision[pair_a].AmountPrecision))
        }
        if( aim_value + position_B.amount*CtVal_b*ticker_B.Last > Trade_Value &&  position_B.amount*CtVal_b*ticker_B.Last > -Max_Value){
            id_B = exchange.CreateOrder(pair_b, "sell", _N(ticker_B.Buy*0.99, precision[pair_b].PricePrecision), _N(Ice_Value / (ticker_B.Buy * CtVal_b), precision[pair_b].AmountPrecision))
        }
        if(id_A){
            exchange.CancelOrder(id_A)
        }
        if(id_B){
            exchange.CancelOrder(id_B)
        }
        let table = {
            type: "table",
            title: "交易信息",
            cols: ["初始权益", "当前权益", Pair_A+"仓位", Pair_B+"仓位", Pair_A+"持仓价", Pair_B+"持仓价", Pair_A+"收益", Pair_B+"收益", Pair_A+"价格", Pair_B+"价格", "当前比价", "平均比价", "偏离均价", "循环延时"],
            rows: [[_N(_G('init_eq'),2), _N(total_eq,2), _N(position_A.amount*CtVal_a*ticker_A.Last, 1), _N(position_B.amount*CtVal_b*ticker_B.Last,1),
                _N(position_A.price, precision[pair_a].PircePrecision), _N(position_B.price, precision[pair_b].PircePrecision),
                _N(position_A.profit, 1), _N(position_B.profit, 1), ticker_A.Last, ticker_B.Last,
                _N(ticker_A.Last / ticker_B.Last,6), _N(ratio, 6), _N(diff, 4), (Date.now() - start_loop_time)+"ms"
            ]]
        }
        LogStatus("`" + JSON.stringify(table) + "`")
        Sleep(Interval * 1000)
    }
}
```

> Detail

https://www.fmz.com/strategy/456143

> Last Modified

2024-07-11 11:29:05
