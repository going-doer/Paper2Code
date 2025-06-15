
> Name

均线突破

> Author

Crypto Joe



> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|PERIOD|120|日线就别|


> Source (javascript)

``` javascript
/*backtest
start: 2019-02-01 00:00:00
end: 2024-03-04 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Binance","currency":"BTC_USDT"}]
*/

function main(){
    var bought = false;
    var lastTime = 0;
    while (true) {
        var needCalculate = false;
        var records = exchange.GetRecords(PERIOD_D1);
        var lastK = records[records.length-1];
        if (lastTime == 0 || lastK.Time > lastTime) {
            lastTime = lastK.Time;
            // 只有在新的一天数据来的时候才需要考虑是否交易
            needCalculate = true;
        }
        if (!needCalculate) continue;
        var ma = TA.MA(records, PERIOD);
        var lastMa = ma[ma.length - 1];
        var account = exchange.GetAccount();
        var ticker = exchange.GetTicker();
        if (lastK.Close >= lastMa && !bought) {
            exchange.Buy(ticker.Last, account.Balance * 0.95 / ticker.Last);
            bought = true;
        } else if (lastK.Close < lastMa && bought) {
            exchange.Sell(ticker.Last, account.Stocks);
            bought = false;
        }
    }

}
```

> Detail

https://www.fmz.com/strategy/443682

> Last Modified

2024-03-24 13:45:19
