
> Name

8小时-ema

> Author

Crypto Joe





> Source (javascript)

``` javascript
/*backtest
start: 2022-11-01 00:00:00
end: 2024-05-14 00:00:00
period: 4h
basePeriod: 15m
exchanges: [{"eid":"Binance","currency":"BTC_USDT","stocks":0}]
*/


function main(){
    let lastK;
    let bought;
    exchange.SetMaxBarLen = 1000;
    while (true) {
        const records = exchange.GetRecords(PERIOD_H1 * 8);
        // if (records.length <= 676) continue;
        // Log(records.length);
        if (!lastK || records[records.length - 1].Time != lastK.Time) {
            lastK = records[records.length - 1];
        } else {
            continue;
        }

        // records 最后一根 k 线是开盘的时候产生的 k 线，需要去掉，因为我们只关心收盘 K 线
        records.splice(-1);

        const ema169 = talib.EMA(records, 169);
        const ema12 = talib.EMA(records, 12);
        const ema676 = talib.EMA(records, 676);
        $.PlotLine('EMA169', ema169[ema169.length - 1]);
        $.PlotLine('EMA12', ema12[ema12.length - 1]);
        $.PlotLine('EMA676', ema676[ema676.length - 1]);

        const account = _C(exchange.GetAccount);
        const ticker = _C(exchange.GetTicker);

        const cross = _Cross(ema12, ema169);

        if (!bought && cross > 0) {
            exchange.Buy(ticker.Last, 0.99 * account.Balance / ticker.Last);
            bought = true;
        }

        if (bought && cross < 0) {
            exchange.Sell(ticker.Last, account.Stocks);
            bought = false;
        }

    }

}
```

> Detail

https://www.fmz.com/strategy/451516

> Last Modified

2024-05-15 21:44:49
