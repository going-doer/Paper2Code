
> Name

多平台期货资金费率获取与监控策略multiple-thread-get-fundings-symbols

> Author

发明者量化

> Strategy Description

[trans]

# 多平台期货资金费率获取与监控策略

### 描述：
该策略用于从多个期货平台（如OKCoin、Binance、Bitget等）获取和监控资金费率。通过并行线程轮询各交易所的永续合约市场，并获取资金费率数据，同时使用延迟机制优化请求频率。

### 功能：
- **多平台支持**：同步多个交易平台的资金费率，并为每个平台设定不同的请求延迟。
- **特定符号获取**：支持获取特定交易对（如BTC/USDT, ETH/USDT）的资金费率。
- **针对不同平台优化**：区分无需逐个市场查询的平台（如Binance）和需要遍历所有市场的平台（如OKCoin）。

### 使用方法：
您可以根据需要调整平台列表、符号列表和轮询间隔，以满足特定的交易需求。

||

# Multi-Platform Futures Funding Rate Retrieval and Monitoring Strategy

### Description:
This strategy retrieves and monitors funding rates across multiple futures platforms (e.g., OKCoin, Binance, Bitget). It uses parallel threads to poll the perpetual contract markets on various exchanges and retrieve funding rate data, with a delay mechanism to optimize request frequency.

### Features:
- **Multi-Platform Support**: Synchronizes funding rates from several trading platforms, each with customized request delays.
- **Symbol Specific Retrieval**: Allows retrieval of funding rates for specific trading pairs (e.g., BTC/USDT, ETH/USDT).
- **Optimized for Different Platforms**: Distinguishes between platforms that require iteration over individual markets (e.g., OKCoin) and those that do not (e.g., Binance).

### Usage:
You can adjust the platform list, symbol list, and polling intervals to suit your specific trading requirements.

[/trans]



> Source (javascript)

``` javascript

function startFundingWorker() {
    exchanges.forEach((_, pos) => {
        __Thread(function (pos) {
            let e = exchanges[pos]
            let eName = e.GetName()
            let delaySettings = {
                'Futures_OKCoin': 20,
                'Futures_Binance': 500,
            }
            let needInterate = ['Futures_OKCoin', 'Futures_Bitget','Futures_OKX', 'Futures_KuCoin', 'Futures_MEXC', 'Futures_Crypto', 'Futures_Deribit']
            let delay = function () {
                let n = delaySettings[eName]
                if (n) {
                    Sleep(n)
                }
            }
            let epoch = 60000*2;
            let ts = 0;
            let fundings = {}
            while (true) {
                let now = new Date().getTime()
                if (now - ts < epoch) {
                    Sleep(1000)
                    continue
                }
                let markets = e.GetMarkets()
                if (!markets) {
                    Sleep(1000)
                    continue
                }
                if (needInterate.includes(eName)) {
                    for (let symbol in markets) {
                        if (symbol.includes('.swap')) {
                            let ret = e.GetFundings(symbol)
                            if (ret) {
                                for (let r of ret) {
                                    fundings[r.Symbol] = r
                                }
                            }
                            delay();
                        }
                    }
                } else {
                    let zones = []
                    for (let symbol in markets) {
                        if (symbol.includes('.swap') && !zones.includes(markets[symbol].QuoteAsset)) {
                            zones.push(markets[symbol].QuoteAsset)
                            let ret = e.GetFundings(markets[symbol].QuoteAsset + '.swap')
                            if (ret) {
                                for (let r of ret) {
                                    fundings[r.Symbol] = r
                                }
                            }
                            delay();
                        }
                    }
                }
                ts = now
                __threadSetData(0, eName+"_funding", fundings)
            }
        }, pos)
    })
}

function getFundings(eName, symbols) {
    let fundings = __threadGetData(0, eName+"_funding")
    if (!fundings) {
        return null
    }

    if (typeof(symbols) === 'undefined') {
        return fundings
    }

    let ret = {}
    symbols.forEach((s) => {
        if (fundings[s]) {
            ret[s] = fundings[s]
        }
    })
    return ret
}

function main() {
    startFundingWorker()
    while (true) {
        exchanges.forEach((e) => {
            let eName = e.GetName()
            let fundings = getFundings(eName, ['BTC_USDT.swap', 'ETH_USDT.swap'])
            Log(eName, fundings)
        })
        Sleep(5000)
    }
}
```

> Detail

https://www.fmz.com/strategy/470345

> Last Modified

2024-10-24 18:36:42
