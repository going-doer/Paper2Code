
> Name

各大交易所资金费率汇总

> Author

小草





> Source (javascript)

``` javascript
var funding_rates = {'Binance':{},'Huobi':{},'OKEX':{},'Gate':{},'Bybit':{}};
var all_symbols = new Set();
var avg = 0;
var update_avg_time = 0;

function CanParse(json_string) {
	try {
		JSON.parse(json_string);
	} catch (e) {
		return false;
	}
	return true;
}

function GetFundingRate(){
    var finding_rate = HttpQuery("https://fapi.binance.com/fapi/v1/premiumIndex");
    var symbol = null;
    if(finding_rate && CanParse(finding_rate)){
        finding_rate = JSON.parse(finding_rate);
        for (var i = 0; i < finding_rate.length; i++) {
            symbol = finding_rate[i].symbol.slice(0, finding_rate[i].symbol.length - 4);
            if(!finding_rate[i].symbol.endsWith("USDT")){
                continue;
            }
            if(symbol.startsWith(1000)){
                symbol = symbol.slice(4,symbol.length);
            }
            all_symbols.add(symbol);
            funding_rates.Binance[symbol] = _N(parseFloat(finding_rate[i].lastFundingRate)*100,5);
        }
    }
    
    
    finding_rate = HttpQuery("https://api.hbdm.com/swap-api/v1/swap_batch_funding_rate");
    if(finding_rate && CanParse(finding_rate)){
        finding_rate = JSON.parse(finding_rate).data;
        for (var i = 0; i < finding_rate.length; i++) {
            symbol = finding_rate[i].contract_code.split('-')[0];
            all_symbols.add(symbol);
            funding_rates.Huobi[symbol] = _N(parseFloat(finding_rate[i].funding_rate)*100,5);
        }
    }
    
    finding_rate = HttpQuery("https://www.okex.com/priapi/v5/public/funding-rate-all?currencyType=1");
    
    if(finding_rate && CanParse(finding_rate)){
        finding_rate = JSON.parse(finding_rate).data[0].fundingList;
        for (var i = 0; i < finding_rate.length; i++) {
            symbol = finding_rate[i].instId.split('-')[0];
            all_symbols.add(symbol);
            funding_rates.OKEX[symbol] = _N(parseFloat(finding_rate[i].fundingRate)*100,5);
        }
    }
    
    finding_rate = HttpQuery("https://api.gateio.ws/api/v4/futures/usdt/tickers");
    if(finding_rate && CanParse(finding_rate)){
        finding_rate = JSON.parse(finding_rate);
        for (var i = 0; i < finding_rate.length; i++) {
            symbol = finding_rate[i].contract.split('_')[0];
            all_symbols.add(symbol);
            funding_rates.Gate[symbol] = _N(parseFloat(finding_rate[i].funding_rate)*100,5);
        }
    }
    
    var info = HttpQuery("https://api.bybit.com/v2/public/symbols");
    if(info && CanParse(info)){
        info = JSON.parse(info).result;
        for (var i = 0; i < info.length; i++) {
            if(info[i].quote_currency != 'USDT'){
                continue
            }
            symbol = info[i].base_currency;
            finding_rate = HttpQuery("https://api-testnet.bybit.com/public/linear/funding/prev-funding-rate?symbol="+info[i].name);
            Sleep(10);
            if(finding_rate && CanParse(finding_rate)){
                finding_rate = JSON.parse(finding_rate).result;
                if(symbol.endsWith(1000)){
                    symbol = symbol.slice(0,symbol.length-4);
                }
                all_symbols.add(symbol);
                funding_rates.Bybit[symbol] = _N(parseFloat(finding_rate.funding_rate)*100,5);
            }
        }
    }

    
}

function GetAvg(symbol){
    var exs = ['Binance', 'Huobi', 'OKEX', 'Gate', 'Bybit'];
    var count = 0;
    var total = 0;
    for(var i = 0; i < exs.length; i++){
        if(symbol in funding_rates[exs[i]]){
            count += 1;
            total += funding_rates[exs[i]][symbol];
        }
    }
    return _N(total/count, 5);
}

function UpdateStatus(){
    let table = { 
        type: 'table', 
        title: 'Funding Rate%', 
        cols: ['symbol'], // 初始化列，包含 symbol 
        rows: [] 
    };

    let fundingRates = {};

    // 直接从已有的 fundings 数据中构建费率信息
    exchanges.forEach((e) => {
        let eName = e.GetName();
        let fundings = getFundings(eName);
        
        if (fundings) {
            for (let symbol in fundings) {
                let rate = fundings[symbol].Rate; // 假设每个返回对象中有一个 Rate 属性
                if (!fundingRates[symbol]) {
                    fundingRates[symbol] = { total: 0, count: 0, exchanges: {} };
                }
                fundingRates[symbol].total += rate;
                fundingRates[symbol].count++;
                fundingRates[symbol].exchanges[eName] = rate; // 记录费率
            }
        }
    });

    // 构建每个交易对的行和平均费率
    for (let symbol in fundingRates) {
        let data = fundingRates[symbol];
        let averageRate = data.total / data.count; // 计算平均费率

        // 将列头添加到表格中
        if (table.cols.length === 1) { // 只在第一次添加交易对列
            for (let e of exchanges) {
                table.cols.push(e.GetName());
            }
            table.cols.push('Average Rate'); // 添加平均费率列
        }

        // 构建行数据
        let row = [symbol];
        for (let e of exchanges) {
            row.push(data.exchanges[e.GetName()] || 0); // 填充各个交易所的费率
        }
        row.push(averageRate); // 填充平均费率

        table.rows.push(row);
    }

    // 记录状态
    LogStatus(logString + '`' + JSON.stringify(table) + '`');
}

function main() {
    while(true){
        try{
            GetFundingRate();
            UpdateStatus();
            Sleep(10*1000);
        }catch(e){
            Log(e);
        }
    }
}
```

> Detail

https://www.fmz.com/strategy/333315

> Last Modified

2024-11-01 17:02:18
