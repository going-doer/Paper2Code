/*
策略名称: RSI超买超卖策略
策略说明: 使用RSI指标判断超买超卖，结合MACD辅助确认，实现高抛低吸
作者: example
*/

function init() {
    // 设置交易对
    exchange.SetContractType("spot");
    exchange.SetCurrency("ETH_USDT");
    
    // 设置参数
    this.rsiPeriod = 14;    // RSI周期
    this.rsiHigh = 70;      // RSI超买线
    this.rsiLow = 30;       // RSI超卖线
}

function onTick() {
    // 获取K线数据
    let records = exchange.GetRecords();
    if (!records || records.length < this.rsiPeriod) {
        return;
    }
    
    // 计算指标
    let rsi = TA.RSI(records, this.rsiPeriod);
    let macd = TA.MACD(records);
    
    // 获取账户信息
    let account = exchange.GetAccount();
    let position = exchange.GetPosition();
    
    // 交易逻辑
    if (rsi[records.length-1] <= this.rsiLow && 
        macd[2][records.length-1] > macd[2][records.length-2]) {
        // RSI超卖且MACD柱转正，做多
        if (position.length === 0 && account.Balance > 0) {
            exchange.Buy(records[records.length-1].Close, account.Balance * 0.99);
            Log("RSI超卖且MACD柱转正，买入");
        }
    } else if (rsi[records.length-1] >= this.rsiHigh && 
               macd[2][records.length-1] < macd[2][records.length-2]) {
        // RSI超买且MACD柱转负，做空
        if (position.length === 0 && account.Stocks > 0) {
            exchange.Sell(records[records.length-1].Close, account.Stocks);
            Log("RSI超买且MACD柱转负，卖出");
        }
    }
}

function onexit() {
    Log("策略退出，清仓所有持仓");
    exchange.Sell(exchange.GetTicker().Last, account.Stocks);
}
