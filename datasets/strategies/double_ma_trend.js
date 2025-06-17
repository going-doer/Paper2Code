/*
策略名称: 双均线趋势跟踪策略
策略说明: 利用短期和长期移动平均线的交叉来判断趋势，实现趋势跟踪交易
作者: example
*/

function init() {
    // 设置交易对
    exchange.SetContractType("swap");
    exchange.SetCurrency("BTC_USD");
    
    // 设置指标参数
    this.shortPeriod = 5;   // 短期均线周期
    this.longPeriod = 20;   // 长期均线周期
}

function onTick() {
    // 获取K线数据
    let records = exchange.GetRecords();
    if (!records || records.length < this.longPeriod) {
        return;
    }
    
    // 计算均线
    let shortMA = TA.MA(records, this.shortPeriod);
    let longMA = TA.MA(records, this.longPeriod);
    
    // 获取仓位信息
    let position = exchange.GetPosition();
    
    // 交易逻辑
    if (shortMA[records.length-1] > longMA[records.length-1] && 
        shortMA[records.length-2] <= longMA[records.length-2]) {
        // 金叉，做多
        if (position.length === 0) {
            exchange.Buy(records[records.length-1].Close, 1);
            Log("开多仓，价格:", records[records.length-1].Close);
        }
    } else if (shortMA[records.length-1] < longMA[records.length-1] && 
               shortMA[records.length-2] >= longMA[records.length-2]) {
        // 死叉，做空
        if (position.length === 0) {
            exchange.Sell(records[records.length-1].Close, 1);
            Log("开空仓，价格:", records[records.length-1].Close);
        }
    }
}
