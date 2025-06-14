# BOLL通道震荡策略

## 策略说明
利用布林带通道的上下轨来捕捉价格震荡，实现高抛低吸。

## 策略代码
```javascript
function init() {
    // 设置交易对
    exchange.SetContractType("spot");
    exchange.SetCurrency("ETH_USDT");
    
    // 设置布林带参数
    this.period = 20;      // 计算周期
    this.multiplier = 2;   // 标准差倍数
}

function onTick() {
    // 获取K线数据
    let records = exchange.GetRecords();
    if (!records || records.length < this.period) {
        return;
    }
    
    // 计算布林带
    let boll = TA.BOLL(records, this.period, this.multiplier);
    let up = boll[0];    // 上轨
    let mid = boll[1];   // 中轨
    let low = boll[2];   // 下轨
    
    // 获取账户信息
    let account = exchange.GetAccount();
    
    // 交易逻辑
    let price = records[records.length-1].Close;
    if (price <= low[records.length-1] && account.Balance > 0) {
        // 价格触及下轨，买入
        exchange.Buy(price, account.Balance / price * 0.99);
        LogProfit(account.Balance, "触及下轨买入");
    } else if (price >= up[records.length-1] && account.Stocks > 0) {
        // 价格触及上轨，卖出
        exchange.Sell(price, account.Stocks);
        LogProfit(account.Balance, "触及上轨卖出");
    }
}
```
