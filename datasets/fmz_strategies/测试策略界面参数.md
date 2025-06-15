
> Name

测试策略界面参数

> Author

发明者量化-小小梦

> Strategy Description

该策略用于测试发明者量化交易平台策略设计中界面参数功能。

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|pNum1|参数pNum1的备注|(?数字型参数)参数pNum1的描述|
|pNum2|参数pNum2的备注|参数pNum2的描述|
|pNum3|参数pNum3的备注|参数pNum3的描述|
|pNum4|参数pNum4的备注|参数pNum4的描述|
|pBool1|参数pBool1的备注|(?布尔型参数)参数pBool1的描述|
|pBool2|参数pBool2的备注，用于控制是否显示pNum1|参数pBool2的描述|
|pStr1|参数pStr1的备注|(?字符串类型)参数pStr1的描述|
|pStr2|参数pStr2的备注|参数pStr2的描述|
|pStr3|参数pStr3的备注|参数pStr3的描述|
|pStr4|参数pStr4的备注|参数pStr4的描述|
|pCombox1|参数pCombox1的备注|(?下拉框类型)参数pCombox1的描述|
|pCombox2|参数pCombox2的备注|参数pCombox2的描述|
|pCombox3|参数pCombox3的备注|参数pCombox3的描述|
|pSecretStr1|参数pSecretStr1的备注|(?加密串类型)参数pSecretStr1的描述|


> Source (javascript)

``` javascript
/*backtest
start: 2023-06-21 00:00:00
end: 2024-06-26 00:00:00
period: 1d
basePeriod: 1h
exchanges: [{"eid":"Futures_OKX","currency":"BTC_USD"}]
*/

function main() {
    Log("---------------------------开始测试数字类型参数---------------------------")
    Log("变量pNum1:", pNum1, ", 变量值类型：", typeof(pNum1))
    Log("变量pNum2:", pNum2, ", 变量值类型：", typeof(pNum2))
    Log("变量pNum3:", pNum3, ", 变量值类型：", typeof(pNum3))
    Log("变量pNum4:", pNum4, ", 变量值类型：", typeof(pNum4))
    
    Log("---------------------------开始测试布尔类型参数---------------------------")
    Log("变量pBool1:", pBool1, ", 变量值类型：", typeof(pBool1))
    Log("变量pBool2:", pBool2, ", 变量值类型：", typeof(pBool2))

    Log("---------------------------开始测试字符串类型参数---------------------------")
    Log("变量pStr1:", pStr1, ", 变量值类型：", typeof(pStr1))
    Log("变量pStr2:", pStr2, ", 变量值类型：", typeof(pStr2))
    Log("变量pStr3:", pStr3, ", 变量值类型：", typeof(pStr3))
    Log("变量pStr4:", pStr4, ", 变量值类型：", typeof(pStr4))

    Log("---------------------------开始测试下拉框类型参数---------------------------")
    Log("变量pCombox1:", pCombox1, ", 变量值类型：", typeof(pCombox1))
    Log("变量pCombox2:", pCombox2, ", 变量值类型：", typeof(pCombox2))
    Log("变量pCombox3:", pCombox3, ", 变量值类型：", typeof(pCombox3))

    Log("---------------------------开始测试加密串类型参数---------------------------")
    Log("变量pSecretStr1:", pSecretStr1, ", 变量值类型：", typeof(pSecretStr1))
}
```

> Detail

https://www.fmz.com/strategy/455212

> Last Modified

2024-06-27 15:05:19
