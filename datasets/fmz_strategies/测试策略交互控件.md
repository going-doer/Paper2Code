
> Name

测试策略交互控件

> Author

发明者量化-小小梦

> Strategy Description

该策略用于测试发明者量化交易平台策略设计中交互控件功能。

> Strategy Arguments





|Button|Default|Description|
|----|----|----|
|cmdNum1|交互控件cmdNum1的备注|(?数值类型)交互控件cmdNum1的描述|
|cmdNum2|交互控件cmdNum2的备注|交互控件cmdNum2的描述|
|cmdNum3|交互控件cmdNum3的备注|交互控件cmdNum3的描述|
|cmdBool1|交互控件cmdBool1的备注|(?布尔类型)交互控件cmdBool1的描述|
|cmdStr1|交互控件cmdStr1的备注|(?字符串类型)交互控件cmdStr1的描述|
|cmdStr2|交互控件cmdStr2的备注|交互控件cmdStr2的描述|
|cmdStr3|交互控件cmdStr3的备注|交互控件cmdStr3的描述|
|cmdStr4|交互控件cmdStr4的备注|交互控件cmdStr4的描述|
|cmdCombox1|交互控件cmdCombox1的备注|(?下拉框类型)交互控件cmdCombox1的描述|
|cmdCombox2|交互控件cmdCombox2的备注|交互控件cmdCombox2的描述|
|cmdCombox3|交互控件cmdCombox3的备注|交互控件cmdCombox3的描述|
|cmdBtn|交互控件cmdBtn的备注|(?按钮类型)交互控件cmdBtn的描述|


> Source (javascript)

``` javascript
function main() {
    var lastCmd = ""
    while (true) {
        var cmd = GetCommand()
        if (cmd) {
            Log(cmd)
            lastCmd = cmd
        }
        LogStatus(_D(), lastCmd)
        Sleep(500)
    }
}
```

> Detail

https://www.fmz.com/strategy/455231

> Last Modified

2024-06-27 15:06:27
