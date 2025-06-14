
> Name

回测系统随机行情生成器

> Author

发明者量化-小小梦

> Strategy Description

相关文章： https://www.fmz.com/bbs-topic/10545

> Strategy Arguments



|Argument|Default|Description|
|----|----|----|
|filePathForCSV|CSV文件路径|CSV文件路径|
|startTime|生成K线数据的起始时间，格式：2023-11-01 00:00:00|起始时间(UTC时间)|
|endTime|生成的K线数据的结束时间，格式：2023-11-02 00:00:00|结束时间（UTC时间）|
|KLinePeriod|所要生成的K线数据的K线周期，1h、1d、1m 等。|K线周期|
|firstPrice|K线数据的起始价格|K线数据的起始价格|
|ratio|波动性系数|波动性系数|
|trendType|波动类型|波动类型|
|pround|价格精度，K线数据的价格精度|价格精度|
|vround|订单量精度|订单量精度|
|ct|合约代码，swap 为永续合约。|合约代码|




|Button|Default|Description|
|----|----|----|
|createRecords|生成随机K线数据|生成随机K线数据|


> Source (python)

``` python
import _thread
import json
import math
import csv
import random
import os
import datetime as dt
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

arrTrendType = ["down", "slow_up", "sharp_down", "sharp_up", "narrow_range", "wide_range", "neutral_random"]

def url2Dict(url):
    query = urlparse(url).query  
    params = parse_qs(query)  
    result = {key: params[key][0] for key in params}  
    return result

class Provider(BaseHTTPRequestHandler):
    def do_GET(self):
        global filePathForCSV, pround, vround, ct
        try:
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            dictParam = url2Dict(self.path)
            Log("自定义数据源服务接收到请求，self.path:", self.path, "query 参数：", dictParam)            
            
            eid = dictParam["eid"]
            symbol = dictParam["symbol"]
            arrCurrency = symbol.split(".")[0].split("_")
            baseCurrency = arrCurrency[0]
            quoteCurrency = arrCurrency[1]
            fromTS = int(dictParam["from"]) * int(1000)
            toTS = int(dictParam["to"]) * int(1000)
            priceRatio = math.pow(10, int(pround))
            amountRatio = math.pow(10, int(vround))

            data = {
                "detail": {
                    "eid": eid,
                    "symbol": symbol,
                    "alias": symbol,
                    "baseCurrency": baseCurrency,
                    "quoteCurrency": quoteCurrency,
                    "marginCurrency": quoteCurrency,
                    "basePrecision": vround,
                    "quotePrecision": pround,
                    "minQty": 0.00001,
                    "maxQty": 9000,
                    "minNotional": 5,
                    "maxNotional": 9000000,
                    "priceTick": 10 ** -pround,
                    "volumeTick": 10 ** -vround,
                    "marginLevel": 10,
                    "contractType": ct
                },
                "schema" : ["time", "open", "high", "low", "close", "vol"],
                "data" : []
            }
            
            listDataSequence = []
            with open(filePathForCSV, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                headerIsNoneCount = 0
                if len(header) != len(data["schema"]):
                    Log("CSV文件格式有误，列数不同，请检查！", "#FF0000")
                    return 
                for ele in header:
                    for i in range(len(data["schema"])):
                        if data["schema"][i] == ele or ele == "":
                            if ele == "":
                                headerIsNoneCount += 1
                            if headerIsNoneCount > 1:
                                Log("CSV文件格式有误，请检查！", "#FF0000")
                                return 
                            listDataSequence.append(i)
                            break
                
                while True:
                    record = next(reader, -1)
                    if record == -1:
                        break
                    index = 0
                    arr = [0, 0, 0, 0, 0, 0]
                    for ele in record:
                        arr[listDataSequence[index]] = int(ele) if listDataSequence[index] == 0 else (int(float(ele) * amountRatio) if listDataSequence[index] == 5 else int(float(ele) * priceRatio))
                        index += 1
                    data["data"].append(arr)            
            Log("数据data.detail：", data["detail"], "响应回测系统请求。")
            self.wfile.write(json.dumps(data).encode())
        except BaseException as e:
            Log("Provider do_GET error, e:", e)
        return 

def createServer(host):
    try:
        server = HTTPServer(host, Provider)
        Log("Starting server, listen at: %s:%s" % host)
        server.serve_forever()
    except BaseException as e:
        Log("createServer error, e:", e)
        raise Exception("stop")

class KlineGenerator:
    def __init__(self, start_time, end_time, interval):
        self.start_time = dt.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        self.end_time = dt.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        self.interval = self._parse_interval(interval)
        self.timestamps = self._generate_time_series()

    def _parse_interval(self, interval):
        unit = interval[-1]
        value = int(interval[:-1])

        if unit == "m":
            return value * 60
        elif unit == "h":
            return value * 3600
        elif unit == "d":
            return value * 86400
        else:
            raise ValueError("不支持的K线周期，请使用 'm', 'h', 或 'd'.")

    def _generate_time_series(self):
        timestamps = []
        current_time = self.start_time
        while current_time <= self.end_time:
            timestamps.append(int(current_time.timestamp() * 1000))
            current_time += dt.timedelta(seconds=self.interval)
        return timestamps

    def generate(self, initPrice, trend_type="neutral", volatility=1):
        data = []
        current_price = initPrice
        angle = 0
        for timestamp in self.timestamps:
            angle_radians = math.radians((angle + random.uniform(0, 360)) % 360)
            cos_value = math.cos(angle_radians)   #  -1 ~ 1

            if trend_type == "down":
                change = random.uniform(-0.5, 0.4) * volatility * abs(cos_value) 
            elif trend_type == "slow_up":
                change = random.uniform(-0.4, 0.5) * volatility * abs(cos_value) 
            elif trend_type == "sharp_down":
                change = random.uniform(-10, 7) * volatility * abs(cos_value) 
            elif trend_type == "sharp_up":
                change = random.uniform(-7, 10) * volatility * abs(cos_value) 
            elif trend_type == "narrow_range":
                change = random.uniform(-0.2, 0.2) * volatility * abs(cos_value) 
            elif trend_type == "wide_range":
                change = random.uniform(-20, 20) * volatility * abs(cos_value) 
            else:
                change = random.uniform(-1, 1) * volatility * abs(cos_value) 
            
            open_price = current_price
            close_price = open_price + change
            high_price = open_price + random.uniform(0, abs(change)) if open_price > close_price else close_price + random.uniform(0, abs(change))
            low_price = close_price - random.uniform(0, abs(change)) if open_price > close_price else open_price - random.uniform(0, abs(change))

            if low_price <= 0:
                change = random.uniform(1, 5) * volatility * abs(cos_value) 
                close_price = open_price + change
                high_price = open_price + random.uniform(0, abs(change)) if open_price > close_price else close_price + random.uniform(0, abs(change))
                low_price = open_price * random.uniform(0.8, 1)


            if (high_price >= open_price and open_price >= close_price and close_price >= low_price) or (high_price >= close_price and close_price >= open_price and open_price >= low_price):
                pass
            else:
                Log("异常数据：", high_price, open_price, low_price, close_price, "#FF0000")

            base_volume = random.uniform(1000, 5000)
            volume = base_volume * (1 + abs(change) * 0.2)

            kline = {
                "Time": timestamp,
                "Open": round(open_price, 2),
                "High": round(high_price, 2),
                "Low": round(low_price, 2),
                "Close": round(close_price, 2),
                "Volume": round(volume, 2),
            }
            data.append(kline)
            current_price = close_price
            angle += 1
        return data

    def save_to_csv(self, filename, data):
        with open(filename, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["", "open", "high", "low", "close", "vol"])
            for idx, kline in enumerate(data):
                writer.writerow(
                    [kline["Time"], kline["Open"], kline["High"], kline["Low"], kline["Close"], kline["Volume"]]
                )
        
        Log("当前路径：", os.getcwd())
        with open("data.csv", "r") as file:
            lines = file.readlines()
            if len(lines) > 1:
                Log("文件写入成功，以下是文件内容的一部分：")
                Log("".join(lines[:5]))
            else:
                Log("文件写入失败，文件为空！")

def main():
    Chart({})
    LogReset(1)
    
    try:
        # _thread.start_new_thread(createServer, (("localhost", 9090), ))
        _thread.start_new_thread(createServer, (("0.0.0.0", 9090), ))
        Log("开启自定义数据源服务线程，数据由CSV文件提供。", ", 地址/端口：0.0.0.0:9090", "#FF0000")
    except BaseException as e:
        Log("启动自定义数据源服务失败！")
        Log("错误信息：", e)
        raise Exception("stop")
    
    while True:
        cmd = GetCommand()
        if cmd:
            if cmd == "createRecords":
                Log("生成器参数：", "起始时间：", startTime, "结束时间：", endTime, "K线周期：", KLinePeriod, "初始价格：", firstPrice, "波动类型：", arrTrendType[trendType], "波动性系数：", ratio)
                generator = KlineGenerator(
                    start_time=startTime,
                    end_time=endTime,
                    interval=KLinePeriod,
                )
                kline_data = generator.generate(firstPrice, trend_type=arrTrendType[trendType], volatility=ratio)
                generator.save_to_csv("data.csv", kline_data)
                ext.PlotRecords(kline_data, "%s_%s" % ("records", KLinePeriod))
        LogStatus(_D())
        Sleep(2000)
```

> Detail

https://www.fmz.com/strategy/473286

> Last Modified

2024-12-07 18:32:50
