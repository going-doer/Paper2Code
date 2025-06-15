1. # coding=utf-8
2. from __future__ import print_function, absolute_import, unicode_literals
3. import numpy as np
4. import pandas as pd
5. from gm.api import *


8. '''
9. 以短期为例：20日线
10. 第一步：获取历史数据，计算唐奇安通道和ATR
11. 第二步：当突破唐奇安通道时，开仓。
12. 第三步：计算加仓和止损信号。
13. '''


16. def init(context):
17. # 设置计算唐奇安通道的参数
18. context.n = 20
19. # 设置合约标的
20. context.symbol = 'DCE.i2012'
21. # 设置交易最大资金比率
22. context.ratio = 0.8
23. # 订阅数据
24. subscribe(symbols=context.symbol, frequency='60s', count=2)
25. # 获取当前时间
26. time = context.now.strftime('%H:%M:%S')
27. # 如果策略执行时间点是交易时间段，则直接执行algo定义atr等参数，以防直接进入on_bar()导致atr等未定义
28. if '09:00:00' < time < '15:00:00' or '21:00:00' < time < '23:00:00':
29. algo(context)
30. # 如果是交易时间段，等到开盘时间确保进入algo()
31. schedule(schedule_func=algo, date_rule='1d', time_rule='09:00:00')
32. schedule(schedule_func=algo, date_rule='1d', time_rule='21:00:00')


35. def algo(context):
36. # 计算通道的数据:当日最低、最高、上一交易日收盘
37. # 注：由于talib库计算ATR的结果与公式求得的结果不符，所以这里利用公式计算ATR
38. # 如果是回测模式,当天的数据直接用history取到
39. if context.mode == 2:
40. data = history_n(symbol=context.symbol, frequency='1d', count=context.n+1, end_time=context.now, fields='close,high,low,bob', df=True) # 计算ATR
41. tr_list = []
42. for i in range(0, len(data)-1):
43. tr = max((data['high'].iloc[i] - data['low'].iloc[i]), data['close'].shift(-1).iloc[i] - data['high'].iloc[i],
44. data['close'].shift(-1).iloc[i] - data['low'].iloc[i])
45. tr_list.append(tr)
46. context.atr = int(np.floor(np.mean(tr_list)))
47. context.atr_half = int(np.floor(0.5 * context.atr))


50. # 计算唐奇安通道
51. context.don_open = np.max(data['high'].values[-context.n:])
52. context.don_close = np.min(data['low'].values[-context.n:])

54. # 如果是实时模式，当天的数据需要用current取到
55. if context.mode == 1:
56. data = history_n(symbol=context.symbol, frequency='1d', count=context.n, end_time=context.now, fields='close,high,low',
57. df=True)  # 计算ATR
58. current_data = current(symbols=context.symbol)   # 最新一个交易日的最高、最低
59. tr_list = []
60. for i in range(1, len(data)):
61. tr = max((data['high'].iloc[i] - data['low'].iloc[i]),
62. data['close'].shift(-1).iloc[i] - data['high'].iloc[i],
63. data['close'].shift(-1).iloc[i] - data['low'].iloc[i])
64. tr_list.append(tr)

66. # 把最新一期tr加入列表中
67. tr_new = max((current_data[0]['high'] - current_data[0]['low']),
68. data['close'].iloc[-1] - current_data[0]['high'],
69. data['close'].iloc[-1] - current_data[0]['low'])
70. tr_list.append(tr_new)
71. context.atr = int(np.floor(np.mean(tr_list)))
72. context.atr_half = int(np.floor(0.5 * context.atr))

74. # 计算唐奇安通道
75. context.don_open = np.max(data['high'].values[-context.n:])
76. context.don_close = np.min(data['low'].values[-context.n:])

78. # 计算加仓点和止损点
79. context.long_add_point = context.don_open + context.atr_half
80. context.long_stop_loss = context.don_open - context.atr_half
81. context.short_add_point = context.don_close - context.atr_half
82. context.short_stop_loss = context.don_close + context.atr_half


85. def on_bar(context, bars):
86. # 提取数据
87. symbol = bars[0]['symbol']
88. recent_data = context.data(symbol=context.symbol, frequency='60s', count=2, fields='close,high,low')
89. close = recent_data['close'].values[-1]

91. # 账户仓位情况
92. position_long = context.account().position(symbol=symbol, side=PositionSide_Long)
93. position_short = context.account().position(symbol=symbol, side=PositionSide_Short)

95. # 当无持仓时
96. if not position_long and not position_short:
97. # 如果向上突破唐奇安通道，则开多
98. if close > context.don_open:
99. order_volume(symbol=symbol, side=OrderSide_Buy, volume=context.atr, order_type=OrderType_Market, position_effect=PositionEffect_Open)
100. print('开多仓atr')

102. # 如果向下突破唐奇安通道，则开空
103. if close < context.don_close:
104. order_volume(symbol=symbol, side=OrderSide_Sell, volume=context.atr, order_type=OrderType_Market, position_effect=PositionEffect_Open)
105. print('开空仓atr')

107. # 有持仓时
108. # 持多仓，继续突破（加仓）
109. if position_long:
110. # 当突破1/2atr时加仓
111. if close > context.long_add_point:
112. order_volume(symbol=symbol, volume=context.atr_half, side=OrderSide_Buy, order_type=OrderType_Market,position_effect=PositionEffect_Open)
113. print('继续加仓0.5atr')
114. context.long_add_point += context.atr_half
115. context.long_stop_loss += context.atr_half
116. # 持多仓，止损位计算
117. if close < context.long_stop_loss:
118. volume_hold = position_long['volume']
119. if volume_hold >= context.atr_half:
120. order_volume(symbol=symbol, volume=context.atr_half, side=OrderSide_Sell, order_type=OrderType_Market, position_effect=PositionEffect_Close)
121. else:
122. order_volume(symbol=symbol, volume=volume_hold, side=OrderSide_Sell, order_type=OrderType_Market,position_effect=PositionEffect_Close)
123. print('平多仓0.5atr')
124. context.long_add_point -= context.atr_half
125. context.long_stop_loss -= context.atr_half

127. # 持空仓，继续突破（加仓）
128. if position_short:
129. # 当跌破加仓点时加仓
130. if close < context.short_add_point:
131. order_volume(symbol = symbol, volume=context.atr_half, side=OrderSide_Sell, order_type=OrderType_Market, position_effect=PositionEffect_Open)
132. print('继续加仓0.5atr')
133. context.short_add_point -= context.atr_half
134. context.short_stop_loss -= context.atr_half

136. # 持多仓，止损位计算
137. if close > context.short_stop_loss:
138. volume_hold = position_short['volume']
139. if volume_hold >= context.atr_half:
140. order_volume(symbol=symbol, volume=context.atr_half, side=OrderSide_Buy, order_type=OrderType_Market, position_effect=PositionEffect_Close)
141. else:
142. order_volume(symbol=symbol, volume=volume_hold, side=OrderSide_Buy, order_type=OrderType_Market,position_effect=PositionEffect_Close)
143. print('平空仓0.5atr')
144. context.short_add_point += context.atr_half
145. context.short_stop_loss += context.atr_half


148. if __name__ == '__main__':
149. '''
150. strategy_id策略ID,由系统生成
151. filename文件名,请与本文件名保持一致
152. mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
153. token绑定计算机的ID,可在系统设置-密钥管理中生成
154. backtest_start_time回测开始时间
155. backtest_end_time回测结束时间
156. backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
157. backtest_initial_cash回测初始资金
158. backtest_commission_ratio回测佣金比例
159. backtest_slippage_ratio回测滑点比例
160. '''
161. run(strategy_id='strategy_id',
162. filename='main.py',
163. mode=MODE_BACKTEST,
164. token='token',
165. backtest_start_time='2020-02-15 09:15:00',
166. backtest_end_time='2020-09-01 15:00:00',
167. backtest_adjust=ADJUST_PREV,
168. backtest_initial_cash=1000000,
169. backtest_commission_ratio=0.0001,
170. backtest_slippage_ratio=0.0001)