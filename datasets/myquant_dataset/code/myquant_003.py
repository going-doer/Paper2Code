1. # coding=utf-8
2. from __future__ import print_function, absolute_import
3. import pandas as pd
4. from gm.api import *
5. from datetime import datetime, timedelta


8. """
9. R-Breaker是一种短线日内交易策略
10. 根据前一个交易日的收盘价、最高价和最低价数据通过一定方式计算出六个价位，从大到小依次为：
11. 突破买入价、观察卖出价、反转卖出价、反转买入、观察买入价、突破卖出价。以此来形成当前交易
12. 日盘中交易的触发条件。
13. 追踪盘中价格走势，实时判断触发条件。具体条件如下：
14. 突破
15. 在空仓条件下，如果盘中价格超过突破买入价，则采取趋势策略，即在该点位开仓做多。
16. 在空仓条件下，如果盘中价格跌破突破卖出价，则采取趋势策略，即在该点位开仓做空。
17. 反转
18. 持多单，当日内最高价超过观察卖出价后，盘中价格出现回落，且进一步跌破反转卖出价构成的支撑线时，采取反转策略，即在该点位反手做空。
19. 持空单，当日内最低价低于观察买入价后，盘中价格出现反弹，且进一步超过反转买入价构成的阻力线时，采取反转策略，即在该点位反手做多。
20. 设定止损条件。当亏损达到设定值后，平仓。

22. 注意：
23. 1：为回测方便，本策略使用了on_bar的一分钟来计算，实盘中可能需要使用on_tick。
24. 2：实盘中，如果在收盘的那一根bar或tick触发交易信号，需要自行处理，实盘可能不会成交。
25. 3：本策略使用在15点收盘时全平的方式来处理不持有隔夜单的情况，实际使用中15点是无法平仓的。
26. """


29. def init(context):
30. # 设置交易品种
31. context.symbol = 'SHFE.ag'
32. # 设置止损点数
33. context.stopLossPrice = 50

35. # 获取前一交易日的主力合约
36. startDate = get_previous_trading_date(exchange='SHFE', date=context.now.date())
37. continuous_contract = get_continuous_contracts(context.symbol, startDate, startDate)
38. context.mainContract = continuous_contract[0]['symbol']

40. # 获取当前时间
41. time = context.now.strftime('%H:%M:%S')

43. # 如果当前时间是非交易时间段，则直接执行algo,以防直接进入on_bar()导致context.bBreak等未定义
44. if '09:00:00' < time < '15:00:00' or '21:00:00' < time < '23:00:00':
45. algo(context)

47. # 如果是交易时间段，等到开盘时间确保进入algo()
48. schedule(schedule_func = algo, date_rule = '1d', time_rule = '09:00:00')
49. schedule(schedule_func = algo, date_rule = '1d', time_rule = '21:00:00')

51. # 订阅行情
52. subscribe(continuous_contract[0]['symbol'], frequency='60s', count=1)


55. def algo(context):
56. # 检查主力和约，发生变化则更换订阅
57. # 由于主力合约在盘后才公布，为了防止未来函数，选择上一交易日的主力合约。
58. startDate = get_previous_trading_date(exchange='SHFE', date=context.now.date())
59. contractInfo = get_continuous_contracts(context.symbol, startDate, startDate)
60. if context.mainContract != contractInfo[0]['symbol']:
61. context.mainContract = contractInfo[0]['symbol']
62. subscribe(context.mainContract, frequency='60s', count=1, unsubscribe_previous=True)

64. # 获取历史数据
65. data = history_n(symbol=context.mainContract, frequency='1d',
66. end_time=context.now, fields='high,low,open,symbol,close', count=2, df=True)
67. high = data['high'].iloc[0]  # 前一日的最高价
68. low = data['low'].iloc[0]  # 前一日的最低价
69. close = data['close'].iloc[0]  # 前一日的收盘价
70. pivot = (high + low + close) / 3  # 枢轴点
71. context.bBreak = high + 2 * (pivot - low)  # 突破买入价
72. context.sSetup = pivot + (high - low)  # 观察卖出价
73. context.sEnter = 2 * pivot - low  # 反转卖出价
74. context.bEnter = 2 * pivot - high  # 反转买入价
75. context.bSetup = pivot - (high - low)  # 观察买入价
76. context.sBreak = low - 2 * (high - pivot)  # 突破卖出价
77. context.data = data


80. def on_bar(context, bars):

82. # 获取止损价
83. STOP_LOSS_PRICE = context.stopLossPrice

85. # 设置参数
86. bBreak = context.bBreak
87. sSetup = context.sSetup
88. sEnter = context.sEnter
89. bEnter = context.bEnter
90. bSetup = context.bSetup
91. sBreak = context.sBreak
92. data = context.data

94. # 获取现有持仓
95. position_long = context.account().position(symbol=context.mainContract, side=PositionSide_Long)
96. position_short = context.account().position(symbol=context.mainContract, side=PositionSide_Short)

98. # 突破策略:
99. if not position_long and not position_short:  # 空仓条件下
100. if bars[0].close > bBreak:
101. # 在空仓的情况下，如果盘中价格超过突破买入价，则采取趋势策略，即在该点位开仓做多
102. order_volume(symbol=context.mainContract, volume=10, side=OrderSide_Buy,
103. order_type=OrderType_Market, position_effect=PositionEffect_Open)  # 做多
104. print("空仓,盘中价格超过突破买入价: 开仓做多")
105. context.open_position_price = bars[0].close
106. elif bars[0].close < sBreak:
107. # 在空仓的情况下，如果盘中价格跌破突破卖出价，则采取趋势策略，即在该点位开仓做空
108. order_volume(symbol=context.mainContract, volume=10, side=OrderSide_Sell,
109. order_type=OrderType_Market, position_effect=PositionEffect_Open)  # 做空
110. print("空仓,盘中价格跌破突破卖出价: 开仓做空")
111. context.open_position_price = bars[0].close

113. # 设置止损条件
114. else:  # 有持仓时
115. # 开仓价与当前行情价之差大于止损点则止损
116. if (position_long and context.open_position_price - bars[0].close >= STOP_LOSS_PRICE) or \
117. (position_short and bars[0].close - context.open_position_price >= STOP_LOSS_PRICE):
118. print('达到止损点，全部平仓')
119. order_close_all()  # 平仓

121. # 反转策略:
122. if position_long:  # 多仓条件下
123. if data.high.iloc[1] > sSetup and bars[0].close < sEnter:
124. # 多头持仓,当日内最高价超过观察卖出价后，
125. # 盘中价格出现回落，且进一步跌破反转卖出价构成的支撑线时，
126. # 采取反转策略，即在该点位反手做空
127. order_close_all()  # 平仓
128. order_volume(symbol=context.mainContract, volume=10, side=OrderSide_Sell,
129. order_type=OrderType_Market, position_effect=PositionEffect_Open)  # 做空
130. print("多头持仓,当日内最高价超过观察卖出价后跌破反转卖出价: 反手做空")
131. context.open_position_price = bars[0].close
132. elif position_short:  # 空头持仓
133. if data.low.iloc[1] < bSetup and bars[0].close > bEnter:
134. # 空头持仓，当日内最低价低于观察买入价后，
135. # 盘中价格出现反弹，且进一步超过反转买入价构成的阻力线时，
136. # 采取反转策略，即在该点位反手做多
137. order_close_all()  # 平仓
138. order_volume(symbol=context.mainContract, volume=10, side=OrderSide_Buy,
139. order_type=OrderType_Market, position_effect=PositionEffect_Open)  # 做多
140. print("空头持仓,当日最低价低于观察买入价后超过反转买入价: 反手做多")
141. context.open_position_price = bars[0].close

143. if context.now.hour == 14 and context.now.minute == 59:
144. order_close_all()
145. print('全部平仓')


148. if __name__ == '__main__':
149. run(strategy_id='strategy_id',
150. filename='main.py',
151. mode=MODE_BACKTEST,
152. token='token_id',
153. backtest_start_time='2019-10-1 15:00:00',
154. backtest_end_time='2020-04-16 15:00:00',
155. backtest_initial_cash=1000000,
156. backtest_commission_ratio=0.0001,
157. backtest_slippage_ratio=0.0001)