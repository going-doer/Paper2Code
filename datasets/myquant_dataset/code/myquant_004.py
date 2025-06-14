1. # coding=utf-8
2. from __future__ import print_function, absolute_import
3. from gm.api import *


6. """
7. 上轨=昨日最高点；
8. 下轨=昨日最低点；
9. 止损=今日开盘价;
10. 如果没有持仓，且现价大于了昨天最高价做多，小于昨天最低价做空。
11. 如果有多头持仓，当价格跌破了开盘价止损。
12. 如果有空头持仓，当价格上涨超过开盘价止损。
13. 选取 进行了回测。
14. 注意：
15. 1：为回测方便，本策略使用了on_bar的一分钟来计算，实盘中可能需要使用on_tick。
16. 2：实盘中，如果在收盘的那一根bar或tick触发交易信号，需要自行处理，实盘可能不会成交。
17. """


20. def init(context):
21. # 设置标的
22. context.symbol = 'SHFE.rb2010'
23. # 订阅一分钟线
24. subscribe(symbols = context.symbol,frequency = '60s',count = 1)
25. # 记录开仓次数，保证一天只开仓一次
26. context.count = 0
27. # 记录当前时间
28. time = context.now.strftime('%H:%M:%S')

30. # 如果当前时间点是交易时间段，则直接执行algo获取历史数据，以防直接进入on_bar()导致context.history_data未定义
31. if '09:00:00' < time < '15:00:00' or '21:00:00' < time < '23:00:00':
32. algo(context)

34. # 如果是非交易时间段，等到上午9点或晚上21点再执行algo()
35. schedule(schedule_func = algo, date_rule = '1d', time_rule = '09:00:00')
36. schedule(schedule_func = algo, date_rule = '1d', time_rule = '21:00:00')


39. def algo(context):
40. # 获取历史的n条信息
41. context.history_data = history_n(symbol=context.symbol, frequency = '1d', end_time = context.now,
42. fields='symbol,open,high,low',count=2, df=True)


45. def on_bar(context,bars):
46. # 取出订阅的一分钟bar
47. bar = bars[0]
48. # 提取数据
49. data = context.history_data
50. # 现有持仓情况
51. position_long = context.account().position(symbol=context.symbol, side=PositionSide_Long)
52. position_short = context.account().position(symbol=context.symbol, side=PositionSide_Short)

54. # 如果是回测模式
55. if context.mode == 2:
56. # 开盘价直接在data最后一个数据里取到,前一交易日的最高和最低价为history_data里面的倒数第二条中取到
57. open = data.loc[1, 'open']
58. high = data.loc[0, 'high']
59. low = data.loc[0, 'low']

61. # 如果是实时模式
62. else:
63. # 开盘价通过current取到
64. open = current(context.symbol)[0]['open']
65. # 实时模式不会返回当天的数据，所以history_data里面的最后一条数据是前一交易日的数据
66. high = data.loc[-1, 'high']
67. low = data.loc[-1, 'low']


70. # 交易逻辑部分
71. if position_long:  # 多头持仓小于开盘价止损。
72. if bar.close < open:
73. order_volume(symbol=context.symbol, volume=1, side=OrderSide_Sell,
74. order_type=OrderType_Market, position_effect=PositionEffect_Close)
75. print('以市价单平多仓')
76. elif position_short: # 空头持仓大于开盘价止损。
77. if bar.close > open:
78. order_volume(symbol=context.symbol, volume=1, side=OrderSide_Buy,
79. order_type=OrderType_Market, position_effect=PositionEffect_Close)
80. print('以市价单平空仓')

82. else:  # 没有持仓。
83. if bar.close > high and not context.count:  # 当前的最新价大于了前一天的最高价
84. # 开多
85. order_volume(symbol=context.symbol, volume=1, side=OrderSide_Buy,
86. order_type=OrderType_Market, position_effect=PositionEffect_Open)
87. print('以市价单开多仓')
88. context.count = 1
89. elif bar.close < low and not context.count:  # 当前最新价小于了前一天的最低价
90. # 开空
91. order_volume(symbol=context.symbol, volume=1, side=OrderSide_Sell,
92. order_type=OrderType_Market, position_effect=PositionEffect_Open)
93. print('以市价单开空仓')
94. context.count = 1

96. # 每天收盘前一分钟平仓
97. if context.now.hour == 14 and context.now.minute == 59:
98. order_close_all()
99. print('全部平仓')
100. context.count = 0


103. if __name__ == '__main__':
104. '''
105. strategy_id策略ID,由系统生成
106. filename文件名,请与本文件名保持一致
107. mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
108. token绑定计算机的ID,可在系统设置-密钥管理中生成
109. backtest_start_time回测开始时间
110. backtest_end_time回测结束时间
111. backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
112. backtest_initial_cash回测初始资金
113. backtest_commission_ratio回测佣金比例
114. backtest_slippage_ratio回测滑点比例
115. '''
116. run(strategy_id='strategy_id',
117. filename='main.py',
118. mode=MODE_BACKTEST,
119. token='token_id',
120. backtest_start_time='2020-01-01 15:00:00',
121. backtest_end_time='2020-09-01 16:00:00',
122. backtest_adjust=ADJUST_PREV,
123. backtest_initial_cash=100000,
124. backtest_commission_ratio=0.0001,
125. backtest_slippage_ratio=0.0001)