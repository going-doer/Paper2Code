1. # coding=utf-8
2. from __future__ import print_function, absolute_import
3. from gm.api import *


6. """
7. Dual Thrust是一个趋势跟踪系统
8. 计算前N天的最高价－收盘价和收盘价－最低价。然后取这2N个价差的最大值，乘以k值。把结果称为触发值。
9. 在今天的开盘，记录开盘价，然后在价格超过上轨（开盘＋触发值）时马上买入，或者价格低于下轨（开盘－触发值）时马上卖空。
10. 没有明确止损。这个系统是反转系统，也就是说，如果在价格超过（开盘＋触发值）时手头有空单，则平空开多。
11. 同理，如果在价格低于（开盘－触发值）时手上有多单，则平多开空。
12. 选用了SHFE的rb2010 在2020-02-07 15:00:00 到 2020-04-15 15:00:00' 进行回测。
13. 注意：
14. 1：为回测方便，本策略使用了on_bar的一分钟来计算，实盘中可能需要使用on_tick。
15. 2：实盘中，如果在收盘的那一根bar或tick触发交易信号，需要自行处理，实盘可能不会成交
16. """


19. # 策略中必须有init方法
20. def init(context):

22. # 设置要进行回测的合约（可以在掘金终端的仿真交易中查询标的代码）
23. context.symbol = 'SHFE.rb2010'  # 订阅&交易标的, 此处订阅的是上期所的螺纹钢 2010

25. # 设置参数
26. context.N = 5
27. context.k1 = 0.2
28. context.k2 = 0.2

30. # 获取当前时间
31. time = context.now.strftime('%H:%M:%S')

33. # 如果策略执行时间点是交易时间段，则直接执行algo定义buy_line和sell_line，以防直接进入on_bar()导致context.buy_line和context.sell_line未定义
34. if '09:00:00' < time < '15:00:00' or '21:00:00' < time < '23:00:00':
35. algo(context)

37. # 如果是交易时间段，等到开盘时间确保进入algo()
38. schedule(schedule_func = algo, date_rule = '1d', time_rule = '09:00:00')
39. schedule(schedule_func = algo, date_rule = '1d', time_rule = '21:00:00')

41. # 只需要最新价，所以只需要订阅一个, 如果用tick，次数太多，用一分钟线代替
42. subscribe(symbols=context.symbol, frequency='60s', count = 1)


45. def algo(context):
46. # 取历史数据
47. data = history_n(symbol=context.symbol, frequency='1d', end_time=context.now,
48. fields='symbol,open,high,low,close', count=context.N + 1, df=True)
49. # 取开盘价
50. # 回测模式下，开盘价可以直接用history_n取到
51. if context.mode == 2:
52. # 获取当天的开盘价
53. current_open = data['open'].loc[context.N]
54. # 去掉当天的实时数据
55. data.drop(context.N, inplace = True)
56. # 如果是实时模式，开盘价需要用current取到
57. else:
58. # 获取当天的开盘价
59. current_open = current(context.symbol)[0]['open']

61. # 计算Dual Thrust 的上下轨
62. HH = data['high'].max()
63. HC = data['close'].max()
64. LC = data['close'].min()
65. LL = data['low'].min()
66. range = max(HH - LC, HC - LL)
67. context.buy_line = current_open + range * context.k1  # 上轨
68. context.sell_line = current_open - range * context.k2  # 下轨


71. def on_bar(context, bars):
72. # 取出订阅的这一分钟的bar
73. bar = bars[0]
74. # 取出买卖线
75. buy_line =  context.buy_line
76. sell_line = context.sell_line

78. # 获取现有持仓
79. position_long = context.account().position(symbol=context.symbol, side=PositionSide_Long)
80. position_short = context.account().position(symbol=context.symbol, side=PositionSide_Short)

82. # 交易逻辑部分
83. # 如果超过range的上界
84. if bar.close > buy_line:
85. if position_long:  # 已经持有多仓，直接返回
86. return
87. elif position_short:  # 已经持有空仓，平仓再做多。
88. order_volume(symbol=context.symbol, volume=1, side=OrderSide_Buy,
89. order_type=OrderType_Market, position_effect=PositionEffect_Close)
90. print('市价单平空仓', context.symbol)
91. order_volume(symbol=context.symbol, volume=1, side=OrderSide_Buy,
92. order_type=OrderType_Market, position_effect=PositionEffect_Open)
93. print('市价单开多仓', context.symbol)
94. else:  # 没有持仓时，市价开多。
95. order_volume(symbol=context.symbol, volume=1, side=OrderSide_Buy,
96. order_type=OrderType_Market, position_effect=PositionEffect_Open)
97. print('市价单开多仓', context.symbol)
98. # 如果低于range的下界
99. elif bar.close < sell_line:
100. if position_long:  # 已经持有多仓， 平多再开空。
101. order_volume(symbol=context.symbol, volume=1, side=OrderSide_Sell,
102. order_type=OrderType_Market, position_effect=PositionEffect_Close)
103. print('市价单平多仓', context.symbol)
104. order_volume(symbol=context.symbol, volume=1, side=OrderSide_Sell,
105. order_type=OrderType_Market, position_effect=PositionEffect_Open)
106. print('市价单开空仓', context.symbol)
107. elif position_short:  # 已经持有空仓，直接返回。
108. return
109. else:  # 没有持仓，直接开空
110. order_volume(symbol=context.symbol, volume=1, side=OrderSide_Sell,
111. order_type=OrderType_Market, position_effect=PositionEffect_Open)
112. print('市价单开空仓', context.symbol)


115. if __name__ == '__main__':
116. '''
117. strategy_id策略ID,由系统生成
118. filename文件名,请与本文件名保持一致
119. mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
120. token绑定计算机的ID,可在系统设置-密钥管理中生成
121. backtest_start_time回测开始时间
122. backtest_end_time回测结束时间
123. backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
124. backtest_initial_cash回测初始资金
125. backtest_commission_ratio回测佣金比例
126. backtest_slippage_ratio回测滑点比例
127. '''
128. run(strategy_id='strategy_id',
129. filename='main.py',
130. mode=MODE_BACKTEST,
131. token='token_id',
132. backtest_start_time='2020-02-07 15:00:00',
133. backtest_end_time='2020-04-15 15:00:00',
134. backtest_initial_cash= 30000,
135. backtest_commission_ratio=0.0001,
136. backtest_slippage_ratio=0.0001)