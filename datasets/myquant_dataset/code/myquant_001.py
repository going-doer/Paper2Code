1. # coding=utf-8
2. from __future__ import print_function, absolute_import
3. from gm.api import *
4. import talib


7. '''
8. 本策略以SHFE.rb2101为交易标的，根据其一分钟(即60s频度）bar数据建立双均线模型，
9. 短周期为20，长周期为60，当短期均线由上向下穿越长期均线时做空，
10. 当短期均线由下向上穿越长期均线时做多,每次开仓前先平掉所持仓位，再开仓。
11. 注：为了适用于仿真和实盘，在策略中增加了一个“先判断是否平仓成功再开仓”的判断逻辑，以避免出现未平仓成功，可用资金不足的情况。
12. 回测数据为:SHFE.rb2101的60s频度bar数据
13. 回测时间为:2020-04-01 09:00:00到2020-05-31 15:00:00
14. '''


17. def init(context):
18. context.short = 20                                             # 短周期均线
19. context.long = 60                                              # 长周期均线
20. context.symbol = 'SHFE.rb2101'                                 # 订阅交易标的
21. context.period = context.long + 1                              # 订阅数据滑窗长度
22. context.open_long = False                                      # 开多单标记
23. context.open_short = False                                     # 开空单标记
24. subscribe(context.symbol, '60s', count=context.period)         # 订阅行情


27. def on_bar(context, bars):
28. # 获取通过subscribe订阅的数据
29. prices = context.data(context.symbol, '60s', context.period, fields='close')

31. # 利用talib库计算长短周期均线
32. short_avg = talib.SMA(prices.values.reshape(context.period), context.short)
33. long_avg = talib.SMA(prices.values.reshape(context.period), context.long)

35. # 查询持仓
36. position_long = context.account().position(symbol=context.symbol, side=1)
37. position_short = context.account().position(symbol=context.symbol, side=2)

39. # 短均线下穿长均线，做空(即当前时间点短均线处于长均线下方，前一时间点短均线处于长均线上方)
40. if long_avg[-2] < short_avg[-2] and long_avg[-1] >= short_avg[-1]:

42. # 无多仓情况下，直接开空
43. if not position_long:
44. order_volume(symbol=context.symbol, volume=1, side=OrderSide_Sell, position_effect=PositionEffect_Open,
45. order_type=OrderType_Market)
46. print(context.symbol, '以市价单调空仓到仓位')

48. # 有多仓情况下，先平多，再开空(开空命令放在on_order_status里面)
49. else:
50. context.open_short = True

52. # 以市价平多仓
53. order_volume(symbol=context.symbol, volume=1, side=OrderSide_Sell, position_effect=PositionEffect_Close,
54. order_type=OrderType_Market)
55. print(context.symbol, '以市价单平多仓')

57. # 短均线上穿长均线，做多（即当前时间点短均线处于长均线上方，前一时间点短均线处于长均线下方）
58. if short_avg[-2] < long_avg[-2] and short_avg[-1] >= long_avg[-1]:

60. # 无空仓情况下，直接开多
61. if not position_short:
62. order_volume(symbol=context.symbol, volume=1, side=OrderSide_Buy, position_effect=PositionEffect_Open,
63. order_type=OrderType_Market)
64. print(context.symbol, '以市价单调多仓到仓位')

66. # 有空仓的情况下，先平空，再开多(开多命令放在on_order_status里面)
67. else:
68. context.open_long = True

70. # 以市价平空仓
71. order_volume(symbol=context.symbol, volume=1, side=OrderSide_Buy,
72. position_effect=PositionEffect_Close, order_type=OrderType_Market)
73. print(context.symbol, '以市价单平空仓')


76. def on_order_status(context, order):

78. # 查看下单后的委托状态
79. status = order['status']

81. # 成交命令的方向
82. side = order['side']

84. # 交易类型
85. effect = order['position_effect']

87. # 当平仓委托全成后，再开仓
88. if status == 3:

90. # 以市价开空仓，需等到平仓成功无仓位后再开仓
91. # 如果无多仓且side=2（说明平多仓成功），开空仓
92. if effect == 2 and side == 2 and context.open_short:
93. context.open_short = False
94. order_volume(symbol=context.symbol, volume=1, side=OrderSide_Sell, position_effect=PositionEffect_Open,
95. order_type=OrderType_Market)
96. print(context.symbol, '以市价单调空仓到仓位')

98. # 以市价开多仓,需等到平仓成功无仓位后再开仓
99. # 如果无空仓且side=1（说明平空仓成功），开多仓
100. if effect == 2 and side == 1 and context.open_long:
101. context.open_long = False
102. order_volume(symbol=context.symbol, volume=1, side=OrderSide_Buy, position_effect=PositionEffect_Open,
103. order_type=OrderType_Market)
104. print(context.symbol, '以市价单调多仓到仓位')


107. if __name__ == '__main__':
108. '''
109. strategy_id策略ID,由系统生成
110. filename文件名,请与本文件名保持一致
111. mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
112. token绑定计算机的ID,可在系统设置-密钥管理中生成
113. backtest_start_time回测开始时间
114. backtest_end_time回测结束时间
115. backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
116. backtest_initial_cash回测初始资金
117. backtest_commission_ratio回测佣金比例
118. backtest_slippage_ratio回测滑点比例
119. '''
120. run(strategy_id='strategy_id',
121. filename='main.py',
122. mode=MODE_BACKTEST,
123. token='token_id',
124. backtest_start_time='2020-04-01 09:00:00',
125. backtest_end_time='2020-05-31 15:00:00',
126. backtest_adjust=ADJUST_NONE,
127. backtest_initial_cash=10000000,
128. backtest_commission_ratio=0.0001,
129. backtest_slippage_ratio=0.0001)