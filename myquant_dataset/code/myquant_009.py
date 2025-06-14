1. # coding=utf-8
2. from __future__ import print_function, absolute_import, unicode_literals
3. import numpy as np
4. import pandas as pd
5. from gm.api import *

7. '''
8. 本策略标的为：SHFE.rb1901
9. 价格中枢设定为：前一交易日的收盘价
10. 从阻力位到压力位分别为：1.03 * open、1.02 * open、1.01 * open、open、0.99 * open、0.98 * open、0.97 * open
11. 每变动一个网格，交易量变化100个单位
12. 回测数据为:SHFE.rb1901的1min数据
13. 回测时间为:2017-07-01 08:00:00到2017-10-01 16:00:00
14. '''


17. def init(context):
18. # 策略标的为SHFE.rb1901
19. context.symbol = 'SHFE.rb1901'
20. # 订阅SHFE.rb1901, bar频率为1min
21. subscribe(symbols = context.symbol, frequency='60s')
22. # 设置每变动一格，增减的数量
23. context.volume = 1
24. # 储存前一个网格所处区间，用来和最新网格所处区间作比较
25. context.last_grid = 0
26. # 以前一日的收盘价为中枢价格
27. context.center = history_n(symbol= context.symbol,frequency='1d',end_time=context.now,count = 1,fields = 'close')[0]['close']
28. # 记录上一次交易时网格范围的变化情况（例如从4区到5区，记为4,5）
29. context.grid_change_last = [0,0]


32. def on_bar(context, bars):
33. bar = bars[0]
34. # 获取多仓仓位
35. position_long = context.account().position(symbol=context.symbol, side=PositionSide_Long)
36. # 获取空仓仓位
37. position_short = context.account().position(symbol=context.symbol, side=PositionSide_Short)

39. # 设置网格和当前价格所处的网格区域
40. context.band = np.array([0.97, 0.98, 0.99, 1, 1.01, 1.02, 1.03]) * context.center
41. grid = pd.cut([bar.close], context.band, labels=[1, 2, 3, 4, 5, 6])[0]

43. # 如果价格超出网格设置范围，则提示调节网格宽度和数量
44. if np.isnan(grid):
45. print('价格波动超过网格范围，可适当调节网格宽度和数量')

47. # 如果新的价格所处网格区间和前一个价格所处的网格区间不同，说明触碰到了网格线，需要进行交易
48. # 如果新网格大于前一天的网格，做空或平多
49. if context.last_grid < grid:
50. # 记录新旧格子范围（按照大小排序）
51. grid_change_new = [context.last_grid,grid]
52. # 几种例外：
53. # 当last_grid = 0 时是初始阶段，不构成信号
54. # 如果此时grid = 3，说明当前价格仅在开盘价之下的3区域中，没有突破网格线
55. # 如果此时grid = 4，说明当前价格仅在开盘价之上的4区域中，没有突破网格线
56. if context.last_grid == 0:
57. context.last_grid = grid
58. return
59. if context.last_grid != 0:
60. # 如果前一次开仓是4-5，这一次是5-4，算是没有突破，不成交
61. if grid_change_new != context.grid_change_last:
62. # 更新前一次的数据
63. context.last_grid = grid
64. context.grid_change_last = grid_change_new
65. # 如果有多仓，平多
66. if position_long:
67. order_volume(symbol=context.symbol, volume=context.volume, side=OrderSide_Sell, order_type=OrderType_Market,
68. position_effect=PositionEffect_Close)
69. print('以市价单平多仓{}手'.format(context.volume))
70. # 否则，做空
71. if not position_long:
72. order_volume(symbol=context.symbol, volume=context.volume, side=OrderSide_Sell, order_type=OrderType_Market,
73. position_effect=PositionEffect_Open)
74. print('以市价单开空{}手'.format(context.volume))

76. # 如果新网格小于前一天的网格，做多或平空
77. if context.last_grid > grid:
78. # 记录新旧格子范围（按照大小排序）
79. grid_change_new = [grid,context.last_grid]
80. # 几种例外：
81. # 当last_grid = 0 时是初始阶段，不构成信号
82. # 如果此时grid = 3，说明当前价格仅在开盘价之下的3区域中，没有突破网格线
83. # 如果此时grid = 4，说明当前价格仅在开盘价之上的4区域中，没有突破网格线
84. if context.last_grid == 0:
85. context.last_grid = grid
86. return
87. if context.last_grid != 0:
88. # 如果前一次开仓是4-5，这一次是5-4，算是没有突破，不成交
89. if grid_change_new != context.grid_change_last:
90. # 更新前一次的数据
91. context.last_grid = grid
92. context.grid_change_last = grid_change_new
93. # 如果有空仓，平空
94. if position_short:
95. order_volume(symbol=context.symbol, volume=context.volume, side=OrderSide_Buy,
96. order_type=OrderType_Market,
97. position_effect=PositionEffect_Close)
98. print('以市价单平空仓{}手'.format(context.volume))

100. # 否则，做多
101. if not position_short:
102. order_volume(symbol=context.symbol, volume=context.volume, side=OrderSide_Buy,
103. order_type=OrderType_Market,
104. position_effect=PositionEffect_Open)
105. print('以市价单开多{}手'.format(context.volume))

107. # 设计一个止损条件：当持仓量达到10手，全部平仓
108. if position_short == 10 or position_long == 10:
109. order_close_all()
110. print('触发止损，全部平仓')


113. if __name__ == '__main__':
114. '''
115. strategy_id策略ID,由系统生成
116. filename文件名,请与本文件名保持一致
117. mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
118. token绑定计算机的ID,可在系统设置-密钥管理中生成
119. backtest_start_time回测开始时间
120. backtest_end_time回测结束时间
121. backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
122. backtest_initial_cash回测初始资金
123. backtest_commission_ratio回测佣金比例
124. backtest_slippage_ratio回测滑点比例
125. '''
126. run(strategy_id='strategy_id',
127. filename='main.py',
128. mode=MODE_BACKTEST,
129. token='token_id',
130. backtest_start_time='2018-07-01 08:00:00',
131. backtest_end_time='2018-10-01 16:00:00',
132. backtest_adjust=ADJUST_PREV,
133. backtest_initial_cash=100000,
134. backtest_commission_ratio=0.0001,
135. backtest_slippage_ratio=0.0001)