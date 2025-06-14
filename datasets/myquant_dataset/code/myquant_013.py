1. # coding=utf-8
2. from __future__ import print_function, absolute_import, unicode_literals
3. import sys
4. try:
5. import talib
6. except:
7. print('请安装TA-Lib库')
8. # 安装talib请看文档https://www.myquant.cn/docs/gm3_faq/154?
9. sys.exit(-1)
10. from gm.api import *


13. def init(context):
14. # 设置标的股票
15. context.symbol = 'SHSE.600000'
16. # 用于判定第一个仓位是否成功开仓
17. context.first = 0
18. # 订阅浦发银行, bar频率为1min
19. subscribe(symbols=context.symbol, frequency='60s', count=35)
20. # 日内回转每次交易100股
21. context.trade_n = 100
22. # 获取昨今天的时间
23. context.day = [0, 0]
24. # 用于判断是否到达接近收盘，所以不再交易
25. context.ending = 1


28. def on_bar(context, bars):
29. bar = bars[0]
30. # 配置底仓
31. if context.first == 0:
32. # 需要保持的总仓位
33. context.total = 10000
34. # 购买10000股浦发银行股票
35. order_volume(symbol=context.symbol, volume=context.total, side=OrderSide_Buy,
36. order_type=OrderType_Market, position_effect=PositionEffect_Open)
37. print(context.symbol, '以市价单开多仓10000股')
38. context.first = 1.
39. day = bar.bob.strftime('%Y-%m-%d')
40. context.day[-1] = int(day[-2:])
41. # 每天的仓位操作
42. context.turnaround = [0, 0]
43. return

45. # 更新最新的日期
46. day = bar.bob.strftime('%Y-%m-%d %H:%M:%S')
47. context.day[0] = bar.bob.day

49. # 若为新的一天,获取可用于回转的昨仓
50. if context.day[0] != context.day[-1]:
51. context.ending = 0
52. context.turnaround = [0, 0]

54. # 如果接近收盘，则不再交易
55. if context.ending == 1:
56. return

58. # 若有可用的昨仓则操作
59. if context.total >= 0:
60. # 获取时间序列数据
61. symbol = bar['symbol']
62. recent_data = context.data(symbol=symbol, frequency='60s', count=35, fields='close')
63. # 计算MACD线
64. macd = talib.MACD(recent_data['close'].values)[0][-1]
65. # 根据MACD>0则开仓,小于0则平仓
66. if macd > 0:
67. # 多空单向操作都不能超过昨仓位,否则最后无法调回原仓位
68. if context.turnaround[0] + context.trade_n < context.total:
69. # 计算累计仓位
70. context.turnaround[0] += context.trade_n
71. order_volume(symbol=context.symbol, volume=context.trade_n, side=OrderSide_Buy,
72. order_type=OrderType_Market, position_effect=PositionEffect_Open)
73. print(symbol, '市价单开多仓', context.trade_n, '股')
74. elif macd < 0:
75. if context.turnaround[1] + context.trade_n < context.total:
76. context.turnaround[1] += context.trade_n
77. order_volume(symbol=context.symbol, volume=context.trade_n, side=OrderSide_Sell,
78. order_type=OrderType_Market, position_effect=PositionEffect_Close)
79. print(symbol, '市价单开空仓', context.trade_n, '股')
80. # 临近收盘时若仓位数不等于昨仓则回转所有仓位
81. if day[11:16] == '14:55' or day[11:16] == '14:57':
82. position = context.account().position(symbol=context.symbol, side=PositionSide_Long)
83. if position['volume'] != context.total:
84. order_target_volume(symbol=context.symbol, volume=context.total, order_type=OrderType_Market,
85. position_side=PositionSide_Long)
86. print('市价单回转仓位操作...')
87. context.ending = 1
88. # 更新过去的日期数据
89. context.day[-1] = context.day[0]


92. if __name__ == '__main__':
93. '''
94. strategy_id策略ID,由系统生成
95. filename文件名,请与本文件名保持一致
96. mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
97. token绑定计算机的ID,可在系统设置-密钥管理中生成
98. backtest_start_time回测开始时间
99. backtest_end_time回测结束时间
100. backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
101. backtest_initial_cash回测初始资金
102. backtest_commission_ratio回测佣金比例
103. backtest_slippage_ratio回测滑点比例
104. '''
105. run(strategy_id='strategy_id',
106. filename='main.py',
107. mode=MODE_BACKTEST,
108. token='token_id',
109. backtest_start_time='2017-09-01 08:00:00',
110. backtest_end_time='2017-10-01 16:00:00',
111. backtest_adjust=ADJUST_PREV,
112. backtest_initial_cash=2000000,
113. backtest_commission_ratio=0.0001,
114. backtest_slippage_ratio=0.0001)