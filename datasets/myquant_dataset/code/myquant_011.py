1. # coding=utf-8
2. from __future__ import print_function, absolute_import, unicode_literals
3. from gm.api import *
4. import numpy as np

6. def init(context):
7. # 选择的两个合约
8. context.symbol = ['DCE.j1901', 'DCE.jm1901']
9. # 订阅历史数据
10. subscribe(symbols=context.symbol,frequency='1d',count=11,wait_group=True)

12. def on_bar(context, bars):
13. # 数据提取
14. j_close = context.data(symbol=context.symbol[0],frequency='1d',fields='close',count=31).values
15. jm_close = context.data(symbol=context.symbol[1],frequency='1d',fields='close',count=31).values
16. # 提取最新价差
17. new_price = j_close[-1] - jm_close[-1]
18. # 计算历史价差,上下限，止损点
19. spread_history = j_close[:-2] -  jm_close[:-2]
20. context.spread_history_mean = np.mean(spread_history)
21. context.spread_history_std = np.std(spread_history)
22. context.up = context.spread_history_mean + 0.75 * context.spread_history_std
23. context.down = context.spread_history_mean - 0.75 * context.spread_history_std
24. context.up_stoppoint = context.spread_history_mean + 2 * context.spread_history_std
25. context.down_stoppoint = context.spread_history_mean - 2 * context.spread_history_std
26. # 查持仓
27. position_jm_long = context.account().position(symbol=context.symbol[0],side=1)
28. position_jm_short = context.account().position(symbol=context.symbol[0],side=2)

30. # 设计买卖信号
31. # 设计开仓信号
32. if not position_jm_short and not position_jm_long:
33. if new_price > context.up:
34. print('做空价差组合')
35. order_volume(symbol=context.symbol[0],side=OrderSide_Sell,volume=1,order_type=OrderType_Market,position_effect=1)
36. order_volume(symbol=context.symbol[1], side=OrderSide_Buy, volume=1, order_type=OrderType_Market, position_effect=PositionEffect_Open)
37. if new_price < context.down:
38. print('做多价差组合')
39. order_volume(symbol=context.symbol[0], side=OrderSide_Buy, volume=1, order_type=OrderType_Market, position_effect=PositionEffect_Open)
40. order_volume(symbol=context.symbol[1], side=OrderSide_Sell, volume=1, order_type=OrderType_Market, position_effect=PositionEffect_Open)

42. # 设计平仓信号
43. # 持jm多仓时
44. if position_jm_long:
45. if new_price >= context.spread_history_mean:
46. # 价差回归到均值水平时，平仓
47. print('价差回归到均衡水平，平仓')
48. order_volume(symbol=context.symbol[0], side=OrderSide_Sell, volume=1, order_type=OrderType_Market, position_effect=PositionEffect_Close)
49. order_volume(symbol=context.symbol[1], side=OrderSide_Buy, volume=1, order_type=OrderType_Market, position_effect=PositionEffect_Close)

51. if new_price < context.down_stoppoint:
52. # 价差达到止损位，平仓止损
53. print('价差超过止损点，平仓止损')
54. order_volume(symbol=context.symbol[0], side=OrderSide_Sell, volume=1, order_type=OrderType_Market, position_effect=PositionEffect_Close)
55. order_volume(symbol=context.symbol[1], side=OrderSide_Buy, volume=1, order_type=OrderType_Market, position_effect=PositionEffect_Close)

57. # 持jm空仓时
58. if position_jm_short:
59. if new_price <= context.spread_history_mean:
60. # 价差回归到均值水平时，平仓
61. print('价差回归到均衡水平，平仓')
62. order_volume(symbol=context.symbol[0], side=OrderSide_Buy, volume=1, order_type=OrderType_Market, position_effect=PositionEffect_Close)
63. order_volume(symbol=context.symbol[1], side=OrderSide_Sell, volume=1, order_type=OrderType_Market, position_effect=PositionEffect_Close)

65. if new_price > context.up_stoppoint:
66. # 价差达到止损位，平仓止损
67. print('价差超过止损点，平仓止损')
68. order_volume(symbol=context.symbol[0], side=OrderSide_Buy, volume=1, order_type=OrderType_Market, position_effect=PositionEffect_Close)
69. order_volume(symbol=context.symbol[1], side=OrderSide_Sell, volume=1, order_type=OrderType_Market, position_effect=PositionEffect_Close)


72. if __name__ == '__main__':
73. '''
74. strategy_id策略ID,由系统生成
75. filename文件名,请与本文件名保持一致
76. mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
77. token绑定计算机的ID,可在系统设置-密钥管理中生成
78. backtest_start_time回测开始时间
79. backtest_end_time回测结束时间
80. backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
81. backtest_initial_cash回测初始资金
82. backtest_commission_ratio回测佣金比例
83. backtest_slippage_ratio回测滑点比例
84. '''
85. run(strategy_id='strategy_id',
86. filename='main.py',
87. mode=MODE_BACKTEST,
88. token='token',
89. backtest_start_time='2018-02-01 08:00:00',
90. backtest_end_time='2018-12-31 16:00:00',
91. backtest_adjust=ADJUST_PREV,
92. backtest_initial_cash=2000000,
93. backtest_commission_ratio=0.0001,
94. backtest_slippage_ratio=0.0001)