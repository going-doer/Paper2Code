1. # coding=utf-8
2. from __future__ import print_function, absolute_import, unicode_literals
3. import numpy as np
4. from gm.api import *
5. '''
6. 通过计算两个真实价格序列回归残差的0.9个标准差上下轨,并在价差突破上轨的时候做空价差,价差突破下轨的时候做多价差
7. 并在回归至标准差水平内的时候平仓
8. 回测数据为:DCE.m1801和DCE.m1805的1min数据
9. 回测时间为:2017-09-25 08:00:00到2017-10-01 15:00:00
10. '''
11. def init(context):
12. context.goods = ['DCE.m1801', 'DCE.m1805']
13. # 订阅品种数据
14. subscribe(symbols = context.goods,frequency = '1d',count = 31,wait_group = True)

16. def on_bar(context, bars):
17. # 获取历史数据
18. close_1801 = context.data(symbol=context.goods[0], frequency='1d', count=31, fields='close')['close'].values
19. close_1805 = context.data(symbol=context.goods[1], frequency='1d', count=31, fields='close')['close'].values
20. # 计算上下轨
21. spread = close_1801[:-2] - close_1805[:-2]
22. spread_new = close_1801[-1] - close_1805[-1]
23. up = np.mean(spread) + 0.75 * np.std(spread)
24. down = np.mean(spread) - 0.75 * np.std(spread)
25. up_stop = np.mean(spread) + 2 * np.std(spread)
26. down_stop = np.mean(spread) - 2 * np.std(spread)

28. # 获取仓位
29. position1801_long = context.account().position(symbol = context.goods[0],side =PositionSide_Long)
30. position1801_short = context.account().position(symbol = context.goods[0],side =PositionSide_Short)

32. # 没有仓位时
33. if not position1801_short and not position1801_long:
34. # 上穿上轨时，买近卖远
35. if spread_new > up:
36. order_volume(symbol=context.goods[0], volume=1, order_type=OrderType_Market, side=OrderSide_Buy, position_effect=PositionEffect_Open)
37. order_volume(symbol=context.goods[1], volume=1, order_type=OrderType_Market, side=OrderSide_Sell, position_effect=PositionEffect_Open)
38. print('上穿上轨，买近卖远')
39. # 下穿下轨时，卖近买远
40. if spread_new < down:
41. order_volume(symbol=context.goods[0], volume=1, order_type=OrderType_Market, side=OrderSide_Sell, position_effect=PositionEffect_Open)
42. order_volume(symbol=context.goods[1], volume=1, order_type=OrderType_Market, side=OrderSide_Buy, position_effect=PositionEffect_Open)
43. print('下穿下轨，卖近买远')

45. # 价差回归到上轨时，平仓
46. if position1801_long:
47. if spread_new <= np.mean(spread):
48. order_close_all()
49. print('价差回归，平仓')
50. if spread_new > up_stop:
51. order_close_all()
52. print('达到止损点，全部平仓')

54. # 价差回归到下轨时，平仓
55. if position1801_short:
56. if spread_new >= np.mean(spread):
57. order_close_all()
58. print('价差回归，平全部仓')
59. if spread_new < down_stop:
60. order_close_all()
61. print('达到止损点，全部平仓')

63. if __name__ == '__main__':
64. '''
65. strategy_id策略ID,由系统生成
66. filename文件名,请与本文件名保持一致
67. mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
68. token绑定计算机的ID,可在系统设置-密钥管理中生成
69. backtest_start_time回测开始时间
70. backtest_end_time回测结束时间
71. backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
72. backtest_initial_cash回测初始资金
73. backtest_commission_ratio回测佣金比例
74. backtest_slippage_ratio回测滑点比例
75. '''
76. run(strategy_id='strategy_id',
77. filename='main.py',
78. mode=MODE_BACKTEST,
79. token='token_id',
80. backtest_start_time='2017-07-01 08:00:00',
81. backtest_end_time='2017-12-31 16:00:00',
82. backtest_adjust=ADJUST_PREV,
83. backtest_initial_cash=2000000,
84. backtest_commission_ratio=0.0001,
85. backtest_slippage_ratio=0.0001)