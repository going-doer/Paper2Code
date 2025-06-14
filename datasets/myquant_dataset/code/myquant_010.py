1. # coding=utf-8
2. from __future__ import print_function, absolute_import, unicode_literals
3. import numpy as np
4. from gm.api import *
5. from pandas import DataFrame


8. '''
9. 本策略以0.8为初始权重跟踪指数标的沪深300中权重大于0.35%的成份股.
10. 个股所占的百分比为(0.8*成份股权重)*100%.然后根据个股是否:
11. 1.连续上涨5天 2.连续下跌5天
12. 来判定个股是否为强势股/弱势股,并对其把权重由0.8调至1.0或0.6
13. 回测时间为:2017-07-01 08:50:00到2017-10-01 17:00:00
14. '''


17. def init(context):
18. # 资产配置的初始权重,配比为0.6-0.8-1.0
19. context.ratio = 0.8

21. # 获取沪深300当时的成份股和相关数据
22. stock300 = get_history_constituents(index='SHSE.000300', start_date='2017-06-30', end_date='2017-06-30')[0][
23. 'constituents']
24. stock300_symbol = []
25. stock300_weight = []
26. for key in stock300:
27. # 保留权重大于0.35%的成份股
28. if (stock300[key] / 100) > 0.0035:
29. stock300_symbol.append(key)
30. stock300_weight.append(stock300[key] / 100)
31. context.stock300 = DataFrame([stock300_weight], columns=stock300_symbol, index=['weight']).T
32. print('选择的成分股权重总和为: ', np.sum(stock300_weight))
33. subscribe(symbols=stock300_symbol, frequency='1d', count=5, wait_group=True)


36. def on_bar(context, bars):
37. # 若没有仓位则按照初始权重开仓
38. for bar in bars:
39. symbol = bar['symbol']
40. position = context.account().position(symbol=symbol, side=PositionSide_Long)
41. if not position:
42. buy_percent = context.stock300['weight'][symbol] * context.ratio
43. order_target_percent(symbol=symbol, percent=buy_percent, order_type=OrderType_Market,
44. position_side=PositionSide_Long)
45. print(symbol, '以市价单开多仓至仓位:', buy_percent)
46. else:

48. # 获取过去5天的价格数据,若连续上涨则为强势股,权重+0.2;若连续下跌则为弱势股,权重-0.2
49. recent_data = context.data(symbol=symbol, frequency='1d', count=5, fields='close')['close'].tolist()
50. if all(np.diff(recent_data) > 0):
51. buy_percent = context.stock300['weight'][symbol] * (context.ratio + 0.2)
52. order_target_percent(symbol=symbol, percent=buy_percent, order_type=OrderType_Market,
53. position_side=PositionSide_Long)
54. print('强势股', symbol, '以市价单调多仓至仓位:', buy_percent)
55. elif all(np.diff(recent_data) < 0):
56. buy_percent = context.stock300['weight'][symbol] * (context.ratio - 0.2)
57. order_target_percent(symbol=symbol, percent=buy_percent, order_type=OrderType_Market,
58. position_side=PositionSide_Long)
59. print('弱势股', symbol, '以市价单调多仓至仓位:', buy_percent)


62. if __name__ == '__main__':
63. '''
64. strategy_id策略ID,由系统生成
65. filename文件名,请与本文件名保持一致
66. mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
67. token绑定计算机的ID,可在系统设置-密钥管理中生成
68. backtest_start_time回测开始时间
69. backtest_end_time回测结束时间
70. backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
71. backtest_initial_cash回测初始资金
72. backtest_commission_ratio回测佣金比例
73. backtest_slippage_ratio回测滑点比例
74. '''
75. run(strategy_id='strategy_id',
76. filename='main.py',
77. mode=MODE_BACKTEST,
78. token='token_id',
79. backtest_start_time='2017-07-01 08:00:00',
80. backtest_end_time='2017-10-01 16:00:00',
81. backtest_adjust=ADJUST_PREV,
82. backtest_initial_cash=10000000,
83. backtest_commission_ratio=0.0001,
84. backtest_slippage_ratio=0.0001)