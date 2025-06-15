1. # coding=utf-8
2. from __future__ import print_function, absolute_import, unicode_literals
3. import numpy as np
4. from gm.api import *
5. '''
6. 本策略每隔1个月定时触发计算SHSE.000910.SHSE.000909.SHSE.000911.SHSE.000912.SHSE.000913.SHSE.000914
7. (300工业.300材料.300可选.300消费.300医药.300金融)这几个行业指数过去
8. 20个交易日的收益率并选取了收益率最高的指数的成份股获取并获取了他们的市值数据
9. 随后把仓位调整至市值最大的5只股票上
10. 回测数据为:SHSE.000910.SHSE.000909.SHSE.000911.SHSE.000912.SHSE.000913.SHSE.000914和他们的成份股
11. 回测时间为:2017-07-01 08:00:00到2017-10-01 16:00:00
12. '''
13. def init(context):
14. # 每月第一个交易日的09:40 定时执行algo任务（仿真和实盘时不支持该频率）
15. schedule(schedule_func=algo, date_rule='1m', time_rule='09:40:00')
16. # 用于筛选的行业指数
17. context.index = ['SHSE.000910', 'SHSE.000909', 'SHSE.000911', 'SHSE.000912', 'SHSE.000913', 'SHSE.000914']
18. # 用于统计数据的天数
19. context.date = 20
20. # 最大下单资金比例
21. context.ratio = 0.8

23. def algo(context):
24. # 获取当天的日期
25. today = context.now
26. # 获取上一个交易日
27. last_day = get_previous_trading_date(exchange='SHSE', date=today)
28. return_index = []

30. # 获取并计算行业指数收益率
31. for i in context.index:
32. return_index_his = history_n(symbol=i, frequency='1d', count=context.date, fields='close,bob',
33. fill_missing='Last', adjust=ADJUST_PREV, end_time=last_day, df=True)
34. return_index_his = return_index_his['close'].values
35. return_index.append(return_index_his[-1] / return_index_his[0] - 1)

37. # 获取指定数内收益率表现最好的行业
38. sector = context.index[np.argmax(return_index)]
39. print('最佳行业指数是: ', sector)

41. # 获取最佳行业指数成份股
42. symbols = get_history_constituents(index=sector, start_date=last_day, end_date=last_day)[0]['constituents'].keys()

44. # 获取当天有交易的股票
45. not_suspended_info = get_history_instruments(symbols=symbols, start_date=today, end_date=today)
46. not_suspended_symbols = [item['symbol'] for item in not_suspended_info if not item['is_suspended']]

48. # 获取最佳行业指数成份股的市值，从大到小排序并选取市值最大的5只股票
49. fin = get_fundamentals(table='trading_derivative_indicator', symbols=not_suspended_symbols, start_date=last_day,
50. end_date=last_day, limit=5, fields='NEGOTIABLEMV', order_by='-NEGOTIABLEMV', df=True)
51. fin.index = fin['symbol']

53. # 计算权重
54. percent = 1.0 / len(fin.index) * context.ratio

56. # 获取当前所有仓位
57. positions = context.account().positions()

59. # 如标的池有仓位,平不在标的池的仓位
60. for position in positions:
61. symbol = position['symbol']
62. if symbol not in fin.index:
63. order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market,
64. position_side=PositionSide_Long)
65. print('市价单平不在标的池的', symbol)

67. # 对标的池进行操作
68. for symbol in fin.index:
69. order_target_percent(symbol=symbol, percent=percent, order_type=OrderType_Market,
70. position_side=PositionSide_Long)
71. print(symbol, '以市价单调整至仓位', percent)

73. if __name__ == '__main__':
74. '''
75. strategy_id策略ID,由系统生成
76. filename文件名,请与本文件名保持一致
77. mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
78. token绑定计算机的ID,可在系统设置-密钥管理中生成
79. backtest_start_time回测开始时间
80. backtest_end_time回测结束时间
81. backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
82. backtest_initial_cash回测初始资金
83. backtest_commission_ratio回测佣金比例
84. backtest_slippage_ratio回测滑点比例
85. '''
86. run(strategy_id='strategy_id',
87. filename='main.py',
88. mode=MODE_BACKTEST,
89. token='token_id',
90. backtest_start_time='2017-07-01 08:00:00',
91. backtest_end_time='2017-10-01 16:00:00',
92. backtest_adjust=ADJUST_PREV,
93. backtest_initial_cash=10000000,
94. backtest_commission_ratio=0.0001,
95. backtest_slippage_ratio=0.0001)