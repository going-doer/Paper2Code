1. # coding=utf-8
2. from __future__ import print_function, absolute_import, unicode_literals
3. from gm.api import *
4. from datetime import timedelta


7. """
8. 小市值策略
9. 本策略每个月触发一次，计算当前沪深市场上市值最小的前30只股票，并且等权重方式进行买入。
10. 对于不在前30的有持仓的股票直接平仓。
11. 回测时间为：2005-01-01 08:00:00 到 2020-10-01 16:00:00
12. """


15. def init(context):
16. # 每月第一个交易日的09:40 定时执行algo任务（仿真和实盘时不支持该频率）
17. schedule(schedule_func=algo, date_rule='1m', time_rule='09:40:00')
18. # 使用多少的资金来进行开仓。
19. context.ratio = 0.8
20. # 定义股票池数量
21. context.num = 30
22. # 通过get_instruments获取所有的上市股票代码
23. context.all_stock = get_instruments(exchanges='SHSE, SZSE', sec_types=[1], skip_suspended=False,
24. skip_st=False, fields='symbol, listed_date, delisted_date',
25. df=True)


28. def algo(context):
29. # 获取筛选时间：date1表示当前日期之前的100天，date2表示当前时间
30. date1 = (context.now - timedelta(days=100)).strftime("%Y-%m-%d %H:%M:%S")
31. date2 = context.now.strftime("%Y-%m-%d %H:%M:%S")

33. # 上市不足100日的新股和退市股和B股
34. code = context.all_stock[(context.all_stock['listed_date'] < date1) & (context.all_stock['delisted_date'] > date2) &
35. (context.all_stock['symbol'].str[5] != '9') & (context.all_stock['symbol'].str[5] != '2')]
36. # 剔除停牌和st股
37. df_code = get_history_instruments(symbols=code['symbol'].to_list(), start_date=date2, end_date=date2, df=True)
38. df_code = df_code[(df_code['is_suspended'] == 0) & (df_code['sec_level'] == 1)]

40. # 获取所有股票市值
41. fundamental = get_fundamentals_n('trading_derivative_indicator', df_code['symbol'].to_list(),
42. context.now, fields='TOTMKTCAP', order_by='TOTMKTCAP', count=1, df=True)

44. # 对市值进行排序（升序），并且获取前30个。 最后将这个series 转化成为一个list即为标的池
45. trade_symbols = fundamental.reset_index(
46. drop=True).loc[:context.num - 1, 'symbol'].to_list()
47. print('本次股票池有股票数目: ', len(trade_symbols))

49. # 计算每个个股应该在持仓中的权重
50. percent = 1.0 / len(trade_symbols) * context.ratio

52. # 获取当前所有仓位
53. positions = context.account().positions()

55. # 平不在标的池的仓位
56. for position in positions:
57. symbol = position['symbol']
58. if symbol not in trade_symbols:
59. order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market,
60. position_side=PositionSide_Long)
61. print('市价单平不在标的池的', symbol)

63. # 将标中已有持仓的和还没有持仓的都调整到计算出来的比例。
64. for symbol in trade_symbols:
65. order_target_percent(symbol=symbol, percent=percent, order_type=OrderType_Market,
66. position_side=PositionSide_Long)
67. print(symbol, '以市价单调整至权重', percent)


70. if __name__ == '__main__':
71. '''
72. strategy_id策略ID,由系统生成
73. filename文件名,请与本文件名保持一致
74. mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
75. token绑定计算机的ID,可在系统设置-密钥管理中生成
76. backtest_start_time回测开始时间
77. backtest_end_time回测结束时间
78. backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
79. backtest_initial_cash回测初始资金
80. backtest_commission_ratio回测佣金比例
81. backtest_slippage_ratio回测滑点比例
82. '''
83. run(strategy_id='13a64e72-e900-11eb-b05f-309c2322ba62',
84. filename='main.py',
85. mode=MODE_BACKTEST,
86. token='2b62e7651c9897d0cdd4a6cd818a7ba8488af710',
87. backtest_start_time='2005-01-01 08:00:00',
88. backtest_end_time='2020-10-01 16:00:00',
89. backtest_adjust=ADJUST_PREV,
90. backtest_initial_cash=1000000,
91. backtest_commission_ratio=0.0001,
92. backtest_slippage_ratio=0.0001)