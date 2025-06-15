1. # coding=utf-8
2. from __future__ import print_function, absolute_import, unicode_literals
3. from gm.api import *


6. '''
7. 本策略每隔1个月定时触发计算SHSE.000300成份股的过去一天EV/EBITDA值并选取30只EV/EBITDA值最小且大于零的股票
8. 对不在股票池的股票平仓并等权配置股票池的标的
9. 并用相应的CFFEX.IF对应的真实合约等额对冲
10. 回测数据为:SHSE.000300和他们的成份股和CFFEX.IF对应的真实合约
11. 回测时间为:2017-07-01 08:00:00到2017-10-01 16:00:00
12. 注意：本策略仅供参考，实际使用中要考虑到期货和股票处于两个不同的账户，需要人为的保证两个账户的资金相同。
13. '''


16. def init(context):
17. # 每月第一个交易日09:40:00的定时执行algo任务（仿真和实盘时不支持该频率）
18. schedule(schedule_func=algo, date_rule='1m', time_rule='09:40:00')
19. # 设置开仓在股票和期货的资金百分比(期货在后面自动进行杠杆相关的调整)
20. context.percentage_stock = 0.4
21. context.percentage_futures = 0.4


24. def algo(context):
25. # 获取当前时刻
26. now = context.now

28. # 获取上一个交易日
29. last_day = get_previous_trading_date(exchange='SHSE', date=now)

31. # 获取沪深300成份股的股票代码
32. stock300 = get_history_constituents(index='SHSE.000300', start_date=last_day,
33. end_date=last_day)[0]['constituents'].keys()

35. # 获取上一个工作日的CFFEX.IF对应的合约
36. index_futures = get_continuous_contracts(csymbol='CFFEX.IF', start_date=last_day, end_date=last_day)[-1]['symbol']

38. # 获取当天有交易的股票
39. not_suspended_info = get_history_instruments(symbols=stock300, start_date=now, end_date=now)
40. not_suspended_symbols = [item['symbol'] for item in not_suspended_info if not item['is_suspended']]

42. # 获取成份股EV/EBITDA大于0并为最小的30个
43. fin = get_fundamentals(table='trading_derivative_indicator', symbols=not_suspended_symbols,
44. start_date=now, end_date=now, fields='EVEBITDA',
45. filter='EVEBITDA>0', order_by='EVEBITDA', limit=30, df=True)
46. fin.index = fin.symbol

48. # 获取当前仓位
49. positions = context.account().positions()

51. # 平不在标的池或不为当前股指期货主力合约对应真实合约的标的
52. for position in positions:
53. symbol = position['symbol']
54. sec_type = get_instrumentinfos(symbols=symbol)[0]['sec_type']

56. # 若类型为期货且不在标的池则平仓
57. if sec_type == SEC_TYPE_FUTURE and symbol != index_futures:
58. order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market,
59. position_side=PositionSide_Short)
60. print('市价单平不在标的池的', symbol)
61. elif symbol not in fin.index:
62. order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market,
63. position_side=PositionSide_Long)
64. print('市价单平不在标的池的', symbol)

66. # 获取股票的权重
67. percent = context.percentage_stock / len(fin.index)

69. # 买在标的池中的股票
70. for symbol in fin.index:
71. order_target_percent(symbol=symbol, percent=percent, order_type=OrderType_Market,
72. position_side=PositionSide_Long)
73. print(symbol, '以市价单调多仓到仓位', percent)

75. # 获取股指期货的保证金比率
76. ratio = get_history_instruments(symbols=index_futures, start_date=last_day, end_date=last_day)[0]['margin_ratio']

78. # 更新股指期货的权重
79. percent = context.percentage_futures * ratio

81. # 买入股指期货对冲
82. # 注意：股指期货的percent参数是按照期货的保证金来算比例，不是按照合约价值， 比如说0.1就是用0.1的仓位的资金全部买入期货。
83. order_target_percent(symbol=index_futures, percent=percent, order_type=OrderType_Market,
84. position_side=PositionSide_Short)
85. print(index_futures, '以市价单调空仓到仓位', percent)


88. if __name__ == '__main__':
89. '''
90. strategy_id策略ID,由系统生成
91. filename文件名,请与本文件名保持一致
92. mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
93. token绑定计算机的ID,可在系统设置-密钥管理中生成
94. backtest_start_time回测开始时间
95. backtest_end_time回测结束时间
96. backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
97. backtest_initial_cash回测初始资金
98. backtest_commission_ratio回测佣金比例
99. backtest_slippage_ratio回测滑点比例
100. '''
101. run(strategy_id='strategy_id',
102. filename='main.py',
103. mode=MODE_BACKTEST,
104. token='token_id',
105. backtest_start_time='2017-07-01 08:00:00',
106. backtest_end_time='2017-10-01 16:00:00',
107. backtest_adjust=ADJUST_PREV,
108. backtest_initial_cash=10000000,
109. backtest_commission_ratio=0.0001,
110. backtest_slippage_ratio=0.0001)