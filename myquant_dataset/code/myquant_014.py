1. # coding=utf-8
2. from __future__ import print_function, absolute_import, unicode_literals
3. from gm.api import *


6. '''
7. 本策略通过不断对CZCE.CF801进行:
8. 买(卖)一价现价单开多(空)仓和卖(买)一价平多(空)仓来做市
9. 并以此赚取差价
10. 回测数据为:CZCE.CF801的tick数据
11. 回测时间为:2017-09-29 11:25:00到2017-09-29 11:30:00
12. 需要特别注意的是:本平台对于回测对限价单固定完全成交,本例子 仅供参考.
13. 敬请通过适当调整回测参数
14. 1.backtest_commission_ratio回测佣金比例
15. 2.backtest_slippage_ratio回测滑点比例
16. 3.backtest_transaction_ratio回测成交比例
17. 以及优化策略逻辑来达到更贴近实际的回测效果
18. 目前只支持最近三个月的tick数据，回测时间和标的需要修改
19. '''


22. def init(context):
23. # 订阅CZCE.CF801的tick数据
24. context.symbol = 'CZCE.CF801'
25. subscribe(symbols=context.symbol, frequency='tick')


28. def on_tick(context, tick):
29. quotes = tick['quotes'][0]
30. # 获取持有的多仓
31. position_long = context.account().position(symbol=context.symbol, side=PositionSide_Long)
32. # 获取持有的空仓
33. position_short = context.account().position(symbol=context.symbol, side=PositionSide_Short)
34. # 没有仓位则双向开限价单
35. # 若有仓位则限价单平仓
36. if not position_long:
37. # 获取买一价
38. price = quotes['bid_p']
39. print('买一价为: ', price)
40. order_target_volume(symbol=context.symbol, volume=1, price=price, order_type=OrderType_Limit,
41. position_side=PositionSide_Long)
42. print('CZCE.CF801开限价单多仓1手')
43. else:
44. # 获取卖一价
45. price = quotes['ask_p']
46. print('卖一价为: ', price)
47. order_target_volume(symbol=context.symbol, volume=0, price=price, order_type=OrderType_Limit,
48. position_side=PositionSide_Long)
49. print('CZCE.CF801平限价单多仓1手')
50. if not position_short:
51. # 获取卖一价
52. price = quotes['ask_p']
53. print('卖一价为: ', price)
54. order_target_volume(symbol=context.symbol, volume=1, price=price, order_type=OrderType_Limit,
55. position_side=PositionSide_Short)
56. print('CZCE.CF801卖一价开限价单空仓')
57. else:
58. # 获取买一价
59. price = quotes['bid_p']
60. print('买一价为: ', price)
61. order_target_volume(symbol=context.symbol, volume=0, price=price, order_type=OrderType_Limit,
62. position_side=PositionSide_Short)
63. print('CZCE.CF801买一价平限价单空仓')


66. if __name__ == '__main__':
67. '''
68. strategy_id策略ID,由系统生成
69. filename文件名,请与本文件名保持一致
70. mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
71. token绑定计算机的ID,可在系统设置-密钥管理中生成
72. backtest_start_time回测开始时间
73. backtest_end_time回测结束时间
74. backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
75. backtest_initial_cash回测初始资金
76. backtest_commission_ratio回测佣金比例
77. backtest_slippage_ratio回测滑点比例
78. backtest_transaction_ratio回测成交比例
79. '''
80. run(strategy_id='strategy_id',
81. filename='main.py',
82. mode=MODE_BACKTEST,
83. token='token_id',
84. backtest_start_time='2017-09-29 11:25:00',
85. backtest_end_time='2017-09-29 11:30:00',
86. backtest_adjust=ADJUST_PREV,
87. backtest_initial_cash=500000,
88. backtest_commission_ratio=0.00006,
89. backtest_slippage_ratio=0.0001,
90. backtest_transaction_ratio=0.5)