1. # coding=utf-8
2. from __future__ import print_function, absolute_import, unicode_literals
3. import numpy as np
4. from gm.api import *
5. from pandas import DataFrame
6. '''
7. 本策略每隔1个月定时触发,根据Fama-French三因子模型对每只股票进行回归，得到其alpha值。
8. 假设Fama-French三因子模型可以完全解释市场，则alpha为负表明市场低估该股，因此应该买入。
9. 策略思路：
10. 计算市场收益率、个股的账面市值比和市值,并对后两个进行了分类,
11. 根据分类得到的组合分别计算其市值加权收益率、SMB和HML.
12. 对各个股票进行回归(假设无风险收益率等于0)得到alpha值.
13. 选取alpha值小于0并为最小的10只股票进入标的池
14. 平掉不在标的池的股票并等权买入在标的池的股票
15. 回测数据:SHSE.000300的成份股
16. 回测时间:2017-07-01 08:00:00到2017-10-01 16:00:00
17. '''
18. def init(context):
19. # 每月第一个交易日的09:40 定时执行algo任务（仿真和实盘时不支持该频率）
20. schedule(schedule_func=algo, date_rule='1m', time_rule='09:40:00')
21. # 数据滑窗
22. context.date = 20
23. # 设置开仓的最大资金量
24. context.ratio = 0.8
25. # 账面市值比的大/中/小分类
26. context.BM_BIG = 3.0
27. context.BM_MID = 2.0
28. context.BM_SMA = 1.0
29. # 市值大/小分类
30. context.MV_BIG = 2.0
31. context.MV_SMA = 1.0

33. # 计算市值加权的收益率的函数,MV为市值的分类对应的组别,BM为账目市值比的分类对应的组别
34. def market_value_weighted(stocks, MV, BM):
35. select = stocks[(stocks['NEGOTIABLEMV'] == MV) & (stocks.['BM'] == BM)] # 选出市值为MV，账目市值比为BM的所有股票数据
36. market_value = select['mv'].values     # 对应组的全部市值数据
37. mv_total = np.sum(market_value)        # 市值求和
38. mv_weighted = [mv / mv_total for mv in market_value]   # 市值加权的权重
39. stock_return = select['return'].values

41. # 返回市值加权的收益率的和
42. return_total = []
43. for i in range(len(mv_weighted)):
44. return_total.append(mv_weighted[i] * stock_return[i])
45. return_total = np.sum(return_total)
46. return return_total

48. def algo(context):
49. # 获取上一个交易日的日期
50. last_day = get_previous_trading_date(exchange='SHSE', date=context.now)
51. # 获取沪深300成份股
52. context.stock300 = get_history_constituents(index='SHSE.000300', start_date=last_day,
53. end_date=last_day)[0]['constituents'].keys()
54. # 获取当天有交易的股票
55. not_suspended = get_history_instruments(symbols=context.stock300, start_date=last_day, end_date=last_day)
56. not_suspended = [item['symbol'] for item in not_suspended if not item['is_suspended']]
57. fin = get_fundamentals(table='trading_derivative_indicator', symbols=not_suspended,
58. start_date=last_day, end_date=last_day,fields='PB,NEGOTIABLEMV', df=True)  # 获取P/B和市值数据

60. # 计算账面市值比,为P/B的倒数
61. fin['PB'] = (fin['PB'] ** -1)
62. # 计算市值的50%的分位点,用于后面的分类
63. size_gate = fin['NEGOTIABLEMV'].quantile(0.50)
64. # 计算账面市值比的30%和70%分位点,用于后面的分类
65. bm_gate = [fin['PB'].quantile(0.30), fin['PB'].quantile(0.70)]
66. fin.index = fin.symbol
67. # 设置存放股票收益率的list
68. x_return = []

70. # 对未停牌的股票进行处理
71. for symbol in not_suspended:
72. # 计算收益率，存放到x_return里面
73. close = history_n(symbol=symbol, frequency='1d', count=context.date + 1, end_time=last_day, fields='close',
74. skip_suspended=True, fill_missing='Last', adjust=ADJUST_PREV, df=True)['close'].values
75. stock_return = close[-1] / close[0] - 1
76. pb = fin['PB'][symbol]
77. market_value = fin['NEGOTIABLEMV'][symbol]
78. # 获取[股票代码， 股票收益率, 账面市值比的分类, 市值的分类, 流通市值]
79. # 其中账面市值比的分类为：大（3）、中（2）、小（1）
80. # 流通市值的分类：大（2）、小（1）
81. if pb < bm_gate[0]:
82. if market_value < size_gate:
83. label = [symbol, stock_return, context.BM_SMA, context.MV_SMA, market_value]
84. else:
85. label = [symbol, stock_return, context.BM_SMA, context.MV_BIG, market_value]
86. elif pb < bm_gate[1]:
87. if market_value < size_gate:
88. label = [symbol, stock_return, context.BM_MID, context.MV_SMA, market_value]
89. else:
90. label = [symbol, stock_return, context.BM_MID, context.MV_BIG, market_value]
91. elif market_value < size_gate:
92. label = [symbol, stock_return, context.BM_BIG, context.MV_SMA, market_value]
93. else:
94. label = [symbol, stock_return, context.BM_BIG, context.MV_BIG, market_value]
95. if len(x_return) == 0:
96. x_return = label
97. else:
98. x_return = np.vstack([x_return, label])

100. # 将股票代码、 股票收益率、 账面市值比的分类、 市值的分类、 流通市值存为数据表
101. stocks = DataFrame(data=x_return, columns=['symbol', 'return', 'BM', 'NEGOTIABLEMV', 'mv'])
102. stocks.index = stocks.symbol
103. columns = ['return', 'BM', 'NEGOTIABLEMV', 'mv']
104. for column in columns:
105. stocks[column] = stocks[column].astype(np.float64)

107. # 计算SMB.HML和市场收益率（市值加权法）
108. smb_s = (market_value_weighted(stocks, context.MV_SMA, context.BM_SMA) +
109. market_value_weighted(stocks, context.MV_SMA, context.BM_MID) +
110. market_value_weighted(stocks, context.MV_SMA, context.BM_BIG)) / 3

112. # 获取大市值组合的市值加权组合收益率
113. smb_b = (market_value_weighted(stocks, context.MV_BIG, context.BM_SMA) +
114. market_value_weighted(stocks, context.MV_BIG, context.BM_MID) +
115. market_value_weighted(stocks, context.MV_BIG, context.BM_BIG)) / 3
116. smb = smb_s - smb_b

118. # 获取大账面市值比组合的市值加权组合收益率
119. hml_b = (market_value_weighted(stocks, context.MV_SMA, 3) +
120. market_value_weighted(stocks, context.MV_BIG, context.BM_BIG)) / 2

122. # 获取小账面市值比组合的市值加权组合收益率
123. hml_s = (market_value_weighted(stocks, context.MV_SMA, context.BM_SMA) +
124. market_value_weighted(stocks, context.MV_BIG, context.BM_SMA)) / 2
125. hml = hml_b - hml_s

127. # 获取市场收益率
128. close = history_n(symbol='SHSE.000300', frequency='1d', count=context.date + 1,
129. end_time=last_day, fields='close', skip_suspended=True,
130. fill_missing='Last', adjust=ADJUST_PREV, df=True)['close'].values
131. market_return = close[-1] / close[0] - 1
132. coff_pool = []

134. # 对每只股票进行回归获取其alpha值
135. for stock in stocks.index:
136. x_value = np.array([[market_return], [smb], [hml], [1.0]])
137. y_value = np.array([stocks['return'][stock]])
138. # OLS估计系数
139. coff = np.linalg.lstsq(x_value.T, y_value)[0][3]
140. coff_pool.append(coff)

142. # 获取alpha最小并且小于0的10只的股票进行操作(若少于10只则全部买入)
143. stocks['alpha'] = coff_pool
144. stocks = stocks[stocks.alpha < 0].sort_values(by='alpha').head(10)
145. symbols_pool = stocks.index.tolist()
146. positions = context.account().positions()

148. # 平不在标的池的股票
149. for position in positions:
150. symbol = position['symbol']
151. if symbol not in symbols_pool:
152. order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market,
153. position_side=PositionSide_Long)
154. print('市价单平不在标的池的', symbol)

156. # 获取股票的权重
157. percent = context.ratio / len(symbols_pool)

159. # 买在标的池中的股票
160. for symbol in symbols_pool:
161. order_target_percent(symbol=symbol, percent=percent, order_type=OrderType_Market,
162. position_side=PositionSide_Long)
163. print(symbol, '以市价单调多仓到仓位', percent)

165. if __name__ == '__main__':
166. '''
167. strategy_id策略ID,由系统生成
168. filename文件名,请与本文件名保持一致
169. mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
170. token绑定计算机的ID,可在系统设置-密钥管理中生成
171. backtest_start_time回测开始时间
172. backtest_end_time回测结束时间
173. backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
174. backtest_initial_cash回测初始资金
175. backtest_commission_ratio回测佣金比例
176. backtest_slippage_ratio回测滑点比例
177. '''
178. run(strategy_id='strategy_id',
179. filename='main.py',
180. mode=MODE_BACKTEST,
181. token='token_id',
182. backtest_start_time='2017-07-01 08:00:00',
183. backtest_end_time='2017-10-01 16:00:00',
184. backtest_adjust=ADJUST_PREV,
185. backtest_initial_cash=10000000,
186. backtest_commission_ratio=0.0001,
187. backtest_slippage_ratio=0.0001)