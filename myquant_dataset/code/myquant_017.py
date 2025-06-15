1. # coding=utf-8
2. from __future__ import print_function, absolute_import, unicode_literals
3. from datetime import datetime
4. import numpy as np
5. from gm.api import *
6. import sys
7. try:
8. from sklearn import svm
9. except:
10. print('请安装scikit-learn库和带mkl的numpy')
11. sys.exit(-1)
12. '''
13. 本策略选取了七个特征变量组成了滑动窗口长度为15天的训练集,随后训练了一个二分类(上涨/下跌)的支持向量机模型.
14. 若没有仓位则在每个星期一的时候输入标的股票近15个交易日的特征变量进行预测,并在预测结果为上涨的时候购买标的.
15. 若已经持有仓位则在盈利大于10%的时候止盈,在星期五损失大于2%的时候止损.
16. 特征变量为:1.收盘价/均值2.现量/均量3.最高价/均价4.最低价/均价5.现量6.区间收益率7.区间标准差
17. 训练数据为:SHSE.600000浦发银行,时间从2016-04-01到2017-07-30
18. 回测时间为:2017-07-01 09:00:00到2017-10-01 09:00:00
19. '''

21. def init(context):
22. # 订阅浦发银行的分钟bar行情
23. context.symbol = 'SHSE.600000'
24. subscribe(symbols=context.symbol, frequency='60s')
25. start_date = '2016-04-01'  # SVM训练起始时间
26. end_date = '2017-07-30'  # SVM训练终止时间

28. # 用于记录工作日
29. # 获取目标股票的daily历史行情
30. recent_data = history(context.symbol, frequency='1d', start_time=start_date, end_time=end_date, fill_missing='last',
31. df=True)
32. days_value = recent_data['bob'].values
33. days_close = recent_data['close'].values
34. days = []

36. # 获取行情日期列表
37. print('准备数据训练SVM')
38. for i in range(len(days_value)):
39. days.append(str(days_value[i])[0:10])
40. x_all = []
41. y_all = []
42. for index in range(15, (len(days) - 5)):
43. # 计算三星期共15个交易日相关数据
44. start_day = days[index - 15]
45. end_day = days[index]
46. data = history(context.symbol, frequency='1d', start_time=start_day, end_time=end_day, fill_missing='last',
47. df=True)
48. close = data['close'].values
49. max_x = data['high'].values
50. min_n = data['low'].values
51. amount = data['amount'].values
52. volume = []
53. for i in range(len(close)):
54. volume_temp = amount[i] / close[i]
55. volume.append(volume_temp)
56. close_mean = close[-1] / np.mean(close)  # 收盘价/均值
57. volume_mean = volume[-1] / np.mean(volume)  # 现量/均量
58. max_mean = max_x[-1] / np.mean(max_x)  # 最高价/均价
59. min_mean = min_n[-1] / np.mean(min_n)  # 最低价/均价
60. vol = volume[-1]  # 现量
61. return_now = close[-1] / close[0]  # 区间收益率
62. std = np.std(np.array(close), axis=0)  # 区间标准差

64. # 将计算出的指标添加到训练集X
65. # features用于存放因子
66. features = [close_mean, volume_mean, max_mean, min_mean, vol, return_now, std]
67. x_all.append(features)

69. # 准备算法需要用到的数据
70. for i in range(len(days_close) - 20):
71. if days_close[i + 20] > days_close[i + 15]:
72. label = 1
73. else:
74. label = 0
75. y_all.append(label)
76. x_train = x_all[: -1]
77. y_train = y_all[: -1]

79. # 训练SVM
80. context.clf = svm.SVC(C=0.6, kernel='rbf', gamma=0.001)
81. context.clf.fit(x_train, y_train)
82. print('训练完成!')

84. def on_bar(context, bars):
85. bar = bars[0]

87. # 获取当前年月日
88. today = bar.bob.strftime('%Y-%m-%d')

90. # 获取数据并计算相应的因子
91. # 于星期一的09:31:00进行操作
92. # 当前bar的工作日
93. weekday = datetime.strptime(today, '%Y-%m-%d').isoweekday()

95. # 获取模型相关的数据
96. # 获取持仓
97. position = context.account().position(symbol=context.symbol, side=PositionSide_Long)

99. # 如果bar是新的星期一且没有仓位则开始预测
100. if not position and weekday == 1:
101. # 获取预测用的历史数据
102. data = history_n(symbol=context.symbol, frequency='1d', end_time=today, count=15,
103. fill_missing='last', df=True)
104. close = data['close'].values
105. train_max_x = data['high'].values
106. train_min_n = data['low'].values
107. train_amount = data['amount'].values
108. volume = []
109. for i in range(len(close)):
110. volume_temp = train_amount[i] / close[i]
111. volume.append(volume_temp)
112. close_mean = close[-1] / np.mean(close)
113. volume_mean = volume[-1] / np.mean(volume)
114. max_mean = train_max_x[-1] / np.mean(train_max_x)
115. min_mean = train_min_n[-1] / np.mean(train_min_n)
116. vol = volume[-1]
117. return_now = close[-1] / close[0]
118. std = np.std(np.array(close), axis=0)

120. # 得到本次输入模型的因子
121. features = [close_mean, volume_mean, max_mean, min_mean, vol, return_now, std]
122. features = np.array(features).reshape(1, -1)
123. prediction = context.clf.predict(features)[0]

125. # 若预测值为上涨则开仓
126. if prediction == 1:
127. # 获取昨收盘价
128. context.price = close[-1]
129. # 把浦发银行的仓位调至95%
130. order_target_percent(symbol=context.symbol, percent=0.95, order_type=OrderType_Market,
131. position_side=PositionSide_Long)
132. print('SHSE.600000以市价单开多仓到仓位0.95')

134. # 当涨幅大于10%,平掉所有仓位止盈
135. elif position and bar.close / context.price >= 1.10:
136. order_close_all()
137. print('SHSE.600000以市价单全平多仓止盈')

139. # 当时间为周五并且跌幅大于2%时,平掉所有仓位止损
140. elif position and bar.close / context.price < 1.02 and weekday == 5:
141. order_close_all()
142. print('SHSE.600000以市价单全平多仓止损')

144. if __name__ == '__main__':
145. '''
146. strategy_id策略ID,由系统生成
147. filename文件名,请与本文件名保持一致
148. mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
149. token绑定计算机的ID,可在系统设置-密钥管理中生成
150. backtest_start_time回测开始时间
151. backtest_end_time回测结束时间
152. backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
153. backtest_initial_cash回测初始资金
154. backtest_commission_ratio回测佣金比例
155. backtest_slippage_ratio回测滑点比例
156. '''
157. run(strategy_id='strategy_id',
158. filename='main.py',
159. mode=MODE_BACKTEST,
160. token='token_id',
161. backtest_start_time='2017-07-01 09:00:00',
162. backtest_end_time='2017-10-01 09:00:00',
163. backtest_adjust=ADJUST_PREV,
164. backtest_initial_cash=10000000,
165. backtest_commission_ratio=0.0001,
166. backtest_slippage_ratio=0.0001)