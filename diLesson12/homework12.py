#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework12.py
@time:2018/4/16 1:04
"""
# 对stock_px 中的三个股票数据拟合ARIMA模型


# 读入数据
import pandas as pd;

data_all = pd.read_csv('D:\\DataguruPyhton\\diLesson04\\stock_px.csv', parse_dates=True, index_col=0);
data = data_all[['AAPL', 'MSFT', 'XOM']]

# 时序图
import matplotlib.pyplot as plt;

# plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
data.plot()
plt.show()

# 自相关图
from statsmodels.graphics.tsaplots import plot_acf

for i in data.columns:
    plot_acf(data[i].iloc[:100]).show()

# 平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF

for i in data.columns:
    print(ADF(data[i]))
# 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore
# 由平稳性检测可知‘AAPL’和‘XOM’需要进行差分处理


# 差分后的结果
D_data = data.diff().dropna()
D_data.columns = ['AAPL', 'MSFT', 'XOM']
D_data.plot()  # 时序图
plt.show()

from statsmodels.graphics.tsaplots import plot_pacf

for i in D_data.columns:
    plot_acf(D_data[i].iloc[:100]).show()  # 自相关图
    plot_pacf(D_data[i].iloc[:200]).show()  # 偏自相关图
    print
    ADF(D_data[i])  # 平稳性检测

# 白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox

for i in D_data.columns:
    print
    acorr_ljungbox(D_data[i], lags=1)  # 返回统计量和p值
# only 'AAPL'  p >0.05


from statsmodels.tsa.arima_model import ARIMA

# 定阶

# pmax = int(len(D_data[i])/10) #一般阶数不超过length/10
# qmax = int(len(D_data[i])/10) #一般阶数不超过length/10
pmax = 3
qmax = 3
for i in D_data.columns:
    bic_matrix = []  # bic矩阵
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:  # 存在部分报错，所以用try来跳过报错。
                tmp.append(ARIMA(data[i], (p, 1, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)

    bic_matrix = pd.DataFrame(bic_matrix)  # 从中可以找出最小值

    p, q = bic_matrix.stack().idxmin()  # 先用stack展平，然后用idxmin找出最小值位置。
    print(u'BIC最小的p值和q值为：%s、%s' % (p, q))
    model = ARIMA(data[i], (0, 1, 1)).fit()  # 建立ARIMA(0, 1, 1)模型
    print
    model.summary()  # 给出一份模型报告
    print
    model.forecast(5)  # 作为期5天的预测，返回预测结果、标准误差、置信区间。
