#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework11.py
@time:2018/4/8 13:08
"""

'''
第1题.data1 是40名癌症病人的一些生存资料，其中，X1表示生活行动能力评分（1~100），X2表示病人的年龄，X3表示由诊断到直入研究时间（月）；X4表示肿瘤类型，X5把ISO两种疗法（“1”是常规，“0”是试验新疗法）；Y表示病人生存时间（“0”表示生存时间小于200天，“1”表示生存时间大于或等于200天）
试建立Y关于X1~X5的logistic回归模型
'''
import pandas as pd

data = pd.read_table('D:\\DataguruPyhton\\diLesson04\\data1.txt', index_col=0)
x = data.iloc[:, 0:5].as_matrix()
y = data.iloc[:, 5].as_matrix()

from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR

rlr = RLR()  # 建立随机逻辑回归模型，筛选变量
rlr.fit(x, y)  # 训练模型
rlr.get_support()  # 获取特征筛选结果，也可以通过.scores_方法获取各个特征的分数
print(u'通过随机逻辑回归模型筛选特征结束。')
print(u'有效特征为：%s' % ','.join(data.columns[rlr.get_support()]))
print(u'无有效特征值')

lr = LR()  # 建立逻辑回归模型
lr.fit(x, y)  # 用筛选后的特征数据来训练模型

print(zip(['X1', 'X2', 'X3', 'X4', 'X5'], lr.coef_[0]))
print(lr.intercept_)
print(u'逻辑回归模型训练结束。')
print(u'模型的平均正确率为：%s' % lr.score(x, y))  # 给出模型的平均正确率

'''
第2题.data2 是关于重伤病人的一些基本资料。自变量X是病人的住院天数，因变量Y是病人出院后长期恢复的预后指数，指数数值越大表示预后结局越好。
尝试对数据拟合合适的线性或非线性模型
'''
import pandas as pd
from pandas import DataFrame

data = pd.read_table('D:\\DataguruPyhton\\diLesson04\\data2.txt')
x = pd.DataFrame(data['X'])
y = pd.DataFrame(data['Y'])

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x, y)
fig.show()

# linear regression
from sklearn import metrics
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(x, y)

# The coefficients
print("Coefficients:", linreg.coef_[0][0])

y_pred = linreg.predict(x)
# The mean square error
print("MSE:", metrics.mean_squared_error(y, y_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % linreg.score(x, y))

# 多项式模型
x1 = x
x2 = x ** 2
x1['x2'] = x2
linreg = LinearRegression()
linreg.fit(x1, y)

# The coefficients
print("Coefficients:", linreg.coef_)

y_pred = linreg.predict(x)
# The mean square error
print("MSE:", metrics.mean_squared_error(y, y_pred))
print('Variance score: %.2f' % linreg.score(x1, y))

# 对数模型
x2 = pd.DataFrame(np.log(x))

linreg = LinearRegression()
linreg.fit(x2, y)

# The coefficients
print("Coefficients:", linreg.coef_)

y_pred = linreg.predict(x2)
# The mean square error
print("MSE:", metrics.mean_squared_error(y, y_pred))
print('Variance score: %.2f' % linreg.score(x2, y))


# 综上，用多项式模型拟合效果最优
