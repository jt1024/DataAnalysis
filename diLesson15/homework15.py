#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework15.py
@time:2018/5/6 12:26
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir(u'D:\DataguruPyhton\DataAnalysis\diLesson15')
df = pd.read_table('ex15.txt', sep='\s+', parse_dates=True, encoding='gb2312')
y = df['y']
x = (df.iloc[:, 0:3])

x.corr()

####多元线性回归分析####
linreg = LinearRegression()
linreg.fit(x, y)
print(linreg.intercept_)
# 运行结果：-10.127988155231051
print(linreg.coef_)
# 运行结果：[-0.05139616  0.58694904  0.28684868]
y_pred = linreg.predict(x)
# 误差评估
print("多元线性回归分析的MAE:", metrics.mean_absolute_error(y, y_pred))
# 运行结果：多元线性回归分析的MAE: 0.3233899552716768
print("多元线性回归分析的MSE:", metrics.mean_squared_error(y, y_pred))
# 运行结果：多元线性回归分析的MSE: 0.15208631466626785
print("多元线性回归分析的RMSE:", np.sqrt(metrics.mean_squared_error(y, y_pred)))
# 运行结果：多元线性回归分析的RMSE: 0.38998245430566214
print("多元线性回归分析的得分", linreg.score(x, y))
# 运行结果：多元线性回归分析的得分 0.9918965520557045

####主成分分析####
pca = PCA()
pca.fit(x)
print(pca.explained_variance_ratio_)  # 主成分为1个

pac = PCA(n_components=1)
reduced_x = pac.fit_transform(x)

linreg2 = LinearRegression()
linreg2.fit(reduced_x, y)
print(linreg2.intercept_)
# 运行结果：[0.99676249 0.00209367 0.00114384]
print(linreg2.coef_)
# 运行结果：21.89090909090909
y_pred2 = linreg2.predict(reduced_x)
# 误差评估

print("主成分分析的MAE:", metrics.mean_absolute_error(y, y_pred2))
# 运行结果：主成分分析的MAE: 0.8936859420929342
print("主成分分析的MSE:", metrics.mean_squared_error(y, y_pred2))
# 运行结果：主成分分析的MSE: 1.182503150421625
print("主成分分析的RMSE:", np.sqrt(metrics.mean_squared_error(y, y_pred2)))
# 运行结果：主成分分析的RMSE: 1.087429607111019
print("主成分分析的得分:", linreg2.score(reduced_x, y))
# 运行结果：主成分分析的得分: 0.9369939843408384

#####画图#####
# 作图，比较实际值y，多元线性回归模型预测值y_pred，主成分分析模型的预测值y_pred2进行比较，直观展现模型的准确度
k = range(1, 12)
plt.scatter(k, y, c='r', marker='x')
plt.scatter(k, y_pred, c='b', marker='D')
plt.scatter(k, y_pred2, c='g', marker='.')
plt.show()
