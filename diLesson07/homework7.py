#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework7.py
@time:2018/3/12 7:22
"""
# # 对macrodata.csv数据集
# # 1. 画出realgdp列的直方图
from __future__ import division
import matplotlib.pyplot as plt
import pandas as pd

macrod = pd.read_csv('D:/DataguruPyhton/diLesson07/macrodata.csv')
fig = plt.figure()
macrod['realgdp'].hist(bins=50)
plt.show()

# 2. 画出realgdp列与realcons列的散点图，初步判断两个变量之间的关系
import matplotlib.pyplot as plt

macrod = pd.read_csv('D:/DataguruPyhton/diLesson07/macrodata.csv')
fig = plt.figure();
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title('scatter')
ax2.set_xlabel('realgdp')
ax2.set_ylabel('realcons')
plt.scatter(macrod['realgdp'], macrod['realcons'])
plt.show()
# 初步判断 realgdp 与 realcons 这两个变量呈线性关系


# 对tips数据集
# 3. 画出不同sex与day的交叉表的柱形图
import pandas as pd

tips = pd.read_csv('D:/DataguruPyhton/diLesson07/tips.csv')
party_sex = pd.crosstab(tips.day, tips.sex)
party_pcts = party_sex.div(party_sex.sum(1).astype(float), axis=0)
party_pcts.plot(kind='bar', stacked=True)
plt.show()

# 4. 画出size的饼图
import matplotlib.pyplot as plt
import pandas as pd

tips = pd.read_csv('D:/DataguruPyhton/diLesson07/tips.csv')
plt.figure(1, figsize=(8, 8))
ax = plt.axes([0.1, 0.1, 0.8, 0.8])
labels = '1', '2', '3', '4', '5', '6'
values = []
for i in range(6):
    values.append(sum(tips['size'] == i + 1))
explode = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
plt.pie(values, explode=explode, labels=labels,
        autopct='%1.1f%%', startangle=67)
plt.title('pie of size')
plt.show()
