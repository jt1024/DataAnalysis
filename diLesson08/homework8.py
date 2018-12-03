#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework15.py
@time:2018/5/6 12:26
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir(u'D:\DataguruPyhton\DataAnalysis\diLesson08')
tips = pd.read_csv('tips.csv')

# 1
tips['tip'].groupby(tips['time']).mean()
tips['tip'].groupby(tips['time']).std()

# 2
group2 = tips[['total_bill', 'tip']].groupby(tips['sex'])
group2.apply(lambda x: (x - x.mean()) / x.std())

# 3
tips['tip_pct'] = tips['tip'] / tips['total_bill']
group3 = tips['tip_pct'].groupby(tips['smoker'])

group3.mean()['Yes'] - group3.mean()['No']

# 4
group4 = tips.groupby(['sex', 'size'])['tip_pct']
tmp4 = group4.agg(['std', 'mean'])

tips2 = pd.merge(tips, tmp4, left_on=['sex', 'size'], right_index=True)

# 5
tmp5 = tips.groupby(['time', 'size'])[['total_bill']].sum()

plt.pie(tmp5, labels=tmp5.index)
plt.show()
