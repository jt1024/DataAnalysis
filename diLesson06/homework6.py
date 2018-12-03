#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework6.py
@time:2018/3/4 22:52
"""

import numpy as np
import pandas as pd

inputfile = u'F:/python数据分析/第6周/第6周/第6周作业数据/肝气郁结证型系数.xls'
data = pd.read_Excel(inputfile)
data = data[u'肝气郁结证型系数']
group_name = ['group1', 'group2', 'group3', 'group4', 'group5']
cats = pd.cut(data, 5, labels=group_name)
df = pd.get_dummies(cats)
df1 = df.join(data)
group1 = df1[u'肝气郁结证型系数'].where(df1['group1'] > 0)
group1 = group1[
    group1.notnull()]  # 剔除空值
print '等距group1的标准差：', np.std(group1)
print '等距group1的中位数：', np.median(group1)
group2 = df1[u'肝气郁结证型系数'].where(df1['group2'] > 0)
group2 = group2[
    group2.notnull()]  # 剔除空值
print '等距group2的标准差：', np.std(group2)
print '等距group2的中位数：', np.median(group2)
group3 = df1[u'肝气郁结证型系数'].where(df1['group3'] > 0)
group3 = group3[
    group3.notnull()]  # 剔除空值
print '等距group3的标准差：', np.std(group3)
print '等距group3的中位数：', np.median(group3)
group4 = df1[u'肝气郁结证型系数'].where(df1['group4'] > 0)
group4 = group4[
    group4.notnull()]  # 剔除空值
print '等距group4的标准差：', np.std(group4)
print '等距group4的中位数：', np.median(group4)
group5 = df1[u'肝气郁结证型系数'].where(df1['group5'] > 0)
group5 = group5[
    group5.notnull()]  # 剔除空值
print '等距group5的标准差：', np.std(group5)
print '等距group5的中位数：', np.median(
    group5)  # 小组等量
cats = pd.qcut(data, 5, labels=group_name)
df = pd.get_dummies(cats)
df1 = df.join(data)
group1 = df1[u'肝气郁结证型系数'].where(df1['group1'] > 0)
group1 = group1[
    group1.notnull()]  # 剔除空值
print '小组等量group1的标准差：', np.std(group1)
print '小组等量group1的中位数：', np.median(group1)
group2 = df1[u'肝气郁结证型系数'].where(df1['group2'] > 0)
group2 = group2[
    group2.notnull()]  # 剔除空值
print '小组等量group2的标准差：', np.std(group2)
print '小组等量group2的中位数：', np.median(group2)
group3 = df1[u'肝气郁结证型系数'].where(df1['group3'] > 0)
group3 = group3[
    group3.notnull()]  # 剔除空值
print '小组等量group3的标准差：', np.std(group3)
print '小组等量group3的中位数：', np.median(group3)
group4 = df1[u'肝气郁结证型系数'].where(df1['group4'] > 0)
group4 = group4[
    group4.notnull()]  # 剔除空值
print '小组等量group4的标准差：', np.std(group4)
print '小组等量group4的中位数：', np.median(group4)
group5 = df1[u'肝气郁结证型系数'].where(df1['group5'] > 0)
group5 = group5[group5.notnull()]  # 剔除空值
print '小组等量group5的标准差：', np.std(group5)
print '小组等量group5的中位数：', np.median(group5)
