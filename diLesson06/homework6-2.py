#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework6-2.py
@time:2018/3/4 23:00
"""

# coding=utf-8
import pandas as pd
import lagrange  # 导入拉格朗日插值函数

inputfile = 'f:/data/BHP1.csv'
data = pd.read_csv(inputfile)
print data  ###缺失值处理——拉格朗日插值法


# 自定义列向量插值函数#s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s, n, k=5):
    y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]  # 取数
    y = y[y.notnull()]  # 剔除空值
    return lagrange(y.index, list(y))(n)  # 插值并返回插值结果  # #逐个元素判断是否需要插值


for i in data.columns:
    for j in range(len(data)):
        if (data.isnull())[j]:  # 如果为空即插值。
            data[j] = ployinterp_column(data, j)
            print data
