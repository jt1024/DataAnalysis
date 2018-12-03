#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework-4.py
@time:2018/3/4 23:09
"""

# coding=utf-8
import pandas as pd

inputfile = 'f:/data/BHP1.csv'
inputfile1 = 'f:/data/BHP2.xlsx'
data1 = pd.read_csv(inputfile)
data2 = pd.read_excel(inputfile1)
data3 = pd.concat([data2, data1], axis=0)
group_name = ['low', 'median', 'high']
data3['volume'] = pd.qcut(data3['volume'], 3, labels=group_name)
print data3
