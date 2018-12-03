#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework-3.py
@time:2018/3/4 23:07
"""

# coding=utf-8
import pandas as pd

inputfile = 'f:/data/BHP1.csv'
inputfile1 = 'f:/data/BHP2.xlsx'
data1 = pd.read_csv(inputfile)
data2 = pd.read_excel(inputfile1)
print pd.concat([data2, data1], axis=0)
