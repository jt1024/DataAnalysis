#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework5.py
@time:2018/2/25 23:11
"""

import pandas as pd

# 读取DressSales.csv
data_csv = pd.read_csv('D:/DataguruPyhton/diLesson05/data/DressSales.csv')
data_csv

# 读取ApplianceShipments.xls
data_xls = pd.read_excel('D:/DataguruPyhton/diLesson05/data/ApplianceShipments.xls')
data_xls

# 读取creditcard-dataset.txt
data_txt = pd.read_table('D:/DataguruPyhton/diLesson05/data/creditcard-dataset.txt')
data_txt
