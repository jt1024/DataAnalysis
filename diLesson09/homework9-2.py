# !/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework9-2.py
@time:2018/3/25 22:11
"""

import pandas as pd

data = pd.read_excel('D:/DataguruPyhton/diLesson09/Amtrak.xls')
result1 = data.describe()
print result1
#运行结果：
#        NumPassengers          t  log(Numpassengers)
# count      87.000000  87.000000           87.000000
# mean     1878.345241  44.000000            7.532932
# std       188.196648  25.258662            0.104117
# min      1371.690000   1.000000            7.223799
# 25%      1766.749000  22.500000            7.476894
# 50%      1864.852000  44.000000            7.530937
# 75%      2014.837500  65.500000            7.608291
# max      2223.349000  87.000000            7.706770
result2 = data.skew()
print result2
# 运行结果：
# NumPassengers        -0.529459
# t                     0.000000
# log(Numpassengers)   -0.801000
# dtype: float64
result3 = data.kurt()
print result3
# 运行结果：
# NumPassengers        -0.055304
# t                    -1.200000
# log(Numpassengers)    0.528965
# dtype: float64
