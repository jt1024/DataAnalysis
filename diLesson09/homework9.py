#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework9.py
@time:2018/3/25 22:06
"""
from pandas import DataFrame, Series
from scipy import stats as ss

df = DataFrame({'data': [3.2, 3.3, 3.0, 3.7, 3.5, 4.0, 3.2, 4.1, 2.9, 3.3]})
result = ss.ttest_1samp(a=df, popmean=3.5)
print result
# 运行结果：Ttest_1sampResult(statistic=array([-0.6289709]), pvalue=array([ 0.54499936]))
# 因为p_value>0.5 所以判定这些糖的包装平均重量为3.5克
