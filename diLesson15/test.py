#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:test.py
@time:2018/5/12 21:08
"""

count = 0
while (count <= 9):
    count = count + 1
print(count)


def ex1(num):
    for i in range(2, num - 1):
        if num % i == 0:
            j = num / i
            print('%d = %d * %d' % (num, i, j))
            break
    else:
        print(num, '不是所求')


ex1(20)

import numpy as np

arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr * arr

a = np.arange(9)
print(a[:7:2])

arr = range(10)
print(arr + arr)

import numpy as np

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = [(x if c else y)
          for x, y, c in zip(xarr, yarr, cond)]
result

from numpy.random import randn
arr = randn(4,4)
arr

np.where(arr<0,0,arr)