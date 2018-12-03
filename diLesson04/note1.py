#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework.py
@time:2018/2/25 22:51
"""
from __future__ import division
from numpy.random import randn
import numpy as np

## 一、numpy通用函数
arr = np.arange(10)
print(arr)
# result: [0 1 2 3 4 5 6 7 8 9]
print(np.sqrt(arr))
# result: [0.         1.         1.41421356 1.73205081 2.         2.23606798 2.44948974 2.64575131 2.82842712 3.        ]
print(np.exp(arr))
# result:  [1.00000000e+00 2.71828183e+00 7.38905610e+00 2.00855369e+01 5.45981500e+01 1.48413159e+02 4.03428793e+02 1.09663316e+03 2.98095799e+03 8.10308393e+03]

x = randn(8)
y = randn(8)
print(x)
# result: [-0.45319514 -0.33634081 -1.5487594   1.05024164  0.03497043 -0.39700642 -0.9009428   0.93696438]
print(y)
# result: [-0.30578685 -1.41005898  1.55261228 -0.79217705  1.26768881  1.16863963  0.93794544  0.24048242]
print(np.maximum(x, y))  # 元素级最大值
# result: [-0.30578685 -0.33634081  1.55261228  1.05024164  1.26768881  1.16863963  0.93794544  0.93696438]

arr = randn(7) * 5
print(arr)
# result: [ 4.11697208  4.36766248  0.84191189 -5.90854028  4.29542469 -3.91484446  2.38288939]
print(np.modf(arr))
# result: (array([ 0.11697208,  0.36766248,  0.84191189, -0.90854028,  0.29542469,  -0.91484446,  0.38288939]), array([ 4.,  4.,  0., -5.,  4., -3.,  2.]))




## 二、利用数组进行数据处理
points = np.arange(-5, 5, 0.01)  # 1000 equally spaced points
xs, ys = np.meshgrid(points, points)
xs
# result:
# array([[-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],
#        [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],
#        [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],
#        ...,
#        [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],
#        [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99],
#        [-5.  , -4.99, -4.98, ...,  4.97,  4.98,  4.99]])
ys
# result:
# array([[-5.  , -5.  , -5.  , ..., -5.  , -5.  , -5.  ],
#        [-4.99, -4.99, -4.99, ..., -4.99, -4.99, -4.99],
#        [-4.98, -4.98, -4.98, ..., -4.98, -4.98, -4.98],
#        ...,
#        [ 4.97,  4.97,  4.97, ...,  4.97,  4.97,  4.97],
#        [ 4.98,  4.98,  4.98, ...,  4.98,  4.98,  4.98],
#        [ 4.99,  4.99,  4.99, ...,  4.99,  4.99,  4.99]])


import matplotlib.pyplot as plt

z = np.sqrt(xs ** 2 + ys ** 2)
z
# result:
# array([[7.07106781, 7.06400028, 7.05693985, ..., 7.04988652, 7.05693985,
#         7.06400028],
#        [7.06400028, 7.05692568, 7.04985815, ..., 7.04279774, 7.04985815,
#         7.05692568],
#        [7.05693985, 7.04985815, 7.04278354, ..., 7.03571603, 7.04278354,
#         7.04985815],
#        ...,
#        [7.04988652, 7.04279774, 7.03571603, ..., 7.0286414 , 7.03571603,
#         7.04279774],
#        [7.05693985, 7.04985815, 7.04278354, ..., 7.03571603, 7.04278354,
#         7.04985815],
#        [7.06400028, 7.05692568, 7.04985815, ..., 7.04279774, 7.04985815,
#         7.05692568]])

plt.imshow(z, cmap=plt.cm.gray);
plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
plt.draw()
plt.show()

# 将条件逻辑表达为数组运算
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

result = [(x if c else y)
          for x, y, c in zip(xarr, yarr, cond)]
result

result = np.where(cond, xarr, yarr)
result

arr = randn(4, 4)
arr
np.where(arr > 0, 2, -2)
np.where(arr > 0, 2, arr)  # set only positive values to 2

# Not to be executed
result = []
for i in range(n):
    if cond1[i] and cond2[i]:
        result.append(0)
    elif cond1[i]:
        result.append(1)
    elif cond2[i]:
        result.append(2)
    else:
        result.append(3)

# Not to be executed
np.where(cond1 & cond2, 0,
         np.where(cond1, 1,
                  np.where(cond2, 2, 3)))

# Not to be executed
result = 1 * cond1 + 2 * cond2 + 3 * -(cond1 | cond2)

# 数学与统计方法

arr = np.random.randn(5, 4)  # 标准正态分布数据
arr.mean()
np.mean(arr)
arr.sum()

arr.mean(axis=1)
arr.sum(0)

arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr.cumsum(0)
arr.cumprod(1)

# 用于布尔型数组的方法
arr = randn(100)
(arr > 0).sum()  # 正值的数量

bools = np.array([False, False, True, False])
bools.any()
bools.all()

# 排序
arr = randn(8)
arr
arr.sort()
arr

arr = randn(5, 3)
arr
arr.sort(1)
arr

large_arr = randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))]  # 5%分位数

# 唯一化以及其他的集合逻辑
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)

sorted(set(names))

values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])

###线性代数
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x
y
x.dot(y)  # 等价于np.dot(x, y)

np.dot(x, np.ones(3))

np.random.seed(12345)

from numpy.linalg import inv, qr

X = randn(5, 5)
mat = X.T.dot(X)
inv(mat)
mat.dot(inv(mat))
q, r = qr(mat)
r

###随机数生成
samples = np.random.normal(size=(4, 4))
samples

from random import normalvariate

N = 1000000
get_ipython().magic(u'timeit samples = [normalvariate(0, 1) for _ in xrange(N)]')
get_ipython().magic(u'timeit np.random.normal(size=N)')

# 范例：随机漫步
import random

position = 0
walk = [position]
steps = 1000
for i in xrange(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

np.random.seed(12345)

nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()

walk.min()
walk.max()

(np.abs(walk) >= 10).argmax()

# 一次模拟多个随机漫步
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))  # 0 or 1
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks

walks.max()
walks.min()

hits30 = (np.abs(walks) >= 30).any(1)
hits30
hits30.sum()  # 到达30或-30的数量

crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
crossing_times.mean()

steps = np.random.normal(loc=0, scale=0.25,
                         size=(nwalks, nsteps))

