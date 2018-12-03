#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework02.py
@time:2018/1/28 23:26
"""


# 1. 编写函数，要求输入x与y，返回x和y的平方差
def sq1(x, y):
    return int(x) ** 2 - int(y) ** 2


sq1(raw_input('Please input x:'), raw_input('Please input y:'))


def sq2(x, y):
    return pow(int(x), 2) - pow(int(y), 2)


sq2(raw_input('Please input x:'), raw_input('Please input y:'))


# 2. 计算1到100的平方的和
def sqr(maxNum):
    sum = 0
    for i in range(1, maxNum + 1):
        sum += pow(i, 2)
    return sum


sqr(100)


# 3. 编写函数，若输入为小于100的数，返回TRUE，大于100的数，返回FALSE
def judge100(x):
    if int(x) < 100:
        return True
    elif int(x) > 100:
        return False


judge100(raw_input('please input num:'))


# 4. 某个公司采用公用电话传递数据，数据是四位的整数，在传递过程中是加密的，加密规则如下：
# 每位数字都加上5,然后用和除以10的余数代替该数字，再将第一位和第四位交换，第二位和第三位交换。
# 编写加密的函数与解密的函数。

def encode(y):
    x = str(y)
    index1 = str((x[0] + 5) % 10)
    index2 = str((x[1] + 5) % 10)
    index3 = str((x[2] + 5) % 10)
    index4 = str((x[3] + 5) % 10)
    return index4 + index3 + index2 + index1


def decode(x):
    index1 = x[3]
    index2 = x[2]
    index3 = x[1]
    index4 = x[0]
    return trans(index1) + trans(index2) + trans(index3) + trans(index4)


def trans(i):
    num = int(i)
    if (num >= 5):
        return str(num - 5)
    if (num < 5):
        return str(num + 10 - 5)


def numToStr(x):
    y = str(x)
    return y


numToStr(10)
