#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework.py
@time:2018/1/20 19:59
"""
''' + - * / % // ** 分别是：加、减、乘、除、取模、整除、幂运算  '''
a = 2
b = 3

c = a + b
print "c = ", c

c = a - b
print "c = ", c

c = a * b
print "c = ", c

c = a / b
print "c = ", c

c = a % b
print "c = ", c

c = a // b
print "c = ", c

c = a ** b
print "c = ", c

''' / % // 三者的区别分别是：除以、取模、整除  '''
a = 9
b = 2
c = 9.0
d = 2.0

y = a / b
print "y = ", y

y = a % b
print "y = ", y

y = a // b
print "y = ", y

y = c / d
print "y = ", y

y = c % d
print "y = ", y

y = c // d
print "y = ", y

''' 位运算  '''
a = 60  # 0011 1100
b = 13  # 0000 1101
c = 0  # 0000 0000

c = a & b
print "c = ", c  # 12 = 0000 1100

c = a | b
print "c = ", c  # 61 = 0011 1101

c = a ^ b
print "c = ", c  # 49 = 0011 0001

c = ~a
print "c = ", c  # -61 = 1100 0011

c = a << 2
print "c = ", c  # 240 = 1111 0000

c = a >> 2
print "c = ", c  # 15 = 0000 1111

''' 逻辑运算符：and、 or、 not  '''
''' 成员运算符：in、not in  '''
''' 身份运算符：is、is not  '''

