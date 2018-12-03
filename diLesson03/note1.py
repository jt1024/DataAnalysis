#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework.py
@time:2018/2/25 22:50
"""


## 一、初识numpy的强大之处
# 1、向量相加-Python
def pythonsum(n):
    a = list(range(n))
    b = list(range(n))
    c = []
    for i in range(len(a)):
        a[i] = i ** 2
        b[i] = i ** 3
        c.append(a[i] + b[i])
    return c


# 2、向量相加-NumPy
import numpy as np


def numpysum(n):
    a = np.arange(n) ** 2
    b = np.arange(n) ** 3
    c = a + b
    return c


# 3、效率比较
from datetime import datetime

size = 1000

start = datetime.now()
c = pythonsum(size)
delta = datetime.now() - start
print("The last 2 elements of the sum", c[-2:])
print("PythonSum elapsed time in microseconds", delta.microseconds)

start = datetime.now()
c = numpysum(size)
delta = datetime.now() - start
print("The last 2 elements of the sum", c[-2:])
print("NumPySum elapsed time in microseconds", delta.microseconds)

# result：
# The last 2 elements of the sum [995007996, 998001000]
# PythonSum elapsed time in microseconds 1002
# The last 2 elements of the sum [995007996 998001000]
# NumPySum elapsed time in microseconds 0


## 二、numpy创建数组
arr0 = np.array([2, 3, 4])  # 通过列表创建一维数组
print(arr0)
# result：[2 3 4]

arr1 = np.array([[1, 2], [3, 4]])  # 通过列表创建二维数组
print(arr1)
# result：
# [[1 2]
#  [3 4]]

arr2 = np.array([(1.3, 9, 2.0), (7, 6, 1)])  # 通过元组创建数组
print(arr2)
# result：
# [[1.3 9.  2. ]
#  [7.  6.  1. ]]

arr3 = np.zeros((2, 3))  # 通过元组(2, 3)生成全零矩阵
print(arr3)
# result：
# [[0. 0. 0.]
#  [0. 0. 0.]]

arr4 = np.identity(3)  # 生成3维的单位矩阵
print(arr4)
# result：
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

arr5 = np.random.random(size=(2, 3))  # 生成每个元素都在[0,1]之间的随机矩阵
print(arr5)
# result：
# [[0.71850624 0.96298612 0.1684721 ]
#  [0.77269464 0.12644678 0.701804  ]]

arr6 = np.arange(5, 20, 3)  # 生成等距序列,参数为起点,终点,步长值.含起点值，不含终点值
print(arr6)
# result：[ 5  8 11 14 17]

arr7 = np.linspace(0, 2, 9)  # 生成等距序列,参数为起点,终点,步长值.含起点值和终点值
print(arr7)
# result：[0.   0.25 0.5  0.75 1.   1.25 1.5  1.75 2.  ]


## 三、numpy访问数组
# 1、查看数组的属性
arr2 = np.array([(1.3, 9, 2.0), (7, 6, 1)])  # 通过元组创建数组
print(arr2.shape)  # 返回矩阵的规格
# result: (2,3)
print(arr2.ndim)  # 返回矩阵的秩
# result: 2
print(arr2.size)  # 返回矩阵元素总数
# result: 6
print(arr2.itemsize) #
# result: 8
print(arr2.nbytes) #
# result: 48
print(arr2.dtype.name)  # 返回矩阵元素的数据类型
# result: float64
print(type(arr2))  # 查看整个数组对象的类型
# result： <class 'numpy.ndarray'>

b = np.array([1. + 1.j, 3. + 2.j])
b.real
# result：array([1., 3.])
b.imag
# result：array([1., 2.])

b = np.arange(4).reshape(2, 2)
b.flat
# result：<numpy.flatiter object at 0x0000024FF4C32CF0>
b.flat[2]
# result：2


# 2、通过索引和切片访问数组元素
def f(x, y):
    return 10 * x + y

arr8 = np.fromfunction(f, (4, 3), dtype=int)
print(arr8)
# result:
# [[ 0  1  2]
#  [10 11 12]
#  [20 21 22]
#  [30 31 32]]

print(arr8[1, 2])  # 返回矩阵第1行，第2列的元素（注意下标从0开始）
# result: 12
print(arr8[0:2, :])  # 切片，返回矩阵前2行
# result:
# [[ 0  1  2]
#  [10 11 12]]
print(arr8[:, 1])  # 切片，返回矩阵第1列
# result: [ 1 11 21 31]
print(arr8[-1])  # 切片，返回矩阵最后一行
# reuslt: [30 31 32]
print(arr8[::-1])
# reuslt:
# [[30 31 32]
#  [20 21 22]
#  [10 11 12]
#  [ 0  1  2]]


# 3、通过迭代器访问数组元素
for row in arr8:
    print(row)
# result:
# [0 1 2]
# [10 11 12]
# [20 21 22]
# [30 31 32]
for element in arr8.flat:
    print(element)
# 输出矩阵全部元素
# result:
# 0
# 1
# 2
# 10
# 11
# 12
# 20
# 21
# 22
# 30
# 31
# 32
print('-' * 70)


## 四、numpy数据类型
# 1、基本数据类型
print("float64(42)=", np.float64(42))
# result：float64(42)= 42.0
print("int8(42.0)=", np.int8(42.0))
# result：int8(42.0)= 42
print("bool(42)=", np.bool(42))
# result：bool(42)= True
print("bool(0)=", np.bool(0))
# result：bool(0)= False
print("bool(42.0)=", np.bool(42.0))
# result： bool(42.0)= True
print("int8(True)=", np.int8(True))
# result：int8(True)= 1
print("int8(False)=", np.int8(False))
# result：int8(False)= 0
print("float(True)=", np.float(True))
# result：float(True)= 1.0
print("float(False)=", np.float(False))
# result：float(False)= 0.0
print("arange(7, dtype=uint16)=", np.arange(7, dtype=np.uint16))
# result：arange(7, dtype=uint16)= [0 1 2 3 4 5 6]

# 2、numpy中的元素要保持同一类型
print("int(42.0 + 1.j)=")
try:
    print(np.int(42.0 + 1.j))
except TypeError:
    print("TypeError")

# result：
# int(42.0 + 1.j)=
# TypeError

print("float(42.0 + 1.j)=", float(42.0 + 1.j))
# result：TypeError: can't convert complex to float


# 3、数据类型转换——int转float
arr = np.array([1, 2, 3, 4, 5])
arr.dtype
# result：dtype('int32')
print(arr)
# result：[1 2 3 4 5]
float_arr = arr.astype(np.float64)
float_arr.dtype
# result：dtype('float64')
print(float_arr)
# result：[ 1.  2.  3.  4.  5.]

# 4、数据类型转换——float转int，结果不会四舍五入，只会直接去掉小数点后的部分
arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr.dtype
# result：dtype('float64')
print(arr)
# result：[  3.7  -1.2  -2.6   0.5  12.9  10.1]
int_arr = arr.astype(np.int32)
int_arr.dtype
# result：dtype('int32')
print(int_arr)
# result：array([ 3, -1, -2,  0, 12, 10])

# 5、数据类型转换——string转float
numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
numeric_strings.dtype
# result: dtype('S4')
print(numeric_strings)
# result：[b'1.25' b'-9.6' b'42']
float_arr = numeric_strings.astype(float)
float_arr.dtype
# result：dtype('float64')
print(float_arr)
# result：[ 1.25 -9.6  42.  ]

# 6、字符编码
print(np.arange(7, dtype='f'))
# result：[ 0.  1.  2.  3.  4.  5.  6.]
print(np.arange(7, dtype='d'))
# result：[ 0.  1.  2.  3.  4.  5.  6.]
print(np.arange(7, dtype='D'))
# result：[ 0.+0.j  1.+0.j  2.+0.j  3.+0.j  4.+0.j  5.+0.j  6.+0.j]
print(np.dtype(float))
# result：float64
print(np.dtype('f'))
# result：float32
print(np.dtype('d'))
# result：float64
print(np.dtype('D'))
# result：complex128
print(np.dtype('f8'))
# result：float64
print(np.dtype('float64'))
# result：float64

# 7、创建自定义数据类型
t = np.dtype([('name', np.str_, 40), ('numitems', np.int32), ('price', np.float32)])
print(t)
# result：[('name', '<U40'), ('numitems', '<i4'), ('price', '<f4')]
print(t['name'])
# result：<U40
itemz = np.array([('Meaning of life DVD', 42, 3.14), ('Butter', 13, 2.72)], dtype=t)
print(itemz[1])
# result：('Butter', 13, 2.72)



## 五、numpy数组的运算
arr9 = np.array([[2,1],[1,2]])
arr10 = np.array([[1,2],[3,4]])
print(arr9 - arr10)
# result:
# [[ 1 -1]
#  [-2 -2]]
print(arr9**2)
# result:
# [[4 1]
#  [1 4]]
print(3*arr10)
# result:
# [[ 3  6]
#  [ 9 12]]
print(arr9*arr10)  #普通乘法
# result：
# [[2 2]
#  [3 8]]
print(np.dot(arr9,arr10))  #矩阵乘法
# result:
# [[ 5  8]
#  [ 7 10]]
print(arr10.T)  #转置
# result:
# [[1 3]
#  [2 4]]
print(np.linalg.inv(arr10)) #返回逆矩阵
# result:
# [[-2.   1. ]
#  [ 1.5 -0.5]]
print(arr10.sum())  #数组元素求和
# result: 10
print(arr10.max())  #返回数组最大元素
# result: 4
print(arr10.cumsum(axis = 1))  #沿行累计总和
# result：
# [[1 3]
#  [3 7]]





## 六、numpy数组的索引与切片

# 1、一维数组的索引与切片
a = np.arange(9)
print(a)
# result:[0 1 2 3 4 5 6 7 8]
print(a[3:7])
# result:[3 4 5 6]
print(a[:7:2])
# result:[0 2 4 6]
print(a[::-1])
# result:[8 7 6 5 4 3 2 1 0]
s = slice(3, 7, 2)
print(a[s])
# result:[3 5]
s = slice(None, None, -1)
print(a[s])
# result:[8 7 6 5 4 3 2 1 0]


# 2、多维数组的切片与索引
b = np.arange(24).reshape(2, 3, 4)
print(b)
# result:
# [[[ 0  1  2  3]
#   [ 4  5  6  7]
#   [ 8  9 10 11]]
#  [[12 13 14 15]
#   [16 17 18 19]
#   [20 21 22 23]]]

print(b.shape)
# result: (2, 3, 4)
print(b[0, 0, 0])
# result: 0
print(b[:, 0, 0])
# result: [ 0 12]
print(b[0])
# result:
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
print(b[0, :, :])
# result:
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
print(b[0, ...])
# result:
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
print(b[0, 1])
# result: [4 5 6 7]
print(b[0, 1, ::2])
# result: [4 6]
print(b[..., 1])
# result:
# [[ 1  5  9]
#  [13 17 21]]
print(b[:, 1])
# result:
# [[ 4  5  6  7]
#  [16 17 18 19]]
print(b[0, :, 1])
# result: [1 5 9]
print(b[0, :, -1])
# result: [ 3  7 11]
print(b[0, ::-1, -1])
# result: [11  7  3]
print(b[0, ::2, -1])
# result: [ 3 11]
print(b[::-1])
# result:
# [[[12 13 14 15]
#   [16 17 18 19]
#   [20 21 22 23]]
#  [[ 0  1  2  3]
#   [ 4  5  6  7]
#   [ 8  9 10 11]]]
s = slice(None, None, -1)
print(b[(s, s, s)])
# result:
# [[[23 22 21 20]
#   [19 18 17 16]
#   [15 14 13 12]]
#  [[11 10  9  8]
#   [ 7  6  5  4]
#   [ 3  2  1  0]]]

# 3、布尔型索引
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)

names
# result: array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'], dtype='<U4')
data
# result:
# array([[-0.1361622 ,  0.98050442, -0.15336564, -0.48760426],
#        [-0.16386874, -0.34817134,  1.1479974 ,  0.70937066],
#        [-1.28186917, -0.48926723,  0.50318499,  0.50662251],
#        [-0.4856206 ,  0.03109503,  0.68854417,  0.14022486],
#        [ 0.00771609,  0.12883784, -1.39586023, -0.10152562],
#        [ 0.60160016, -0.66399474,  1.20907536,  0.95262856],
#        [-1.86564241, -0.59888113, -1.23990637,  0.18873638]])

names == 'Bob'
# result: array([ True, False, False,  True, False, False, False])
data[names == 'Bob']
# result:
# array([[-0.1361622 ,  0.98050442, -0.15336564, -0.48760426],
#        [-0.4856206 ,  0.03109503,  0.68854417,  0.14022486]])
data[names == 'Bob', 2:]
# result:
# array([[-0.15336564, -0.48760426],
#        [ 0.68854417,  0.14022486]])
data[names == 'Bob', 3]
# result: array([-0.48760426,  0.14022486])
names != 'Bob'
# result: array([False,  True,  True, False,  True,  True,  True])
data[~(names == 'Bob')]
# result:
# array([[-0.16386874, -0.34817134,  1.1479974 ,  0.70937066],
#        [-1.28186917, -0.48926723,  0.50318499,  0.50662251],
#        [ 0.00771609,  0.12883784, -1.39586023, -0.10152562],
#        [ 0.60160016, -0.66399474,  1.20907536,  0.95262856],
#        [-1.86564241, -0.59888113, -1.23990637,  0.18873638]])
mask = (names == 'Bob') | (names == 'Will')
mask
# result: array([ True, False,  True,  True,  True, False, False])
data[mask]
# result:
# array([[-0.1361622 ,  0.98050442, -0.15336564, -0.48760426],
#        [-1.28186917, -0.48926723,  0.50318499,  0.50662251],
#        [-0.4856206 ,  0.03109503,  0.68854417,  0.14022486],
#        [ 0.00771609,  0.12883784, -1.39586023, -0.10152562]])
data[data < 0] = 0
data
# result:
# array([[0.        , 0.98050442, 0.        , 0.        ],
#        [0.        , 0.        , 1.1479974 , 0.70937066],
#        [0.        , 0.        , 0.50318499, 0.50662251],
#        [0.        , 0.03109503, 0.68854417, 0.14022486],
#        [0.00771609, 0.12883784, 0.        , 0.        ],
#        [0.60160016, 0.        , 1.20907536, 0.95262856],
#        [0.        , 0.        , 0.        , 0.18873638]])
data[names != 'Joe'] = 7
data
# result:
# array([[7.        , 7.        , 7.        , 7.        ],
#        [0.        , 0.        , 1.1479974 , 0.70937066],
#        [7.        , 7.        , 7.        , 7.        ],
#        [7.        , 7.        , 7.        , 7.        ],
#        [7.        , 7.        , 7.        , 7.        ],
#        [0.60160016, 0.        , 1.20907536, 0.95262856],
#        [0.        , 0.        , 0.        , 0.18873638]])


# 4、花式索引
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr
# result:
# array([[0., 0., 0., 0.],
#        [1., 1., 1., 1.],
#        [2., 2., 2., 2.],
#        [3., 3., 3., 3.],
#        [4., 4., 4., 4.],
#        [5., 5., 5., 5.],
#        [6., 6., 6., 6.],
#        [7., 7., 7., 7.]])
arr[[4, 3, 0, 6]]
# result:
# array([[4., 4., 4., 4.],
#        [3., 3., 3., 3.],
#        [0., 0., 0., 0.],
#        [6., 6., 6., 6.]])
arr[[-3, -5, -7]]
# result:
# array([[5., 5., 5., 5.],
#        [3., 3., 3., 3.],
#        [1., 1., 1., 1.]])
arr = np.arange(32).reshape((8, 4))
arr
# result:
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11],
#        [12, 13, 14, 15],
#        [16, 17, 18, 19],
#        [20, 21, 22, 23],
#        [24, 25, 26, 27],
#        [28, 29, 30, 31]])
arr[[1, 5, 7, 2], [0, 3, 1, 2]]
# result: array([ 4, 23, 29, 10])
arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]
# result:
# array([[ 4,  7,  5,  6],
#        [20, 23, 21, 22],
#        [28, 31, 29, 30],
#        [ 8, 11,  9, 10]])
arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]
# result:
# array([[ 4,  7,  5,  6],
#        [20, 23, 21, 22],
#        [28, 31, 29, 30],
#        [ 8, 11,  9, 10]])



## 七、 改变数组的维度
b = np.arange(24).reshape(2, 3, 4)
print(b)
# result:
# [[[ 0  1  2  3]
#   [ 4  5  6  7]
#   [ 8  9 10 11]]
#  [[12 13 14 15]
#   [16 17 18 19]
#   [20 21 22 23]]]
print(b.ravel())
# result: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
print(b.flatten())
# result: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]

b.shape = (6, 4)
print(b)
# result:
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]
#  [16 17 18 19]
#  [20 21 22 23]]
print(b.transpose())
# result:
# [[ 0  4  8 12 16 20]
#  [ 1  5  9 13 17 21]
#  [ 2  6 10 14 18 22]
#  [ 3  7 11 15 19 23]]
b.resize((2, 12))
print(b)
# result:
# [[ 0  1  2  3  4  5  6  7  8  9 10 11]
#  [12 13 14 15 16 17 18 19 20 21 22 23]]




## 八、数组的合并
a = np.arange(9).reshape(3, 3)
print(a)
# result:
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]
b = 2 * a
print(b)
# result:
# [[ 0  2  4]
#  [ 6  8 10]
#  [12 14 16]]

print(np.vstack((a, b))) #纵向合并数组，由于与堆栈类似，故命名为vstack
# result:
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 0  2  4]
#  [ 6  8 10]
#  [12 14 16]]
print(np.concatenate((a, b), axis=0))
# result:
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 0  2  4]
#  [ 6  8 10]
#  [12 14 16]]
print(np.hstack((a, b))) #横向合并数组
# result:
# [[ 0  1  2  0  2  4]
#  [ 3  4  5  6  8 10]
#  [ 6  7  8 12 14 16]]
print(np.concatenate((a, b), axis=1))
# result:
# [[ 0  1  2  0  2  4]
#  [ 3  4  5  6  8 10]
#  [ 6  7  8 12 14 16]]

print(np.dstack((a, b)))
# result:
# [[[ 0  0]
#   [ 1  2]
#   [ 2  4]]
#  [[ 3  6]
#   [ 4  8]
#   [ 5 10]]
#  [[ 6 12]
#   [ 7 14]
#   [ 8 16]]]
oned = np.arange(2)
print(oned)
# result: [0 1]
twice_oned = 2 * oned
print(twice_oned)
# result: [0 2]
print(np.column_stack((oned, twice_oned)))
# result:
# [[0 0]
#  [1 2]]
print(np.column_stack((a, b)))
# result:
# [[ 0  1  2  0  2  4]
#  [ 3  4  5  6  8 10]
#  [ 6  7  8 12 14 16]]
print(np.column_stack((a, b)) == np.hstack((a, b)))
# result:
# [[ True  True  True  True  True  True]
#  [ True  True  True  True  True  True]
#  [ True  True  True  True  True  True]]
print(np.row_stack((oned, twice_oned)))
# result:
# [[0 1]
#  [0 2]]
print(np.row_stack((a, b)))
# result:
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 0  2  4]
#  [ 6  8 10]
#  [12 14 16]]
print(np.row_stack((a, b)) == np.vstack((a, b)))
# result:
# [[ True  True  True]
#  [ True  True  True]
#  [ True  True  True]
#  [ True  True  True]
#  [ True  True  True]
#  [ True  True  True]]


## 九、数组的分割
a = np.arange(9).reshape(3, 3)
print(a)
# result:
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]

print(np.hsplit(a, 3)) # 将数组横向分为2部分
# result:
# [array([[0],
#        [3],
#        [6]]),
#  array([[1],
#        [4],
#        [7]]),
#  array([[2],
#        [5],
#        [8]])]
print(np.split(a, 3, axis=1))
# result:
# [array([[0],
#        [3],
#        [6]]),
#  array([[1],
#        [4],
#        [7]]),
#  array([[2],
#        [5],
#        [8]])]
print(np.vsplit(a, 3)) # 数组纵向分为2部分
# result: [array([[0, 1, 2]]), array([[3, 4, 5]]), array([[6, 7, 8]])]
print(np.split(a, 3, axis=0))
# result: [array([[0, 1, 2]]), array([[3, 4, 5]]), array([[6, 7, 8]])]
c = np.arange(27).reshape(3, 3, 3)
print(c)
# result:
# [[[ 0  1  2]
#   [ 3  4  5]
#   [ 6  7  8]]
#  [[ 9 10 11]
#   [12 13 14]
#   [15 16 17]]
#  [[18 19 20]
#   [21 22 23]
#   [24 25 26]]]
print(np.dsplit(c, 3))
# result:
# [array([[[ 0],
#         [ 3],
#         [ 6]],
#        [[ 9],
#         [12],
#         [15]],
#        [[18],
#         [21],
#         [24]]]),
#  array([[[ 1],
#         [ 4],
#         [ 7]],
#        [[10],
#         [13],
#         [16]],
#        [[19],
#         [22],
#         [25]]]),
#  array([[[ 2],
#         [ 5],
#         [ 8]],
#        [[11],
#         [14],
#         [17]],
#        [[20],
#         [23],
#         [26]]])]



# 十、数组的转换
b = np.array([1. + 1.j, 3. + 2.j])
print(b)
# result: [1.+1.j 3.+2.j]
print(b.tolist())
# result: [(1+1j), (3+2j)]
print(b.tostring())
# result: b'\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x00@'
print(np.fromstring(b'\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x00@', dtype=complex))
# result: [1.+1.j 3.+2.j]
print(np.fromstring('20:42:52', sep=':', dtype=int))
# result: [20 42 52]
print(b.astype(int))
# result: [1 3]
# <string>:1: ComplexWarning: Casting complex values to real discards the imaginary part
print(b.astype('complex'))
# result: [1.+1.j 3.+2.j]

