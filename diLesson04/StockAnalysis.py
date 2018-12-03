#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:StockAnalysis.py
@time:2018/4/7 13:20
@description: 利用Numpy进行股票分析
"""

## 一、计算平均价格和极值、极值波动范围
import numpy as np

# 读入文件 参数1-源文件路径；参数2-分割符号；参数3-读取的列数下标（第7列是收盘价格，第8列是成交量）；参数4-是否分开放置数据
c,v = np.loadtxt('D:\\DataguruPyhton\\diLesson04\\AppleStock.csv', delimiter=',',usecols=(6,7),unpack=True)
print(c)
# result:
# [336.1  339.32 345.03 344.32 343.44 346.5  351.88 355.2  358.16 354.54
#  356.85 359.18 359.9  363.13 358.3  350.56 338.61 342.62 342.88 348.16
#  353.21 349.31 352.12 359.56 360.   355.36 355.76 352.47 346.67 351.99]
print(v)
# result:
# [21144800. 13473000. 15236800.  9242600. 14064100. 11494200. 17322100.
#  13608500. 17240800. 33162400. 13127500. 11086200. 10149000. 17184100.
#  18949000. 29144500. 31162200. 23994700. 17853500. 13572000. 14395400.
#  16290300. 21521000. 17885200. 16188000. 19504300. 12718000. 16192700.
#  18138800. 16824200.]

# 1.1、计算成交量的加权平均价格
vwap = np.average(c,weights=v)
print('vwap=',vwap)
# result: vwap= 350.5895493532009

# 1.2、计算收盘的算数平均价格
mean = np.mean(c)
print('mean=',mean)
# result: mean= 351.0376666666667

# 1.3、计算收盘的加权平均价格（时间越靠近现在权重越大）
t = np.arange(len(c))
twap = np.average(c,weights=t)
print('twap=',twap)
# result: twap= 352.4283218390804

# 1.4、寻找最大值和最小值
h,l = np.loadtxt('D:\\DataguruPyhton\\diLesson04\\AppleStock.csv', delimiter=',',usecols=(4,5),unpack=True)
print(h)
# result:
# [344.4  340.04 345.65 345.25 344.24 346.7  353.25 355.52 359.   360.
#  357.8  359.48 359.97 364.9  360.27 359.5  345.4  344.64 345.15 348.43
#  355.05 355.72 354.35 359.79 360.29 361.67 357.4  354.76 349.77 352.32]
print(l)
# result:
# [333.53 334.3  340.98 343.55 338.55 343.51 347.64 352.15 354.87 348.
#  353.54 356.71 357.55 360.5  356.52 349.52 337.72 338.61 338.37 344.8
#  351.12 347.68 348.4  355.92 357.75 351.31 352.25 350.6  344.9  345.  ]

heighest = np.max(h)
print('heighest =',heighest)
# result: heighest = 364.9
lowest = np.min(l)
print('lowest =',lowest)
# result: lowest = 333.53
average = (heighest+lowest)/2
print('average = ',average)
# result: average =  349.215

# 1.5、计算最大值的波动范围和最小值的波动范围
print('Spread high price =',np.ptp(h))
# result: Spread high price = 24.859999999999957
print('Spread low price =',np.ptp(l))
# result: Spread low price = 26.970000000000027




## 二、对股票进行统计分析：计算中位数和方差
c = np.loadtxt('D:\\DataguruPyhton\\diLesson04\\AppleStock.csv', delimiter=',',usecols=(6,),unpack=True)
print(c)
# result:
# [336.1  339.32 345.03 344.32 343.44 346.5  351.88 355.2  358.16 354.54
#  356.85 359.18 359.9  363.13 358.3  350.56 338.61 342.62 342.88 348.16
#  353.21 349.31 352.12 359.56 360.   355.36 355.76 352.47 346.67 351.99]

# 2.1 计算中位数
# 计算中位数方法一
median = np.median(c)
print('median = ',median)
# result: median =  352.055
# 计算中位数方法二：先排序，再取中间的数
sorted = np.msort(c)
print('sorted = ',sorted)
# result:
# sorted =  [336.1  338.61 339.32 342.62 342.88 343.44 344.32 345.03 346.5  346.67
#  348.16 349.31 350.56 351.88 351.99 352.12 352.47 353.21 354.54 355.2
#  355.36 355.76 356.85 358.16 358.3  359.18 359.56 359.9  360.   363.13]
N = len(c)
middle = sorted[int((N-1)/2)]
print('middle = ',middle)
# result: middle =  351.99
average_middle = (sorted[int(N/2)] + sorted[int((N-1)/2)])/2
print('average_middle = ',average_middle)
# result: average_middle =  352.055

# 2.2 计算方差
# 计算方差方法一
variance = np.var(c)
print('variance = ',variance)
# result: variance =  50.126517888888884
# 计算方差方法二
variance_from_definition = np.mean((c-c.mean())**2)
print('variance_from_definition = ',variance_from_definition)
# result: variance_from_definition =  50.126517888888884




## 三、计算股票收益率：普通收益率和对数收益率以及收益波动率
c = np.loadtxt('D:\\DataguruPyhton\\diLesson04\\AppleStock.csv', delimiter=',',usecols=(6,),unpack=True)
print(c)
# result:
#  [336.1  339.32 345.03 344.32 343.44 346.5  351.88 355.2  358.16 354.54
#  356.85 359.18 359.9  363.13 358.3  350.56 338.61 342.62 342.88 348.16
#  353.21 349.31 352.12 359.56 360.   355.36 355.76 352.47 346.67 351.99]
diff = np.diff(c) # diff函数就是执行的是后一个元素减去前一个元素
print('diff =',diff)
# result:
# diff = [  3.22   5.71  -0.71  -0.88   3.06   5.38   3.32   2.96  -3.62   2.31
#    2.33   0.72   3.23  -4.83  -7.74 -11.95   4.01   0.26   5.28   5.05
#   -3.9    2.81   7.44   0.44  -4.64   0.4   -3.29  -5.8    5.32]

# 3.1、计算普通收益率
returns = diff/c[:-1]
print('returns =',returns)
# result:
# returns = [ 0.00958048  0.01682777 -0.00205779 -0.00255576  0.00890985  0.0155267
#   0.00943503  0.00833333 -0.01010721  0.00651548  0.00652935  0.00200457
#   0.00897472 -0.01330102 -0.02160201 -0.03408832  0.01184253  0.00075886
#   0.01539897  0.01450483 -0.01104159  0.00804443  0.02112916  0.00122372
#  -0.01288889  0.00112562 -0.00924781 -0.0164553   0.01534601]
# 计算标准差
standard_deviation = np.std(returns)
print('standard_deviation =',standard_deviation)
# result: standard_deviation = 0.012922134436826306

# 3.2、计算对数收益率
logreturns = np.diff(np.log(c))
print('logreturns =',logreturns)
# result:
# logreturns = [ 0.00953488  0.01668775 -0.00205991 -0.00255903  0.00887039  0.01540739
#   0.0093908   0.0082988  -0.01015864  0.00649435  0.00650813  0.00200256
#   0.00893468 -0.01339027 -0.02183875 -0.03468287  0.01177296  0.00075857
#   0.01528161  0.01440064 -0.011103    0.00801225  0.02090904  0.00122297
#  -0.01297267  0.00112499 -0.00929083 -0.01659219  0.01522945]

# 3.3、找出正收益率的日子
posretindices = np.where(returns>0)
print('Indices with postive returns =',posretindices)
# result:
# Indices with postive returns = (array([ 0,  1,  4,  5,  6,  7,  9, 10, 11, 12, 16, 17, 18, 19, 21, 22, 23, 25, 28], dtype=int64),)

# 3.4、计算价格年度波动率
annual_volatility = np.std(logreturns)/np.mean(logreturns)
annual_volatility = annual_volatility/np.sqrt(1./252.) # 252是一年的交易日总和
print('Annual_volatility =',annual_volatility)
# result: Annual_volatility = 129.27478991115132

# 3.5、计算价格月度波动率
monthly_volatility = annual_volatility*np.sqrt(1./12.)
print('monthly_volatility =',monthly_volatility)
# result: monthly_volatility = 37.318417377317765




## 四、对股票进行日期分析
from datetime import datetime

# Monday 0; Tuesday 1; Wednesday 2; Thursday 3; Friday 4; Saturday 5; Sunday 6
def datestr2num(s):
    # return datetime.strptime(s,'%d-%m-%Y').date().weekday()
    return datetime.strptime(s.decode('ascii'),'%d-%m-%Y').date().weekday()

dates, close = np.loadtxt('D:\\DataguruPyhton\\diLesson04\\AppleStock.csv', delimiter=',',usecols=(1,6),converters={1:datestr2num},unpack=True)
print('Dates =',dates)
# result:
# Dates = [4. 0. 1. 2. 3. 4. 0. 1. 2. 3. 4. 0. 1. 2. 3. 4. 1. 2. 3. 4. 0. 1. 2. 3. 4. 0. 1. 2. 3. 4.]

# 4.1、计算每周几的收盘价和平均股价
averages = np.zeros(5)
for i in range(5):
    indices = np.where(dates == i)
    prices = np.take(close, indices)
    avg = np.mean(prices)
    print('Day',i,'price',prices,'Average',avg)
    averages[i] = avg
# result:
# Day 0 price [[339.32 351.88 359.18 353.21 355.36]] Average 351.7900000000001
# Day 1 price [[345.03 355.2  359.9  338.61 349.31 355.76]] Average 350.63500000000005
# Day 2 price [[344.32 358.16 363.13 342.62 352.12 352.47]] Average 352.1366666666666
# Day 3 price [[343.44 354.54 358.3  342.88 359.56 346.67]] Average 350.8983333333333
# Day 4 price [[336.1  346.5  356.85 350.56 348.16 360.   351.99]] Average 350.0228571428571

# 4.2、每周中收盘价最高的日期
top = np.max(averages)
print('Highest averages',top)
# result: Highest averages 352.1366666666666
print('Top day of the week',np.argmax(averages))
# result: Top day of the week 2

# 4.3、每周中收盘价最低的日期
bottom = np.min(averages)
print('Lowest averages',bottom)
# result: Lowest averages 350.0228571428571
print('Top day of the week',np.argmin(averages))
# result: Top day of the week 4




## 五、对股票进行周汇总
from datetime import datetime
import numpy as np

# Monday 0; Tuesday 1; Wednesday 2; Thursday 3; Friday 4; Saturday 5; Sunday 6
def datestr2num(s):
    # return datetime.strptime(s,'%d-%m-%Y').date().weekday()
    return datetime.strptime(s.decode('ascii'),'%d-%m-%Y').date().weekday()

dates, open, hight, low, close = np.loadtxt('D:\\DataguruPyhton\\diLesson04\\AppleStock.csv', delimiter=',',usecols=(1,3,4,5,6),converters={1:datestr2num},unpack=True)
# 数据中包含2月21日，周一，总统纪念日，节假日休市，数据的第一天是周五，也不太好处理，因此本例中暂时不考虑这两天，排除这两天，我们只考虑前三周的数据
close = close[:16] # 提取前三周数据
dates = dates[:16] # 提取前三周数据

# 找到第一个周一
first_monday = np.ravel(np.where(dates==0))[0] # where返回的是一个多维数组，所以最后要[0]
print('The first Monday index is',first_monday)
# result: The first Monday index is 1
# 找到最后一个周五
last_friday = np.ravel(np.where(dates==4))[-1]
print('The last Friday index is',last_friday)
# result: The last Friday index is 15
# 创建一个数组，组成三周里每一天的一个索引值
weeks_indices = np.arange(first_monday,last_friday+1)
print('Weeks indices initial',weeks_indices)
# result: Weeks indices initial [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
# 编制3个数组，每个数组里面有5个元素
weeks_indices = np.split(weeks_indices,3)
print('Weeks indices after spilt',weeks_indices)
# result: Weeks indices after spilt [array([1, 2, 3, 4, 5], dtype=int64), array([ 6,  7,  8,  9, 10], dtype=int64), array([11, 12, 13, 14, 15], dtype=int64)]
def summarize(all,o,h,l,c):
    monday_open = o[all[0]]
    week_high = np.max(np.take(h,all))
    week_low = np.min(np.take(l,all))
    friday_close = c[all[-1]]
    return ('Apple',monday_open,week_high,week_low,friday_close)
weeksummary = np.apply_along_axis(summarize, 1, weeks_indices, open, hight, low, close)
print('Week summary',weeksummary)
# result:
# Week summary [['Apple' '335.8' '346.7' '334.3' '346.5']
#  ['Apple' '347.8' '360.0' '347.6' '356.8']
#  ['Apple' '356.7' '364.9' '349.5' '350.5']]

# 保存数据到本地磁盘
np.savetxt('D:\\DataguruPyhton\\diLesson04\\stockAnalysisWeeksummary.csv',weeksummary,delimiter=',',fmt='%s')




## 六、计算股票的真实波动幅度均值（ATR）
N = 20 # ATR 通常取20个交易日
h, l, c = np.loadtxt('D:\\DataguruPyhton\\diLesson04\\AppleStock.csv', delimiter=',',usecols=(4,5,6),converters={1:datestr2num},unpack=True)
h = h[-N:]
l = l[-N:]
print('len(h)=',len(h),'--len(l)=',len(l))
# result: len(h)= 20 --len(l)= 20
print('Close=',c)
# result:
# Close= [336.1  339.32 345.03 344.32 343.44 346.5  351.88 355.2  358.16 354.54
#  356.85 359.18 359.9  363.13 358.3  350.56 338.61 342.62 342.88 348.16
#  353.21 349.31 352.12 359.56 360.   355.36 355.76 352.47 346.67 351.99]

previousclose = c[-N -1: -1] # 前一个交易日的收盘价
print('len(previousclose)',len(previousclose))
# result: len(previousclose) 20
print('Previous close ',previousclose)
# result: Previous close  [354.54 356.85 359.18 359.9  363.13 358.3  350.56 338.61 342.62 342.88 348.16 353.21 349.31 352.12 359.56 360.   355.36 355.76 352.47 346.67]

truerange = np.maximum(h-l, h-previousclose, previousclose-l)
print('True range ',truerange)
# result:
# True range  [ 4.26  2.77  2.42  5.    3.75  9.98  7.68  6.03  6.78  5.55  6.89  8.04  5.95  7.67  2.54 10.36  5.15  4.16  4.87  7.32]
atr = np.zeros(N)
atr[0] = np.mean(truerange)
for i in range(1,N):
    atr[i] = (N-1)*atr[i-1]+truerange[i]
    atr[i]/=N
print('ATR=',atr)
# result:
# ATR= [5.8585     5.704075   5.53987125 5.51287769 5.4247338  5.65249711
#  5.75387226 5.76767864 5.81829471 5.80487998 5.85913598 5.96817918
#  5.96727022 6.05240671 5.87678637 6.10094705 6.0533997  5.95872972
#  5.90429323 5.97507857]




## 七、画出股票的移动平均线：简单移动平均线和指数移动平均线
# 7.1、简单移动平均线
import numpy as np
from matplotlib.pyplot import plot
from matplotlib.pyplot import show

N =5
weights = np.ones(N)/N
print('Weights =',weights)
# result: Weights = [0.2 0.2 0.2 0.2 0.2]
c = np.loadtxt('D:\\DataguruPyhton\\diLesson04\\AppleStock.csv', delimiter=',',usecols=(6,),unpack=True)
sma = np.convolve(weights,c)[N-1:-N+1]
t = np.arange(N-1, len(c))
plot(t, c[N-1:], lw=1.0)
plot(t, sma, lw=2.0)
show()

# 7.2、指数移动平均线
x = np.arange(5)
print('Exp =',np.exp(x))
# result: Exp = [ 1.          2.71828183  7.3890561  20.08553692 54.59815003]
print('Linspace =',np.linspace(-1,0,5))
# result: Linspace = [-1.   -0.75 -0.5  -0.25  0.  ]
N = 5
weights = np.exp(np.linspace(-1.,0.,N))
weights/=weights.sum()
print('Weights =',weights)
# result: Weights = [0.11405072 0.14644403 0.18803785 0.24144538 0.31002201]
c = np.loadtxt('D:\\DataguruPyhton\\diLesson04\\AppleStock.csv', delimiter=',',usecols=(6,),unpack=True)
ema = np.convolve(weights,c)[N-1:-N+1]
t = np.arange(N-1,len(c))
plot(t,c[N-1:],lw=1.0)
plot(t,ema,lw=2.0)
show()




## 八、画出股票的布林带
N = 5
weights = np.ones(N)/N
print('Weights =',weights)
# result: Weights = [0.2 0.2 0.2 0.2 0.2]
c = np.loadtxt('D:\\DataguruPyhton\\diLesson04\\AppleStock.csv', delimiter=',',usecols=(6,),unpack=True)
sma = np.convolve(weights,c)[N-1:-N+1]
deviation = []
C = len(c)
for i in range(N-1,C):
    if i+N < C:
        dev = c[i:i+N]
    else:
        dev = c[-N:]
    averages = np.zeros(N)
    averages.fill(sma[i-N-1])
    dev = dev -averages
    dev = dev**2
    dev = np.sqrt(np.mean(dev))
    deviation.append(dev)

deviation = 2*np.array(deviation)
print(len(deviation), len(sma))
# result: 26 26
upperBB = sma + deviation
lowerBB = sma - deviation

c_slice = c[N-1:]
between_bands = np.where((c_slice<upperBB)&(c_slice>lowerBB))
print(lowerBB[between_bands])
# result:
# [329.23044409 335.70890572 318.53386282 321.90858271 327.74175968
#  331.5628136  337.94259734 343.84172744 339.99900409 336.58687297
#  333.15550418 328.64879207 323.61483771 327.25667796 334.30323599
#  335.79295948 326.55905786 324.27329493 325.47601386 332.85867025
#  341.63882551 348.75558399 348.48014357 348.01342992 343.56371701
#  341.85163786]
print(c[between_bands])
# result:
# [336.1  339.32 345.03 344.32 343.44 346.5  351.88 355.2  358.16 354.54
#  356.85 359.18 359.9  363.13 358.3  350.56 338.61 342.62 342.88 348.16
#  353.21 349.31 352.12 359.56 360.   355.36]
print(upperBB[between_bands])
# result:
# [354.05355591 351.73509428 373.93413718 374.62741729 374.33024032
#  374.9491864  372.70940266 369.73027256 375.45299591 380.85312703
#  385.78849582 387.77920793 384.58516229 374.03132204 358.88476401
#  353.33904052 363.63294214 370.19870507 372.79598614 372.08532975
#  368.04117449 361.78441601 364.63985643 365.24657008 364.54028299
#  363.04836214]
between_bands = len(np.ravel(between_bands))
print('Ratio between bands ',float(between_bands)/len(c_slice))
# result: Ratio between bands  1.0

t = np.arange(N-1,C)
plot(t, c_slice, lw=1.0)
plot(t, sma, lw=2.0)
plot(t, upperBB, lw=3.0)
plot(t, lowerBB, lw=4.0)
show()




## 九、线性模型
import sys

N = int(sys.argv[1])
c = np.loadtxt('D:\\DataguruPyhton\\diLesson04\\AppleStock.csv', delimiter=',', usecols=(6,), unpack=True)
b = c[-N:]
b = b[::-1]
print("b", b)
# result:
# b [351.99 346.67 352.47 355.76 355.36 360.   359.56 352.12 349.31 353.21
#  348.16 342.88 342.62 338.61 350.56 358.3  363.13 359.9  359.18 356.85
#  354.54 358.16 355.2  351.88 346.5  343.44 344.32 345.03 339.32 336.1 ]

A = np.zeros((N, N), float)
print("Zeros N by N", A)
# result: 报错 MemoryError
for i in range(N):
    A[i,] = c[-N - 1 - i: - 1 - i]

print("A", A)
# result:
(x, residuals, rank, s) = np.linalg.lstsq(A, b)
print(x, residuals, rank, s)
# result:
print(np.dot(b, x))
# result:




## 十、趋势线
from __future__ import division
from matplotlib.pyplot import plot
from matplotlib.pyplot import show

def fit_line(t, y):
    A = np.vstack([t, np.ones_like(t)]).T
    return np.linalg.lstsq(A, y)[0]

h, l, c = np.loadtxt('D:\\DataguruPyhton\\diLesson04\\AppleStock.csv', delimiter=',', usecols=(4, 5, 6), unpack=True)
pivots = (h + l + c) / 3
print("Pivots", pivots)
# result:
# Pivots [338.01       337.88666667 343.88666667 344.37333333 342.07666667
#  345.57       350.92333333 354.29       357.34333333 354.18
#  356.06333333 358.45666667 359.14       362.84333333 358.36333333
#  353.19333333 340.57666667 341.95666667 342.13333333 347.13

t = np.arange(len(c))
sa, sb = fit_line(t, pivots - (h - l))
ra, rb = fit_line(t, pivots + (h - l))
support = sa * t + sb
resistance = ra * t + rb
condition = (c > support) & (c < resistance)
print("Condition", condition)
# result:
# Condition [False False  True  True  True  True  True False False  True False False
#  False False False  True False False False  True  True  True  True False
#  False  True  True  True False  True]

between_bands = np.where(condition)
print(support[between_bands])
# result:
# [341.92421382 342.19081893 342.45742405 342.72402917 342.99063429
#  343.79044964 345.39008034 346.4565008  346.72310592 346.98971104
#  347.25631615 348.0561315  348.32273662 348.58934174 349.12255197]
print(c[between_bands])
# result: [345.03 344.32 343.44 346.5  351.88 354.54 350.56 348.16 353.21 349.31 352.12 355.36 355.76 352.47 351.99]
print(resistance[between_bands])
# result:
# [352.61688271 352.90732765 353.19777259 353.48821753 353.77866246
#  354.64999728 356.39266691 357.55444667 357.84489161 358.13533655
#  358.42578149 359.2971163  359.58756124 359.87800618 360.45889606]
between_bands = len(np.ravel(between_bands))
print("Number points between bands", between_bands)
# result: Number points between bands 15
print("Ratio between bands", float(between_bands) / len(c))
# result: Ratio between bands 0.5
print("Tomorrows support", sa * (t[-1] + 1) + sb)
# result: Tomorrows support 349.38915708812254
print("Tomorrows resistance", ra * (t[-1] + 1) + rb)
# result: Tomorrows resistance 360.7493409961686
a1 = c[c > support]
a2 = c[c < resistance]
print("Number of points between bands 2nd approach", len(np.intersect1d(a1, a2)))
# result: Number of points between bands 2nd approach 15
plot(t, c)
plot(t, support)
plot(t, resistance)
show()