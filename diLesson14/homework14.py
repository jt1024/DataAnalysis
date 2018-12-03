#!/usr/bin/python
# encoding:utf-8

"""
@author:jiat
@contact:cctvjiatao@163.com
@file:homework14.py
@time:2018/4/29 11:36
"""

'''
数据集 ex14.csv 是关于中国各个省份的三项指标数值。
请根据这些指标数值，将各个省份分为3类，并尝试归纳出各个类别的特点'''
import os
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

ex14 = pd.read_csv('D:\\DataguruPyhton\\DataAnalysis\\diLesson14\\ex14.csv', header=0, index_col=0, parse_dates=True, encoding='gb18030')
ex14.head()
ex14.index
# 查看图形
fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
ax3D.scatter(ex14['DXBZ'], ex14['CZBZ'], ex14['WMBZ'], marker='o')
ax3D.set_xlabel('DXBZ')
ax3D.set_ylabel('CZBZ')
ax3D.set_zlabel('WMBZ')
plt.show()

# cluster
for linkage in ('ward', 'average', 'complete'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=3)
    clustering.fit(ex14)
    cluster_pred = clustering.fit_predict(ex14)
    print
    'clusters :', cluster_pred
    ex14[u'类型'] = cluster_pred
    print
    ex14
    fig = plt.figure()
    ax3D = fig.add_subplot(111, projection='3d')
    ax3D.scatter(ex14['DXBZ'], ex14['CZBZ'], ex14['WMBZ'], c=cluster_pred, marker='o')
    ax3D.set_xlabel('DXBZ')
    ax3D.set_ylabel('CZBZ')
    ax3D.set_zlabel('WMBZ')
    plt.show()

# kmeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(ex14)
kmeans_pred = kmeans.fit_predict(ex14)
print
'kmeans cluster:', kmeans_pred
ex14[u'类型'] = kmeans_pred
print
ex14
fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
ax3D.scatter(ex14['DXBZ'], ex14['CZBZ'], ex14['WMBZ'], c=kmeans_pred, marker='o')
ax3D.set_xlabel('DXBZ')
ax3D.set_ylabel('CZBZ')
ax3D.set_zlabel('WMBZ')
plt.show()

# DBSCAN
db = DBSCAN(eps=2.2, min_samples=3)
db.fit(ex14)
db_pred = db.fit_predict(ex14)
print
'db cluster:', db_pred
print
'clster 1:', len(db_pred[db_pred == 1]) / len(db_pred)
print
'clster 0:', len(db_pred[db_pred == 0]) / len(db_pred)
print
'clster -1:', len(db_pred[db_pred == -1]) / len(db_pred)
db_pred[db_pred == 1] = 3
db_pred[db_pred == 0] = 2
db_pred[db_pred == -1] = 1
fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
ax3D.scatter(ex14['DXBZ'], ex14['CZBZ'], ex14['WMBZ'], c=db_pred, marker='o')
ax3D.set_xlabel('DXBZ')
ax3D.set_ylabel('CZBZ')
ax3D.set_zlabel('WMBZ')
plt.show()


# 结果:
# 1.cluster聚类算法的结果
#      DXBZ   CZBZ   WMBZ  类型
# 北京   9.30  30.55   8.70   0
# 天津   4.67  29.38   8.92   0
# 河北   0.96  24.69  15.21   0
# 山西   1.38  29.24  11.30   0
# 内蒙古  1.48  25.47  15.39   0
# 辽宁   2.60  32.32   8.81   0
# 吉林   2.15  26.31  10.49   0
# 黑龙江  2.14  28.46  10.87   0
# 上海   6.53  31.59  11.04   0
# 江苏   1.47  26.43  17.23   0
# 浙江   1.17  23.74  17.46   0
# 安徽   0.88  19.97  24.43   2
# 福建   1.23  16.87  15.63   0
# 江西   0.99  18.84  16.22   0
# 山东   0.98  25.18  16.87   0
# 河南   0.85  26.55  16.15   0
# 湖北   1.57  23.16  15.79   0
# 湖南   1.14  22.57  12.10   0
# 广东   1.34  23.04  10.45   0
# 广西   0.79  19.14  10.61   0
# 海南   1.24  22.53  13.97   0
# 四川   0.96  21.65  16.24   0
# 贵州   0.78  14.65  24.27   2
# 云南   0.81  13.85  25.44   2
# 西藏   0.57   3.85  44.43   1
# 陕西   1.67  24.36  17.62   0
# 甘肃   1.10  16.85  27.93   2
# 青海   1.49  17.76  27.70   2
# 宁夏   1.61  20.27  22.06   2
# 新疆   1.85  20.66  12.75   0
#
# 分成三类：第一类包含一个省份：西藏,第二类包含6个省份，分别为：甘肃、青海、宁夏、云南、贵州、安徽, 第三类为是剩下所有省份
# 分在第一类是DXBZ偏低， CZBZ偏低，WMBZ 偏高，分在第二类是WMBZ偏高
#
# 2.kmeans聚类算法的结果
#     DXBZ   CZBZ   WMBZ  类型
# 北京   9.30  30.55   8.70   1
# 天津   4.67  29.38   8.92   1
# 河北   0.96  24.69  15.21   1
# 山西   1.38  29.24  11.30   1
# 内蒙古  1.48  25.47  15.39   1
# 辽宁   2.60  32.32   8.81   1
# 吉林   2.15  26.31  10.49   1
# 黑龙江  2.14  28.46  10.87   1
# 上海   6.53  31.59  11.04   1
# 江苏   1.47  26.43  17.23   1
# 浙江   1.17  23.74  17.46   1
# 安徽   0.88  19.97  24.43   0
# 福建   1.23  16.87  15.63   1
# 江西   0.99  18.84  16.22   1
# 山东   0.98  25.18  16.87   1
# 河南   0.85  26.55  16.15   1
# 湖北   1.57  23.16  15.79   1
# 湖南   1.14  22.57  12.10   1
# 广东   1.34  23.04  10.45   1
# 广西   0.79  19.14  10.61   1
# 海南   1.24  22.53  13.97   1
# 四川   0.96  21.65  16.24   1
# 贵州   0.78  14.65  24.27   0
# 云南   0.81  13.85  25.44   0
# 西藏   0.57   3.85  44.43   2
# 陕西   1.67  24.36  17.62   1
# 甘肃   1.10  16.85  27.93   0
# 青海   1.49  17.76  27.70   0
# 宁夏   1.61  20.27  22.06   0
# 新疆   1.85  20.66  12.75   1
#
# 分成三类：第一类包含一个省份：西藏,第二类包含6个省份，分别为：甘肃、青海、宁夏、云南、贵州、安徽, 第三类为是剩下所有省份
# 分在第一类是DXBZ偏低， CZBZ偏低，WMBZ 偏高，分在第二类是WMBZ偏高