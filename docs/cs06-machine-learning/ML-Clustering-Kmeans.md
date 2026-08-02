---
title: ML-Clustering-Kmeans
date: 2024-04-24 17:36:07
author: sheldon
categories: 
- ML
tags:
- ML
mathjax: true
---

## Introduction

In this notebook, we shall be looking at how the kmeans algorithm works. KMeans is an **unsupervised learning** algorithm that is used to cluster data in groups - without knowing which group given data elements belong to as are going to see.

```python
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import matplotlib as mpl
```

Let's shall generate a random dataset of 1000 points clustered into 3 groups.

```python
# 设置数据集中样本点的数量
N = 1000

# 生成具有3个中心点的聚类数据集
X_, y_ = sklearn.datasets.make_blobs(n_samples=N+5, centers=3) 

# 从生成的数据集中提取前N个样本作为训练数据
X, y = X_[:N], y_[:N]

# 从生成的数据集中提取后面5个样本作为测试数据
X_test, y_test = X_[N:], y_[N:]

# 绘制数据集的散点图
plt.figure(figsize=(8, 6))
for cls in np.unique(y):
    plt.scatter(X[y==cls][:, 0], X[y==cls][:, 1], s=2)
plt.title("Plot of features of dataset X", fontsize=14)
plt.xlabel("x1", fontsize=12)
plt.ylabel("x2", fontsize=12)
plt.show()
```

<!-- ![sampla dataset](./Kmeans/sample_dataset_X.png) -->

{% asset_img sample_dataset_X.png %}

## 1.1 First things first

1. We need to determine and set a value k, the number of clusters we **think** the data has. KMeans is unsupervised. So it is not the case that you will always know how many clusters (k) exist in the data. You will have to experiment with different values using certain techniques to find the best value of k. For our case, we know that there are 3 clusters, therefore we shall set k to 3. (You can try a different value after the points get clear).

2. We also need to **find k=3 random points** that will represent the centers of our clusters if the clustering is successfull. These k random points are called centroids.Let's work on these two steps next.

```python
k = 3
N, feature_size = X.shape

# 获取特征的最大值和最小值范围
min_feature_range = np.min(X, axis=0)
max_feature_range = np.max(X, axis=0)

# 在上述范围内生成k个随机点
centroids = np.zeros((len(max_feature_range), k))

for i, (l, h) in enumerate(zip(min_feature_range, max_feature_range)):
    # 使用random.uniform()函数从均匀分布中抽取样本
    centroids[i, :] = np.random.uniform(low=l, high=h, size=k)

# 转置centroids矩阵，使得每行表示一个聚类中心
centroids = centroids.T
print(centroids)

# 绘制数据集的散点图和聚类中心
plt.figure(figsize=(8, 6))
for cls in np.unique(y):
    plt.scatter(X[y==cls][:, 0], X[y==cls][:, 1], s=2)

# 绘制聚类中心
for i, (x_, y_) in enumerate(centroids):
    plt.scatter(x_, y_, marker='x', c='k')
    plt.annotate(xy=(x_+.1, y_-.1), text='c'+str(i), color='r')
plt.title("Plot of features of dataset X", fontsize=14)
plt.xlabel("x1", fontsize=12)
plt.ylabel("x2", fontsize=12)
plt.show()
```

<!-- ![1.1](./Kmeans/1.1.png) -->

{% asset_img 1.1.png %}

The points are scattered and may not be close to the cluster centers. There are other methods like the **kmeans++**.

## 1.2 Calculate distances to centroids

In this notebook, we shall be calculating the euclidean distance. We have 2 columns in X *(x1 and x2)* and we have to calculate the euclidean distance between each centroid and every data point in X. Centroids are of the form *(xx, yy)* i.e they have two points just like our dataset X has 2 columns.

计算每个质心和 X 中每个数据点之间的欧几里得距离。

We want to calculate something of the form $sqrt((x1-xx)^2 + (x2-yy)^2)$ for each data point/row in X.
To accelerate operations, we shall be using a vectorized approach to calculate that.

使用矢量化方法来计算。

1. We have k=3 centroids, so we shall first duplicate **X** 3 times or k times. The shape of X is (1000, 2). The result of the duplication, **Xc** will be (1000, 6).
2. Then we shall flatten the centroids so that its a single vector, **centroidsc** of 6 elements to match our 6 columns in **Xc**.
3. We shall subtract **Xc** and **centroidsc** to give us a result **D** of shape (1000, 6). This step is equivalent to performing **(x1-xx)** and **(x2-yy)** for all centroids at once.

- The first column of **D** corresponds to **(x1-xx)** where xx is the x of the first centroid.
- The second column of **D** corresponds to **(x2-yy)** where yy is the y of the first centroid.
- The third column of **D** corresponds to **(x1-xx)** where xx is the x of the *second* centroid.
- The forth column of **D** corresponds to **(x2-yy)** where yy is the y of the *second* centroid. And so on.

4.The next step is to square these results, add them and apply sqrt. This whole operation results in what is called the *L2 norm* and is all performed by the **np.linalg.norm** function.

Note that we have to calculate the norm over a given set of columns e.g the first and second columns' norm corresponds to the first centroid, the third and forth to the 2nd centroid and the last 2 to the 3rd centroid. So in the end we have a (1000, 3) array containing euclidean distances of each of the 1000 data rows/points in X in correspondence to each of the 3 centroids.

```python
Xc = np.concatenate([X for c in centroids], axis=1) # duplicate X k times
centroidsc = centroids.ravel() # ravel to allow broadcast; Return a contiguous flattened array.
D = (Xc - centroidsc) # raw diff

Norms = np.zeros((N, k)) # distances to each centroid
for i in range(0, k): 
    m = i*feature_size
    Norms[:, i] = np.linalg.norm(D[:, m:m+feature_size], axis=1) # Calculating the norms (Euclidean distance)
```

## 1.3 Attach instances or rows to the closest centroid

We shall now assign each data row in X the index of the centroid with which it has the shortest distance. We do that using **np.argmin** which returns the index of the minimum distance in our *Norms* array.

使用 **np.argmin** 来为 X 中的每个数据行分配与其距离最短的质心的索引，它返回 *Norms* 数组中最小距离的索引。

```python
# sample of indices for smallest indices for sample_norms
np.argmin(sample_norms, axis=1)
# we do this for all Norms
ypred = np.argmin(Norms, axis=1) #Returns the indices of the minimum values along an axis.
plt.figure(figsize=(8, 6))
for i, (x_, y_) in enumerate(centroids):  # with underscore
    p = plt.scatter(X[y==i][:, 0], X[y==i][:, 1], s=2)
    clr = mpl.colors.to_rgba(p.get_facecolor()) # get color used by matplotlib
    plt.scatter(x_, y_, marker='x', c='k')
    anot = 'c'+str(i) + " at " + str(np.round(Norms[0, i], 3))
    plt.plot([X[0][0], x_], [X[0][1], y_])
    plt.annotate(xy=(x_+.2, y_-.1), text=anot, color='k', size=10)
    
plt.scatter(X[0][0], X[0][1], marker='o', c='r', s=35)

plt.title("Plot showing distance sample (point belongs to c"+str(ypred[0])+")", fontsize=14)
plt.xlabel("x1", fontsize=12)
plt.ylabel("x2", fontsize=12)

plt.show()
```

<!-- ![1.3](./Kmeans/1.3.png) -->

{% asset_img 1.3.png %}

## 1.4 Update centroids

The last step is to update the centroids by setting the new centroids at the mean positions of the points they were closest to i.e the data points they influence.

Below is a plot showing the un-updated centroids and their influence on the data points. We shall have to move the centroids so that they are at the center of the points they influence.

```python
plt.figure(figsize=(8, 6))
for i, (x_, y_) in enumerate(centroids):
    p = plt.scatter(X[ypred==i][:, 0], X[ypred==i][:, 1], s=2)
    clr = mpl.colors.to_rgba(p.get_facecolor()) # get color used by matplotlib
    plt.scatter(x_, y_, marker='x', c='k')
    plt.annotate(xy=(x_+.1, y_-.1), text='c'+str(i), color=clr, size=14)
plt.title("Plot showing influence of recent centroids on data points", fontsize=14)
plt.xlabel("x1", fontsize=12)
plt.ylabel("x2", fontsize=12)
plt.show()
```

<!-- ![1.4.1](./Kmeans/1.4.1.png) -->

{% asset_img 1.4.1.png %}

- To do that, we shall calculate the mean of the data points each centroid influences and put the centroid at that mean location.
- If a centroid has no points it influences (yes, this can happen), we leave the centroid where it is.

```python
centroids_ = []
for i in range(k):
    if len(X[ypred == i]) == 0:
        centroids_.append(centroids[i]) # use old
    else:
        centroids_.append(np.mean(X[ypred == i], axis=0))
centroids_ = np.array(centroids_)
centroids_

#After the update, we have the following plot
plt.figure(figsize=(8, 6))
for i, (x_, y_) in enumerate(centroids_):  # with underscore
    p = plt.scatter(X[ypred==i][:, 0], X[ypred==i][:, 1], s=2)
    clr = mpl.colors.to_rgba(p.get_facecolor()) # get color used by matplotlib
    plt.scatter(x_, y_, marker='x', c='k')
    plt.annotate(xy=(x_+.1, y_-.1), text='c'+str(i), color=clr, size=14)
plt.title("Plot showing mean-centered centroids ", fontsize=14)
plt.xlabel("x1", fontsize=12)
plt.ylabel("x2", fontsize=12)

plt.show()
```

[//]: ![1.4.2](./Kmeans/1.4.2.png)
{% asset_img 1.4.2.png %}

## Note

- The mean-centered points may look good, however the data they influence may still be bad
  
## 1.5 Repeat the above steps

- Now we repeat the steps above until the centroids don't update or move (significantly) anymore. That will be the case when **(centroids_ - centroids)^2** is a low value below a certain threshold. A good threshold has to be as low as possible i.e close to or equal to zero.
- You can run the following code multiple times and observe **shift** value (which is **changes of centroids** ) carefully
- Here, we are using **^2** to make larger shifts/updates significant. It doesn't have to be that way. A norm can also work.

```python
shift = np.sum((centroids_ - centroids)**2)
print(shift)
# Below we repeat all the previous steps in one run
centroids = centroids_
Xc = np.concatenate([X for c in centroids], axis=1) # duplicate k times
centroidsc = centroids.ravel() # ravel to allow broadcast
D = (Xc - centroidsc) # raw diff

Norms = np.zeros((N, k)) # distances to each cluster
for i in range(0, k):
    m = i*feature_size
    Norms[:, i] = np.linalg.norm(D[:, m:m+feature_size], axis=1)

# Choose the nearest cluster for every point, and save the result in ypred which is used in next code segment
# We can use np.argmin function
ypred = np.argmin(Norms, axis=1) 

# new clusters are mean X along 
centroids_ = []
for i in range(k):
    if len(X[ypred == i]) == 0:
        centroids_.append(centroids[i]) # use old
    else:
        centroids_.append(np.mean(X[ypred == i], axis=0))
centroids_ = np.array(centroids_)
centroids_

plt.figure(figsize=(8, 6))
for i, (x_, y_) in enumerate(centroids_):  # with underscore
    p = plt.scatter(X[ypred==i][:, 0], X[ypred==i][:, 1], s=2)
    clr = mpl.colors.to_rgba(p.get_facecolor()) # get color used by matplotlib
    plt.scatter(x_, y_, marker='x', c='k')
    plt.annotate(xy=(x_+.1, y_-.1), text='c'+str(i), color=clr, size=14)
plt.title("Plot showing mean-centered centroids ", fontsize=14)
plt.xlabel("x1", fontsize=12)
plt.ylabel("x2", fontsize=12)

plt.show()
```

{% asset_img 1.5.png %}
