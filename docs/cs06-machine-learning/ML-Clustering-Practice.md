---
title: ML-Clustering-Practice
date: 2024-04-10 22:22:21
author: sheldon
categories: 
- ML
tags:
- ML
---

The notebook performs three tasks on the dataset `Wine`.

- Task 1: Implement and try different clustering techniques
- Task 2: Evaluate the clustering results against ground truth
- Task 3: Dimensionality reduction and visualization

## Preliminaries

```python
# Load necessary libraries.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

np.random.seed(20240410)

def plot_clusters(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', cmap=plt.cm.coolwarm)
    plt.xlabel('x1')
    plt.ylabel('x2')

# load dataset
wine = load_wine()
print(wine.feature_names)
print(wine.DESCR)
X, y = load_wine(return_X_y=True)

# feature normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

```txt
...

Wine recognition dataset
------------------------

**Data Set Characteristics:**

    :Number of Instances: 178 (50 in each of three classes)
    :Number of Attributes: 13 numeric, predictive attributes and the class
    :Attribute Information:
    - Alcohol
    - Malic acid
    - Ash
    - Alcalinity of ash  
    - Magnesium
    - Total phenols
    - Flavanoids
    - Nonflavanoid phenols
    - Proanthocyanins
    - Color intensity
    - Hue
    - OD280/OD315 of diluted wines
    - Proline

    - class:
            - class_0
            - class_1
            - class_2

    :Summary Statistics:
    
    ============================= ==== ===== ======= =====
                                   Min   Max   Mean     SD
    ============================= ==== ===== ======= =====
    Alcohol:                      11.0  14.8    13.0   0.8
    Malic Acid:                   0.74  5.80    2.34  1.12
    Ash:                          1.36  3.23    2.36  0.27
    Alcalinity of Ash:            10.6  30.0    19.5   3.3
    Magnesium:                    70.0 162.0    99.7  14.3
    Total Phenols:                0.98  3.88    2.29  0.63
    Flavanoids:                   0.34  5.08    2.03  1.00
    Nonflavanoid Phenols:         0.13  0.66    0.36  0.12
    Proanthocyanins:              0.41  3.58    1.59  0.57
    Colour Intensity:              1.3  13.0     5.1   2.3
    Hue:                          0.48  1.71    0.96  0.23
    OD280/OD315 of diluted wines: 1.27  4.00    2.61  0.71
    Proline:                       278  1680     746   315
    ============================= ==== ===== ======= =====
...
```

## Task 1: Implement and try different clustering techniques

- Implement the following clustering methods and tune their hyperparameters accordingly. Set the random_state for ALL methods when applicable as random_state=20240410 to ensure the reproducibility.
  - kmeans, using silhouette analysis to find the optimal number of clusters, ranging from 2 to 5 (both included), same for other methods
  - Hierarchical (Agglomerative) clustering, using silhouette analysis to find the optimal number of clusters and linkage functions
  - Gaussian Mixture Model, using silhouette analysis to find the optimal number of clusters and covariance types

### Kmeans

- kmeans, using silhouette analysis to find the optimal number of clusters, ranging from 2 to 5 (both included), same for other methods

```python
# kmeans
n_clusters = range(2,6)
highest_scores = 0

for n in n_clusters:
    kmeans = KMeans(n_clusters=n, random_state=20240410)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, cluster_labels)
    if silhouette_avg > highest_scores:
        highest_scores = silhouette_avg
        best_n = n

kmeans = KMeans(n_clusters=best_n, random_state=20240410)
kmeans.fit(X)
cluster_labels = kmeans.labels_
silhouette_avg = silhouette_score(X, cluster_labels)

print(f"The best parameters are: n_clusters={best_n}")
print("The average silhouette_score is :", silhouette_avg)
```

```txt
The best parameters are: n_clusters=3
The average silhouette_score is : 0.2848589191898987
```

### Agglomerative clustering

- Hierarchical (Agglomerative) clustering, using silhouette analysis to find the optimal number of clusters and linkage functions

```python
# agglomerative clustering
linkages = ['ward', 'complete', 'average', 'single']
n_clusters = range(2,6)
highest_scores = 0

for n in n_clusters:
    for link in linkages:
        agglo = AgglomerativeClustering(n_clusters=n, linkage = link)
        agglo.fit(X)
        cluster_labels = agglo.labels_
        silhouette_avg = silhouette_score(X, cluster_labels)
        if silhouette_avg > highest_scores:
            highest_scores = silhouette_avg
            best_n = n
            best_linkage = link


agglo = AgglomerativeClustering(n_clusters=best_n, linkage = best_linkage)
agglo.fit(X)
cluster_labels = agglo.labels_
silhouette_avg = silhouette_score(X, cluster_labels)

print(f"The best parameters are: n_clusters={best_n}, linkage={best_linkage}")
print("The average silhouette_score is :", silhouette_avg)
```

```txt
The best parameters are: n_clusters=3, linkage=ward
The average silhouette_score is : 0.2774439826952266
```

### Gaussian mixture model

```python
# gaussian mixture model

covariance_type = ['full','tied','diag','spherical']
n_clusters = range(2,6)
highest_scores = 0
for n in n_clusters:
    for covar in covariance_type:
        gmm = GaussianMixture(n_components=n, covariance_type=covar, random_state=20240410)
        cluster_labels = gmm.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        # print(f"Covar: {covar} cluster: {n} Score: {silhouette_avg}")
        if silhouette_avg > highest_scores:
            highest_scores = silhouette_avg
            best_n = n
            best_covar = covar

gmm = GaussianMixture(n_components=best_n, covariance_type=best_covar, random_state=20240410)
cluster_labels = gmm.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)

print(f"The best parameters are: n_clusters={best_n}, covariance_type={best_covar}")
print("The average silhouette_score is :", silhouette_avg)
```

```txt
The best parameters are: n_clusters=3, covariance_type=full
The average silhouette_score is : 0.28356363134288903
```

### Summary

- What are the best-performing hyperparameters for each method?
  - kmeans:
    - n_clusters=3, aver_silhouette_score: 0.2848589191898987
  - Hierarchical (Agglomerative):
    - n_clusters=3, linkage=ward, aver_silhouette_score: 0.2774439826952266
  - Gaussian Mixture Model:
    - n_clusters=3, covariance_type=full, aver_silhouette_score is : 0.28356363134288903

## Task 2: Evaluate the clustering results against ground truth

- Select the best-performing hyperparameters for each method, set the random_state for ALL methods when applicable as random_state=20240410 to ensure the reproducibility.
- Evaluate the clustering results from the three methods (kmeans, agglomerative clustering, and Gaussian mixture model) against ground truth y using
  - 1) adjusted rand index and
  - 2) normalized mutual information.

```python
# evaluation using ARI and NMI

plt.figure(figsize=(12, 4))

kmeans = KMeans(n_clusters=3, random_state=20240410)
kmeans.fit(X)
cluster_labels = kmeans.labels_
ari = adjusted_rand_score(cluster_labels, y)
nmi = normalized_mutual_info_score(cluster_labels, y)
print("kmeans ARI is:", ari,  "NMI is :", nmi)
plt.subplot(1,3,1, title="kmeans")
plot_clusters(X, cluster_labels)

agglo = AgglomerativeClustering(n_clusters=3, linkage = 'ward')
agglo.fit(X)
cluster_labels = agglo.labels_
ari = adjusted_rand_score(cluster_labels, y)
nmi = normalized_mutual_info_score(cluster_labels, y)
print("Agglo ARI is:", ari,  "NMI is :", nmi)
plt.subplot(1,3,2, title="agglo")
plot_clusters(X, cluster_labels)

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=20240410)
cluster_labels = gmm.fit_predict(X)
ari = adjusted_rand_score(cluster_labels, y)
nmi = normalized_mutual_info_score(cluster_labels, y)
print("GMM ARI is:", ari,  "NMI is :", nmi)
plt.subplot(1,3,3, title="gmm")
plot_clusters(X, cluster_labels)
```

```txt
kmeans ARI is: 0.8974949815093207 NMI is : 0.8758935341223069
Agglo ARI is: 0.7899332213582837 NMI is : 0.7864652657004837
GMM ARI is: 0.8974949815093207 NMI is : 0.8758935341223069
```

<!-- ![Wine-Task2-Clustering](./Wine-Task2-Clustering.png) -->
{% asset_img Wine-Task2-Clustering.png %}

- Which one is the best performing method?
  - both kmeans and GMM are best performing methods with (ARI=0.8974949815093207,NMI=0.8758935341223069)

## Task 3:  Dimensionality reduction and visualization

### PCA

- The dataset contains 13 features, making it hard to visualize. Perform PCA to reduce it to a 2D space and visualize it using plot_clusters.

```python
# PCA visualization

pca = PCA(n_components=2, random_state=20240410)
X_reduced = pca.fit_transform(X)

print("Singular values are:", pca.singular_values_)
print("Absolute explained variance are:", pca.explained_variance_)
print("Ratio of explained variance are:", pca.explained_variance_ratio_)
print("Cumulative ratio of explained variance are:", np.cumsum(pca.explained_variance_ratio_))

plot_clusters(X_reduced, y)
```

```txt
Singular values are: [28.94203422 21.08225141]
Absolute explained variance are: [4.73243698 2.51108093]
Ratio of explained variance are: [0.36198848 0.1920749 ]
Cumulative ratio of explained variance are: [0.36198848 0.55406338]
```

- What is the ratio of explained variance captured by the 2D space?
  - 0.55406338

```python
pca = PCA(random_state=20240410)
X_pca = pca.fit_transform(X)
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_components = np.where(cumulative_variance_ratio >= 0.8)[0][0] + 1
print(f"The number of PCs to retain >80% data info is: {n_components}")
```

```txt
The number of PCs to retain >80% data info is: 5
```

### T-SNE

- If the ratio of captured variance is too low (<0.8 for example), it means a large loss of information by using linear transformation. So you may want to try t-SNE for visualization. Can you use t-SNE to project X to 2D and visualize clusters with ground truth y? Set random_state=20240410 to ensure the reproducibility

```python
# TSNE visualization

tsne = TSNE(n_components=2, random_state=20240410)
X_reduced = tsne.fit_transform(X)  

plt.figure(figsize=(12, 6))

plt.subplot(1,2,1, title="original")
plot_clusters(X, y)

plt.subplot(1,2,2, title="2D-X_reduced")
plot_clusters(X_reduced, y)
```

<!-- ![TSNE](./Wine-Task3-TSNE.png) -->
{% asset_img Wine-Task3-TSNE.png %}

### Visualization

- Visualize the clustering results (predicted labels) from the three methods (kmeans, agglomerative clustering, and Gaussian mixture model). Compare the clustering results visually against the ground truth, and think about whether the visualization corresponds to your answer to the last question in Task 2.

```python
plt.figure(figsize=(12, 12))

plt.subplot(2,2,1, title="original")
plot_clusters(X, y)

kmeans = KMeans(n_clusters=3, init='random', random_state=20240410)
kmeans.fit(X)
cluster_labels = kmeans.labels_
ari = adjusted_rand_score(cluster_labels, y)
nmi = normalized_mutual_info_score(cluster_labels, y)
print("kmeans ARI is:", ari,  "NMI is :", nmi)
plt.subplot(2,2,2, title="kmeans")
plot_clusters(X, cluster_labels)

agglo = AgglomerativeClustering(n_clusters=3, linkage = 'ward')
agglo.fit(X)
cluster_labels = agglo.labels_
ari = adjusted_rand_score(cluster_labels, y)
nmi = normalized_mutual_info_score(cluster_labels, y)
print("Agglo ARI is:", ari,  "NMI is :", nmi)
plt.subplot(2,2,3, title="agglo")
plot_clusters(X, cluster_labels)

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=20240410)
cluster_labels = gmm.fit_predict(X)
ari = adjusted_rand_score(cluster_labels, y)
nmi = normalized_mutual_info_score(cluster_labels, y)
print("GMM ARI is:", ari,  "NMI is :", nmi)
plt.subplot(2,2,4, title="gmm")
plot_clusters(X, cluster_labels)
```

<!-- ![Wine-Task3-Clustering-Compare](./Wine-Task3-Clustering-Compare.png) -->
{% asset_img Wine-Task3-Clustering-Compare.png %}
