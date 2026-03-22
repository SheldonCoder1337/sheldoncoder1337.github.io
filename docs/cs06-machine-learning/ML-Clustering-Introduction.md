---
title: ML-Clustering-Introduction
date: 2024-07-06 11:29:27
author: sheldon
categories: 
- ML
tags:
- ML
---

## Content

For Clustering, the following exercises are included

- ML-clustering-Kmeans
  - Exercise 1: kmeans clustering
  - Exercise 2: kmeans clustering result analysis using silhouette analysis¶
  - Exercise 3: How kmeans performs across datasets with different structures
  - Exercise 4: Silhouette analysis on these three datasets of different structures
- ML-clustering-Hierarchical
  - Exercise 5: Hierarchical (Agglomerative) clustering dendrogram
  - Exercise 6: Hierarchical (Agglomerative) clustering results analysis
  - Exercise 7: Hierarchical (Agglomerative) clustering with different linkage methods
  - Exercise 8: Hierarchical (Agglomerative) clustering with silhouette analysis
- ML-clustering-DBSCAN
  - Exercise 9: DBSCAN
- ML-clustering-Mixture
  - Exercise 10: Gaussian Mixture Model
  - Exercise 11: Choose the covariance type of GMM using Bayesian Information Criterion (BIC)
  - Exercise 12: Choose the number of clusters using Bayesian Information Criterion (BIC)
  - Exercise 13: GMM clustering result analysis using silhouette analysis
  - Exercise 14: When GMM with BIC is powerful?
  - Exercise 15: GMM on three datasets of different structures

## Summary

- Clustering analysis：Unsupervised learning
- Kmeans: A partitioning-based method
  - Initialization matters -> K-means++
  - Choose number of clusters -> silhouette analysis
- Hierarchical Clustering
  - Dendrogram
  - Linkage methods matter
- DBSCAN
  - Outlier/noise detection
  - Choose ε can be a tricky task
- Mixture Models
  - Limitations of k-means: Hard assignment to clusters -> Soft (probabilistic) assignment
  - Mixture Models: Mixture of Gaussian
    - Soft assignment
    - Model parameters {πk , μk , Σk}
  - Expectation Maximization
    - E-step: estimate cluster responsibilities given current parameter estimates
    - M-step: maximize likelihood over parameters given current responsibilities
    - Connection to k-means (Infinitely Small Variance Gaussian Mixture EM = k-means)
