---
title: 'Expected Number of Points Whose Neighbor of Neighbor is itself'
date: 2023-03-08
permalink: /posts/2023/03/neighbor_of_neighbor/
tags:
  - probability
---

**Problem**: Consider drawing $n$ ($n \geq 3$) independent and identically distributed (i.i.d.) points uniformly from the interval $[0,1]$. Let the neighbor of a point be defined as the closest point to it, excluding itself. If the neighbor of the neighbor is the point itself, then we call this point a good point. In this context, we want to determine the expected number of good points, denoted by $f(n)$. Can we find a general expression for $f(n)$?


**Solution**: To start, we introduce some notation. Let $X_1,X_2,\dots,X_n$ be $n \geq 3$ random variables uniformly distributed on $[0,1]$. Let $X_{(1)},X_{(2)},\dots,X_{(n)}$ denote their order statistics, which is the sorted array of $X_1,X_2,\dots,X_n$. Note that the order statistics are random variables themselves. We use $N(i)$ to denote the index of the neighbor of $X_i$. Therefore, if $X_i$ is a good point, then we have $N(N(i))=i$.

Initially, the problem may seem challenging because the concept of the neighbor of neighbor involves multiple random variables. However, after further analysis, we make the following observations:

- The neighbor of $X_{(i)}$ can only be $X_{(i-1)}$ (if $i > 2$) or $X_{(i+1)}$ (if $i < n$).
- If the neighbor of $X_{(i)}$ is $X_{(j)}$ (where $j=i\pm 1$) and if $X_{(i)}$ is a good point, then $X_{(j)}$ is a good point as well. Based on this observation, we refer to the edge ${X_{(i)},X_{(j)}}$ as a good edge if either, and thus both, of them are good points. Therefore, the total number of good points is given by $$\mathbb{E}\left[\sum_{i=1}^{n-1} 1_{\{X_{(i)},X_{(j)}\} \text{is a good edge}}\right].$$ Using the linearity of expectation, we obtain $$\sum_{i=1}^{N-1} \mathbb{E}[1_{\{X_{(i)},X_{(j)}\} \text{is a good edge}}]$$.

(not finished yet. to be continued.)