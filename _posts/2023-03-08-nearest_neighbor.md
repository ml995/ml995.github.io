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
- If the neighbor of $X_{(i)}$ is $X_{(j)}$ (where $j=i\pm 1$) and if $X_{(i)}$ is a good point, then $X_{(j)}$ is a good point as well. Based on this observation, we refer to the edge ${X_{(i)},X_{(j)}}$ as a good edge if either, and thus both, of them are good points. Therefore, the expected value of the total number of good points can be expressed as $$f(n) = 2\mathbb{E}\left[\sum_{i=1}^{n-1} 1_{(X_{(i)},X_{(i+1)}) \text{ is a good edge}}\right]$$ where the factor of 2 is due to each good edge contributing 2 good points. Using the linearity of expectation, we obtain $$f(n)= 2\sum_{i=1}^{n-1} \mathbb{E}[1_{(X_{(i)},X_{(i+1)}) \text{ is a good edge}}] = 2\sum_{i=1}^{n-1} \Pr[(X_{(i)},X_{(i+1)}) \text{ is a good edge}] $$. 

We first define $Y_i = X_{(i+1)} - X_{(i)}$ for $i=1,2,\dots,n-1$. Note that $Y_i$ has the same distribution. There are two cases to consider:

1. When $i=1$ or $i=n-1$, the edge $(X_{(1)},X_{(2)})$ is a good edge if $Y_1 < Y_2$, which happens with probability $1/2$. Similarly, the edge $(X_{(n-1)},X_{(n)})$ is a good edge if $Y_{n-2} < Y_{n-1}$, which also happens with probability $1/2$.

2. When $i\ne 1,n-1$, the edge $(X_{(i)},X_{(i+1)})$ is a good edge if $Y_i < Y_{i+1}$ and $Y_i < Y_{i-1}$. In other words, $Y_i$ is the minimum one among $Y_{i-1},Y_i,Y_{i+1}$, which happens with probability $1/3$.

Hence, we have $f(n) = 2\cdot ( 1/2\cdot 2 + 1/3\cdot (n-3))=2n/3$. 