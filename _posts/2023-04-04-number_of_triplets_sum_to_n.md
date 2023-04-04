---
title: 'Counting Triplets with a Sum Constraint'
date: 2023-04-04
permalink: /posts/2023/04/number_of_triplets_sum_to_n/
comments: true
tags:
  - combinatorics
---

**Problem**: Let $f(n)$ denote the number of triplets $(x, y, z)$ of non-negative integers, where $x \leq y \leq z$, that satisfy the equation $x + y + z = n$. Find $\lim_{n\to \infty} \frac{f(n)}{n^2}$.

**Solution**: We utilize analytic combinatorics to solve this problem, where we define $a=x$, $b=y-x$, and $c=z-y$, with $a,b,c\ge0$. We need to find the number of solutions for $c+2b+3a=n$, which is the coefficient of $u^{n}$ in the Taylor expansion of the function 

$$
f(u)=\frac{1}{(1-u)(1-u^{2})(1-u^{3})}=\frac{1}{(1-u)^{3}(1+u)(u-w)(u-w^{2})},
$$

where $w=\frac{-1+\sqrt{3}i}{2}$. We use $[u^n]f(u)$ to denote the coefficient of $u^n$ in the Taylor expansion of $f(u)$. 

The function $f(u)$ has four dominant singularities at $u=1,-1,w,w^2$. As $u\to1$, $f(u)\sim\frac{1}{6}(1-u)^{-3}$, and as $u\to -1$, $f(u)\sim  \frac{1}{8}(1+u)^{-1}$. Similarly, as $u\to w$, $f(u)\sim \bar{c_0}(1-w)^{-1}$, and as $u\to w^2$, $f(u)\sim c_0(1-w^2)^{-1}$, where $c_0 = (\frac{1}{18}+\frac{1}{6\sqrt{3}}i)$ and $\bar{c_0}$ is its complex conjugate. 

Using the result 

$$
[u^n](1-u)^{-\alpha} \sim \frac{n^{\alpha-1}}{\Gamma(\alpha)} ,
$$

where $\Gamma(\alpha)$ is the gamma function, we find that $[u^n]f(u)\sim\frac{1}{6\Gamma(3)}n^{2}=\frac{1}{12}n^{2}$, since $u=1$ is the pole of highest order with a pole of order $3$ in the function $f(u)$ and its corresponding term in $[u^n]f(u)$ dominates. 

