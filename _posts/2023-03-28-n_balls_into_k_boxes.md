---
title: 'How Many Ways Can You Distribute Balls into Boxes?'
date: 2023-03-28
permalink: /posts/2023/03/n_balls_into_k_boxes/
comments: true
tags:
  - probability
  - combinatorics
---

In this blog post, we will examine the ways in which $n$ (labeled or unlabeled) balls can be distributed into $k$ (labeled or unlabeled) boxes, taking into account whether or not empty boxes are allowed. To provide a comprehensive overview of the results, we will begin with a table that summarizes all of the possible combinations. Our table was created with reference to a similar table presented in this [lecture note](http://epgp.inflibnet.ac.in/epgpdata/uploads/epgp_content/S000034ST/P001011/M028536/ET/1522153313OccupancyProblem.pdf). 

|                     | $k$ Labeled Boxes     | $k$ Labeled Boxes (no empty) | $k$ Unlabeled Boxes                       | $k$ Unlabeled Boxes (no empty) |
| ------------------- | --------------------- | ---------------------------- | ----------------------------------------- | ------------------------------ |
| $n$ Labeled Balls   | $k^n$                 | $k!{n \brace k}$             | $\sum_{i=1}^k {n \brace i}$               | ${n \brace k}$                 |
| $n$ Unlabeled Balls | ${n+k-1 \choose k-1}$ | ${n-1 \choose k-1}$          | $\sum_{i=1}^k \left\|{n \atop i}\right\|$ | $\left\|n \atop k\right\|$     |

# Unlabeled Boxes

The last two columns of the table correspond to situations where we have unlabeled boxes. When there are k unlabeled boxes that can be empty (the third column), the number of ways to put balls in them is simply the sum of i from 1 to k. Therefore, we focus on the case where the boxes cannot be empty (the fourth column).

The notation ${n \brace k}$ (also known as Stirling numbers of the second kind) and $\left|n \atop k\right|$ are used by definition. We can calculate these numbers using recurrence relations and initial conditions as follows:

- For $n \geq 0$, we have ${n \brace n} = \left|n \atop n\right| = 1$.
- For $n > 0$, we have ${n \brace 0} = \left|n \atop 0\right| = 0$.
- For $0 < k < n$, we have:

$$
\begin{aligned}
{n \brace k} &= k{n-1 \brace k} + {n-1 \brace k-1} \\
\left|n \atop k\right| &= \left|n-1 \atop k-1\right| + \left|n-k \atop k\right|
\end{aligned}
$$

These formulas can be used to recursively compute the Stirling numbers and $\left|n \atop k\right|$ for any given values of n and k.

(to be continued)
