---
title: 'Sign of Correlation between Standard Normal Random Variables Conditioned on Negative Sum'
date: 2023-03-20
permalink: /posts/2023/03/correlation_negative_sum/
comments: true
tags:
  - probability
  - statistics
---



**Problem:** Suppose we have two independent and identically distributed random variables $X$ and $Y$, both following the standard normal distribution. We take $1,000,000$ sample pairs and want to determine the sign of the correlation between $X$ and $Y$ in the pairs where $X+Y<0$.

**Solution:**
We will solve this problem in two steps. Firstly, the problem asks for the correlation between two vectors, denoted by $r$, which is the sample correlation and given by

$$
r = \frac{\sum_i (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_i (X_i - \bar{X})^2 \sum_i (Y_i - \bar{Y})^2}}.
$$

Also, we recall that the correlation between two random variables $X$ and $Y$ is given by

$$
\rho = \frac{\mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]}{\sqrt{\mathrm{Var}[X]\mathrm{Var}[Y]}}.
$$

Unfortunately, $r$ is a biased estimator of $\rho$. However, it is asymptotically unbiased, and quantitatively,

$$
\mathbb{E}[r] = \rho \left(1 - \frac{1 - \rho^2}{n^2} + O\left(\frac{1}{n^2}\right)\right).
$$

You can find further details on the above equation on page 96 of the book titled *Theory of Point Estimation* by Lehmann and George Casella, published in 1998.

Since $n = 1,000,000$ in this problem is very large, it suffices to study the sign of the correlation conditioned on $X + Y < 0$. Moreover, it suffices to study the sign of $\mathbb{E}[XY \mid X+Y < 0] - \mathbb{E}[X \mid X+Y < 0]\mathbb{E}[Y \mid X+Y < 0]$. In an interview, one rarely has time to compute them using integrals, so we resort to a quick geometric method.

We illustrate the area in the following figure, divided into 4 octants and labeled as $A$, $B$, $C$, and $D$.

![Illustration of the area X + Y < 0](/images/xy.svg)

To compute $\mathbb{E}[XY\mid X+Y<0]$, we calculate the expected value of $XY$ on the gray area $X+Y<0$ under the Gaussian measure. Note that $A$ and $B$ are symmetric with respect to the $x$-axis, meaning there is a bijection $(x,y)\leftrightarrow (x,-y)$ between $A$ and $B$. Thus, the expected value of $XY$ on $A\cup B$ is 0. Similarly, $C$ and $B$ are symmetric with respect to the $y$-axis, meaning there is a bijection $(x,y)\leftrightarrow (-x,y)$ between $C$ and $D$. Hence, the expected value of $XY$ on $C\cup D$ is also 0. Therefore, we conclude that $\mathbb{E}[XY\mid X+Y<0]=0$.

Next, we consider $\mathbb{E}[X\mid X+Y<0]$. Due to the bijection $(x,y)\leftrightarrow (-x,y)$ between $C$ and $D$, the expected value of $X$ on $C\cup D$ is 0. Moreover, since $X$ is negative on $A\cup B$, we have $\mathbb{E}[X\mid X+Y<0]<0$. Similarly, we have $\mathbb{E}[Y\mid X+Y<0]<0$.

Thus, we conclude that $\mathbb{E}[XY\mid X+Y<0]-\mathbb{E}[X\mid X+Y<0]\mathbb{E}[Y\mid X+Y<0]<0$. Therefore, the sign of correlation is negative.
