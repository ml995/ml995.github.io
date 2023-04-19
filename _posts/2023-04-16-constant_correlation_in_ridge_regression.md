---
title: 'Constant Correlation between Predicted and Actual Values in Ridge Regression'
date: 2023-04-16
permalink: /posts/2023/04/constant_correlation_in_ridge_regression/
comments: true
tags:
  - statistics
  - ridge-regression
  - regression
---

**Question**: What might be the reason for the correlation $\mathrm{corr}(\hat{y},y)$ between the predicted values ($\hat{y}$) and actual values ($y$) to stay constant while changing the value of $\lambda$ in ridge regression?

**Answer**: Recall that in ridge regression, the predicted value is given by:

$$
\hat{y} = H(\lambda)y, \quad H(\lambda) = X(X^\top X+\lambda I)^{-1}X^\top.
$$

To answer this question, it's better to look at $H(\lambda)$ through the lens of singular value decomposition. Suppose the design matrix $X$ has singular value decomposition $X = UDV^\top$, where $U$ and $V$ are orthogonal matrices and $D$ is a rectangular diagonal matrix. We have:

$$
H(\lambda) = UD(D^\top D+\lambda I)^{-1}D^\top U^\top.
$$

To ensure that the correlation $\mathrm{corr}(\hat{y},y)$ remains constant, we need to ensure that for every $\lambda$, there exists $\mu(\lambda)$ such that

$$
(D^\top D+\lambda I)^{-1} = \mu(\lambda)(D^\top D)^{-1},
$$

which is equivalent to 

$$
(D^\top D)(D^\top D+\lambda I)^{-1} = \mu(\lambda).
$$

So we have 

$$
\frac{s_i^2}{s_i^2+\lambda} = \mu(\lambda),
$$

where $s_i$'s are the singular values of $X$, also the diagonal entries of $D$. Therefore, we conclude that the singular values are equal.


