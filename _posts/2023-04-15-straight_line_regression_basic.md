---
title: 'Straight Line Regression: Basic Knowledge and Notation'
date: 2023-04-15
permalink: /posts/2023/04/straight_line_regression_basic/
comments: true
tags:
  - statistics
  - linear-regression
  - regression
---

Straight line regression is a common topic in interviews, so it is essential to understand the basics. A helpful reference for this topic is available at [this link](https://web.archive.org/web/20230416020029/https://mathworld.wolfram.com/CorrelationCoefficient.html). In straight line regression, we aim to fit $y$ by $\beta_1 x + \beta_0$.

To introduce the notation, let us define:

- $S_{xy} = \sum_{i}^n (x_i - \bar{x})(y_i - \bar{y})$
- $S_{xx} = \sum_{i=1}^n (x_i - \bar{x})^2$
- $S_{yy} = \sum_{i=1}^n (y_i - \bar{y})^2$
- $\bar{y} = \frac{1}{n} \sum_{i=1}^n y_i$
- $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$

The estimates for $\beta_0$ and $\beta_1$ are:

$$
\begin{align}
  \hat{\beta}_1 ={}& \frac{S_{xy}}{S_{xx}} \\
\hat{\beta}_0 ={}& \bar{y} - \hat{\beta}_1 \bar{x}
\end{align}
$$

From the expression of $\hat{\beta}\_1$, we can observe that the product of the regression coefficient of $y$ against $x$ and the coefficient of $x$ against $y$ is $\frac{S_{xy}^2}{S_{xx}S_{yy}} = r^2 \leq 1$, where $r^2$ represents the sample correlation coefficient between $x$ and $y$.

We can define $\bar{\hat{y}} = \frac{1}{n} \sum_{i=1}^n \hat{y}_i$, where $\hat{y}_i$ is the predicted value of $y$ based on $\beta_0$ and $\beta_1$. Note that $\bar{\hat{y}} = \bar{y}$. 

The explained sum of squares (ESS) is given by:

$$
\mathrm{ESS} = \sum_{i=1}^n (\hat{y}_i - \bar{y})^2 = \mathrm{Var}[\hat{y}] = \mathrm{Var}[\hat{\beta}_1 x] = \hat{\beta}_1^2 \mathrm{Var}[x] = \frac{S_{xy}^2}{S_{xx}}
$$

The total sum of squares (TSS) is given by:

$$
\mathrm{TSS} = S_{yy}
$$

Therefore, the coefficient of determination $R^2$ is given by:

$$
R^2 = \frac{\mathrm{ESS}}{\mathrm{TSS}} = \frac{S_{xy}^2}{S_{xx}S_{yy}} = r^2
$$

Thus, in straight line regression with an intercept, $R^2 = r^2$.