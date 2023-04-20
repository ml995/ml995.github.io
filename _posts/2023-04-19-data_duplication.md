---
title: 'Effect of Data Duplication on Statistics'
date: 2023-04-19
permalink: /posts/2023/04/data_duplication/
comments: true
tags:
  - statistics
  - linear-regression
  - regression
---

**Question**: We perform linear regression of $y$ against $x$ with a total of $n$ samples. If we were to duplicate the $n$ samples and increase the sample size to $2n$, what effect would this have on the values of $R^2$, the correlation between $x$ and $y$, and the $t$-statistic of $x$?

**Solution**: We use the notation convention we used in [this post](/posts/2023/04/straight_line_regression_basic/). For readers' convenience, we review the notation convention below.

In straight line regression, we aim to fit $y$ by $\beta_1 x + \beta_0$.

To introduce the notation, let us define:

- $S_{xy,n} = \sum_{i}^n (x_i - \bar{x})(y_i - \bar{y})$
- $S_{xx,n} = \sum_{i=1}^n (x_i - \bar{x})^2$
- $S_{yy,n} = \sum_{i=1}^n (y_i - \bar{y})^2$
- $\bar{y}\_n = \frac{1}{n} \sum_{i=1}^n y_i$
- $ \bar{x}\_n = \frac{1}{n} \sum_{i=1}^n x_i $

The estimates for $\beta_0$ and $\beta_1$ are:

$$
\begin{align}
  \hat{\beta}_1 ={}& \frac{S_{xy}}{S_{xx}} \\
  \hat{\beta}_0 ={}& \bar{y} - \hat{\beta}_1 \bar{x}
\end{align}
$$

We can see that both the new $S_{xy}$ and the new $S_{xx}$ are twice of the original. So both $\hat{\beta}_1$ and $\hat{\beta}_0$ remain invariant. And according to [the post](/posts/2023/04/straight_line_regression_basic/), 

$$
R^2 = \frac{S_{xy}^2}{S_{xx}S_{yy}} = r^2
$$

We can see that $R^2$ and $r^2$ remain invariant as well. 

Now we look at the $t$-statistic of $x$. Recall that the $t$-statistic of $x$ is given by

$$
t_n = \frac{\bar{x}_n-\mu}{s_n/\sqrt{n}},
$$

where $s_n = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i-\bar{x})^2}$. Note that $\bar{x}_n$ remains invariant as we double the data. We have

$$
t_{2n}/t_n = \sqrt{2}s_{n}/s_{2n} = \sqrt{\frac{2n-1}{n-1}}
$$

which is approximately $\sqrt{2}$ if $n$ is large.

We verify the above result using R code.


```r
# Generate data
set.seed(123)
x <- rnorm(100)
y <- 2*x + rnorm(100)

# Calculate R-squared
fit <- lm(y ~ x)
r_squared <- summary(fit)$r.squared

# Calculate correlation coefficient
cor_coef <- cor(x, y)

# Perform t-test on x
t_test <- t.test(x)

# Duplicate samples and increase sample size to 200
x_new <- rep(x, 2)
y_new <- rep(y, 2)

# Calculate R-squared for new data
fit_new <- lm(y_new ~ x_new)
r_squared_new <- summary(fit_new)$r.squared

# Calculate correlation coefficient for new data
cor_coef_new <- cor(x_new, y_new)

# Perform t-test on x for new data
t_test_new <- t.test(x_new)

# Compare results
cat("Original R-squared:", r_squared, "\n")
cat("New R-squared:", r_squared_new, "\n")
cat("Original t-statistic:", t_test$statistic, "\n")
cat("New t-statistic:", t_test_new$statistic, "\n")
cat("Ratio of t-statistics:", t_test_new$statistic / t_test$statistic, "\n")
cat("Theoretical value of ratio of t-statistics:", sqrt((2 * 100 - 1) / (100 - 1)), "\n")

# Output:

# Original R-squared: 0.7721124 
# New R-squared: 0.7721124 
# Original t-statistic: 0.9904068 
# New t-statistic: 1.404179 
# Ratio of t-statistics: 1.41778 
# Theoretical value of ratio of t-statistics: 1.41778
```