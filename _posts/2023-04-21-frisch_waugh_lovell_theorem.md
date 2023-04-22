---
title: 'Frisch–Waugh–Lovell theorem'
date: 2023-04-21
permalink: /posts/2023/04/frisch_waugh_lovell_theorem/
comments: true
tags:
  - statistics
  - linear-regression
  - regression
---

This post was partially inspired by [this great tutorial](http://mtor.sci.yorku.ca/MATH4939/files/Regression_Review/Three_Basic_Theorems.pdf). We begin with the projection theorem.

## Projection Theorem
Let $X\in \mathbb{R}^{n\times p}$ be a matrix of full column rank and $y\in \mathbb{R}^n$ be a column vector. Then the vector $b$ is the OLS solution to regressing $y$ against $X$ if and only if there exists a vector $e\in\mathbb{R}^n$ such that $e\perp\mathrm{colspace}(X)$ and $y=Xb+e$. 

## Proof

**"If" part**: If $y=Xb+e$ and $e\perp\mathrm{colspace}(X)$, we have 

$$
\hat{\beta} = (X^\top X)^{-1}X^\top y = (X^\top X)^{-1}X^\top (Xb+e) = b\,.
$$

**"Only if" part**: If $b = (X^\top X)^{-1}X^\top y$, then let $e=y-Xb=(I-X(X^\top X)^{-1}X^\top)y$. Since $X^\top e=0$, we have $e\perp\mathrm{colspace}(X)$.

## Frisch-Waugh-Lovell (FWL) Theorem

The OLS solution $\hat{\beta}_2$ to the regression problem $y \sim X_1\beta_1+X_2\beta_2$ can be obtained by solving $M_1y\sim M_1X_2\beta_2$, where $M_1 = I - X_1(X_1^\top X_1)^{-1}X_1^\top$ is the residual maker matrix.

### Proof

Let $\hat{\beta}_1$ and $\hat{\beta}_2$ be the OLS solutions to the regression problem $y \sim X_1\beta_1+X_2\beta_2$. By the projection theorem, there exists $e\perp\mathrm{colspace}(X_1,X_2)$ such that $y=X_1\hat{\beta}_1+X_2\hat{\beta}_2+e$. Multiplying both sides by $M_1$ yields:

$$
M_1y=M_1X_1\hat{\beta}_1+M_1X_2\hat{\beta}_2+M_1e\,.
$$

Note that $(M_1e)^\top M_1X_2=0$. Therefore, $\hat{\beta}_2$ is also the OLS solution to the regression problem $M_1y\sim M_1X_2\beta_2$. 

## Application

By the FWL theorem, the OLS solution $\hat{\beta}_2$ to the regression problem $y \sim X_1\beta_1+X_2\beta_2$ is given by

$$
\hat{\beta}_2 = (X_2^\top M_1X_2)^{-1}X_2^\top M_1y\,.
$$

Now consider the regression problem $y\sim X\beta$. We can use the FWL to get the covariance matrix of $\hat{\beta}$. Let $M_i$ be the residual maker matrix of the submatrix obtained by deleting the $i$-th column of $X$. By the FWL theorem, we have

$$
\hat{\beta}_i = (X_i^\top M_iX_i)^{-1}X_i^\top M_iy
$$

where $X_i$ is the $i$-th column of $X$. Then we have

$$
\begin{aligned}
\mathrm{cov}(\hat{\beta}_i,\hat{\beta}_j) &= \mathrm{cov}((X_i^\top M_iX_i)^{-1}X_i^\top M_iy, (X_j^\top M_jX_j)^{-1}X_j^\top M_jy) \\
&=(X_i^\top M_iX_i)^{-1}X_i^\top M_i\mathrm{cov}(y, y)M_jX_j(X_j^\top M_jX_j)^{-1} \\
&= \sigma^2\frac{e_i^\top e_j}{\|e_i\|_2^2\|e_j\|_2^2}
\end{aligned}
$$
where $\sigma^2$ is the variance of noise, $e_i = M_iX_i$ is the residual vector of regressing $X_i$ against the remaining variables.

We can observe that the $(i,j)$ entry of the matrix $\sigma^2(X^\top X)^{-1}$ is actually the covariance between $\hat{\beta}_i$ and $\hat{\beta}_j$. For $i=j$, the $(i,i)$ entry is $\sigma^2/\|e_i\|_2^2$, which is specifically utilized in the $t$-test of $\hat{\beta}_i$.