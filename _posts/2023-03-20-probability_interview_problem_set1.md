---
title: 'Probability Interview Problem Set 1'
date: 2023-03-20
permalink: /posts/2023/03/probability_problem_set1/
comments: true
tags:
  - interview
---

In this blog post, I will present a list of interview problems related to probability. I intend to publish solutions for some of these problems in the future. I hope you find these problems interesting and enjoyable! I will continue to update this blog post by including additional problems and their respective solutions. The most recent update to this post was made on March 30, 2023.

# SmFuZSBTdHJlZXQ=

**Problem 1.1:** You play a game in which you throw a dice repeatedly, and your payoff is the cumulative sum of the points you throw. However, if the sum of the points you throw is a perfect square, you lose the game. You have the option to stop the game at any time and take the amount of money equal to the current sum. Currently, your score is 35. Would you like to continue playing the game?

This problem was discussed [here](https://web.archive.org/web/20230320233237/https://math.stackexchange.com/questions/977679/toss-a-fair-die-until-the-cumulative-sum-is-a-perfect-square-expected-value) and [here](https://web.archive.org/web/20230320233310/https://math.stackexchange.com/questions/1176195/would-you-ever-stop-rolling-the-die).



**Problem 1.2:** You and your opponent each roll a dice randomly. While you can see the value of one of the dice, your opponent can see the value of the other. Money equal to the sum of the two dice is placed in a box. You and your opponent take turns bidding to buy the box. The question is, who should bid first? Additionally, assuming that you know the value you can see on your dice, what is the optimal bidding strategy for you?

This problem was discussed [here](https://web.archive.org/web/20230320233056/https://math.stackexchange.com/questions/3406588/optimal-strategy-of-a-dice-game).



**Problem 1.3:** What is the first date from today onwards that has eight unique digits in its `YYYY:MM:DD` format?



**Problem 1.4:** You are playing a game with a 100-sided dice. Each time you roll the dice, you can either earn money equal to the number on the dice or pay one dollar to roll again. What is the expected value of this game?

If the dice is replaced with a 1000-sided dice, how would the expected value change? Would it be larger or smaller than 10 times the expected value of the 100-sided dice?

Alternatively, if you have 10 10-sided dice instead, what would be the expected value range of this game?



**Problem 1.5:** What is the expected number of rolls required to obtain all six numbers when rolling a dice? What is the probability that the product of two dice is a perfect square?



**Problem 1.6:** You have four coins. What is the probability of obtaining two heads? What is the probability of obtaining an even number of heads?



**Problem 1.7:** What is the smallest integer whose digits multiply to 96?

This problem was discussed [here](https://web.archive.org/web/20230320233451/https://www.glassdoor.com/Interview/I-am-thinking-of-a-number-that-does-not-contain-the-digit-1-The-product-of-the-digits-is-96-What-is-the-smallest-possible-QTN_129149.htm#:~:text=The%20smallest%20number%20we%20can,96%20and%20contain%20no%201's.).



**Problem 1.8:** Suppose you flip four coins. What is the probability of obtaining an even number of heads? If you flip nine coins, what is the probability of obtaining an even number of heads? If you flip an odd number of coins, what is the probability of obtaining an even number of heads? Lastly, what is the probability of obtaining an even number of heads when flipping an even number of coins?

# SFJU

**Problem 2.1:** Consider the sum of 100 dice rolls, denoted by $X$. Find $\Pr(X\ge 400)$. 

*Solution 2.1:* We can use the normal approximation of $X$, which is $X'\sim \mathcal{N}(100\cdot 3.5, 100\cdot \frac{6^2-1}{12})=\mathcal{N}(350, 875/3)$. Then, we have:

$$
\Pr(X\ge 400)\approx \Pr(X'\ge 400) = 1-\Pr(X'<400) = 1-\Phi\left(\frac{400-\mu}{\sigma}\right) = 1- \Phi\left(\frac{2\sqrt{105}}{7}\right) \approx 0.001707,
$$

where $\mu = 350$ and $\sigma^2 = 875/3$. Here, $\Phi$ is the CDF of the standard normal distribution.



**Problem 2.2:** Consider the sum of rolling a die 100 times, denoted by $X$, and flipping a coin 600 times and counting the number of heads, denoted by $Y$. Find the probability that $X$ is greater than $Y$, i.e., find $\Pr(X > Y)$.

*Solution 2.2:* Let us again use normal approximation. The normal approximation for $X$ is $X' \sim \mathcal{N}(100\cdot 3.5, 100\cdot \frac{6^2-1}{12})$, and that for $Y$ is $Y' \sim \mathcal{N}(600\cdot 1/2, 600\cdot 1/2 \cdot 1/2)$. Simplifying them, we get $X' \sim \mathcal{N}(350, 875/3)$ and $Y' \sim \mathcal{N}(300, 150)$. We have $Z = X' - Y' \sim \mathcal{N}(350 - 300,875/3 + 150)$. Therefore,

$$
\Pr(X > Y) \approx \Pr(X' > Y') = \Pr(Z > 0) = 1 - \Phi\left(\frac{-50}{\sqrt{875/3 + 150}}\right) \approx 0.9913.
$$



**Problem 2.3:** Suppose we conduct two linear regressions on the sets of paired data $(x_i, y_i)$. One regression generates an equation in the form of $y = ax + b_1$, while the other regression results in an equation of the form $x = by + b_2$. What is the relationship between the coefficients $a$ and $b$ in these equations?



**Problem 2.4:** We assume the heights of people on a street are iid. We randomly select a person with height $X$ and sample heights until finding someone taller. Let $N$ be the number of people we look at. What is the expected value of $N$?

*Solution 2.4:* Notice that $N$ is a geometric random variable conditioned on $X$, with $N\mid X\sim \mathrm{Geom}(1-F(X))$, where $F$ is the cumulative distribution function of the height distribution. Then, we have:

$$
\begin{aligned}
\mathbb{E}[N] &= \mathbb{E}\left[\frac{1}{1-F(X)}\right]\\
&= \int_{t=-\infty}^{\infty} \frac{1}{1-F(t)} dF(t) \\
&= -\ln(1-F(t))\bigg|_{t=-\infty}^{\infty} \\
&= \infty.
\end{aligned}
$$

In other words, the expected value of $N$ is infinite, which means that on average, we would need to look at an infinite number of people before finding someone taller than $X$.



**Problem 2.5:** A white cube with sides measuring 3 units each is painted red on all of its surfaces. The cube is then cut into 27 smaller cubes, each measuring 1 unit on each side. If someone randomly selects one of the smaller cubes and throws it onto a table, and five of its faces are observed to be white, what is the probability that the last face is red?

*Solution 2.5:* We are given a set of small cubes, each of which can have at most three red faces. Out of the 27 cubes, one has no red faces, six have one red face, twelve have two red faces, and eight have three red faces.

Let $O$ denote the event of observing that five of the faces of a randomly selected cube are white. Let $R$ denote the number of red faces of the chosen cube. We know that $\Pr(O\mid R=0)=1$, $\Pr(O\mid R=1)=1/6$ (the probability that the face that lands on the table is the red one and hence remains invisible), and $\Pr(O\mid R=2) = \Pr(O\mid R=3) = 0$. 

Given the observation $O$, the event that the last face is red is exactly the event that $R=1$. Using Bayes' theorem, we have:

$$
\Pr(R=1\mid O) = \frac{\Pr(O\mid R=1)\Pr(R=1)}{\sum_{i=0}^3 \Pr(O\mid R=i)\Pr(R=i)} = \frac{1/6\cdot 6/27}{1/6\cdot 6/27+1\cdot 1/27} = \frac{1}{2}.
$$

Therefore, the probability that the last face of the cube is red, given that five of its faces are observed to be white, is $1/2$.



**Problem 2.6:** On a circular table, $x$ individuals from country A and $y$ individuals from country B are seated randomly. Each person from the same country shakes hands with only those from their own country, and only with the person to their left or right. What is the probability of two randomly chosen individuals shaking hands?

*Solution 2.6:* The probability of a handshake is the product of the probability of two people sitting next to each other and the probability of them being from the same country. The probability of two people sitting next to each other is $\frac{x+y}{\binom{x+y}{2}}$, and the probability of them being from the same country is $\frac{\binom{x}{2}+\binom{y}{2}}{\binom{x+y}{2}}$. Therefore, the probability of a handshake is:

$$
\frac{x+y}{\binom{x+y}{2}}\cdot \frac{\binom{x}{2}+\binom{y}{2}}{\binom{x+y}{2}} = \frac{2(x(x-1)+y(y-1))}{(x+y)(x+y-1)^2}.
$$



**Problem 2.7**: Given an $m \times n$ 2D array where all the numbers in the array are in increasing order from left to right and top to bottom, provide an algorithm that searches and determines if a target number is in the array. Also, provide a lower bound for the time complexity of this problem.

This problem was discussed [here](https://web.archive.org/web/20230401024022/https://stackoverflow.com/questions/2457792/how-do-i-search-for-a-number-in-a-2d-array-sorted-left-to-right-and-top-to-botto).



**Problem 2.8**: Suppose there is a one-way road with $N$ cars travelling on it, and each car has a constant speed sampled independently and identically from a continuous distribution. When a car at the front of the line has a lower speed than the cars behind it, those cars become blocked behind it. What is the expected number of these blocks?

This problem was discussed [here](https://web.archive.org/web/20230401195156/https://math.stackexchange.com/questions/201807/probability-problem-cars-on-the-road).



**Problem 2.9**: Find $\Pr(X>3Y)$ where $X,Y\sim \mathcal{N}(0,1)$ are independent standard normal random variables.

*Solution 2.9*: Let $Z = X - 3Y$. Then $Z$ is also a normal random variable with mean 0 and variance 4. Therefore, the answer is $\frac{1}{2}$.

# Others

**Problem 3.1**: Let $n$ be a positive integer. Define $f(n)$ to be the number of non-negative integer solutions $(x,y,z)$ to the equation $x+y+z=n$, where $x\le y\le z$. Find the limit of $\frac{f(n)}{n^2}$ as $n\to \infty$.

# Q2l0YWRlbA==

**Problem 4.1**: Examples of two random variables that are uncorrelated but dependent.

*Solution 4.1*: $X\sim \mathbb{N}(0,1)$, $Y=X^2$. In this case, $\mathrm{cov}(X,Y)=\mathbb{E}[X^3]=0$ but $X,Y$ are clearly dependent.

**Problem 4.2**: You can roll a dice a maximum of three times and get the amount shown on the dice face. After each roll, you can choose to stop playing and take the money earned without continuing to roll. The money earned from each roll cannot be added up. What is the anticipated amount of money you can win from this game?

*Solution 4.2*: $14/3$. This problem was discussed [here](https://web.archive.org/web/20230402182156/https://math.stackexchange.com/questions/179534/the-expected-payoff-of-a-dice-game:).

**Problem 4.3**: Explain Kelly criteron.

**Problem 4.4**: Assuming you have a uniform random integer generator that generates integers between 1 and 5, how can you generate a uniform random integer between 1 and 7?

*Solution 4.4*: To get a random integer between 1 and 7 from two random integers between 1 and 5, use the first 21 of the 25 possible ordered pairs. Each set of three pairs represents a number between 1 and 7. Discard any combinations outside the first 21.

**Problem 4.5**: How to uniformly sample within a unit ball in a 3-dimensional space? What is the marginal distribution along the $x$-axis?

*Solution 4.5*: We can use three methods to generate random samples in a unit ball. The first method involves rejection sampling by sampling from a hypercube, while the second method uses spherical coordinates.

To use the spherical coordinate method, we first define the coordinates as:

$$
\begin{align}
  x &= r\sin \theta \cos \phi \\
  y &= r\sin \theta \sin \phi \\
  z &= r \cos \theta
\end{align}
$$

The volume element in spherical coordinates is $dxdydz = r^2 \sin \theta drd\theta d\phi$, where $r$ has a density proportional to $r^2$ and $\theta$ has a density proportional to $\sin \theta$. We can use the CDFs of $r$ and $\theta$ to generate random samples using inverse transform sampling. The CDF of $r$ is given by $\int_0^x 3r^2 dr = x^3$, while the CDF of $\theta$ is $\int_0^x \frac{\sin \theta}{2} d\theta = \frac{1}{2}(1-\cos x)$.

Using inverse transform sampling, we can generate $\phi \sim \mathrm{Unif}([0,2\pi])$, $r= U_1^{1/3}$, $\theta=\arccos(1-2 U_2)$, where $U_1,U_2\sim \mathrm{Unif}([0,1])$.

The third method is to use $U^{1/3}\frac{\mathbf{x}}{\|\mathbf{x}\|_2}$, where $U\sim \mathrm{Unif}([0,1])$ and $\mathbf{x}\sim \mathcal{N}(0,I_3)$.


## Programming

- LeetCode `Best Time to Buy and Sell Stock I-IV`. Not only must the algorithm produce the highest possible profit, but it must also identify the optimal buying and selling prices for the stock. Moreover, it is imperative to conduct testing using test cases generated by the interviewee.
- LeetCode `53. Maximum Subarray`.
- LeetCode `560. Subarray Sum Equals K`
- The problem of hash collisions in HashMap, and how to solve them.