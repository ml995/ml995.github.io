---
title: 'Probability Interview Problem Set 1'
date: 2023-03-20
permalink: /posts/2023/03/probability_problem_set1/
comments: true
tags:
  - interview
---

In this blog post, I will present a list of interview problems related to probability. I intend to publish solutions for some of these problems in the future. I hope you find these problems interesting and enjoyable! I will continue to update this blog post by including additional problems and their respective solutions. The most recent update to this post was made on March 30, 2023.

# `SmFuZSBTdHJlZXQ=`

**Problem 1.1:** You play a game in which you throw a dice repeatedly, and your payoff is the cumulative sum of the points you throw. However, if the sum of the points you throw is a perfect square, you lose the game. You have the option to stop the game at any time and take the amount of money equal to the current sum. Currently, your score is 35. Would you like to continue playing the game?

This problem was discussed [here](https://web.archive.org/web/20230320233237/https://math.stackexchange.com/questions/977679/toss-a-fair-die-until-the-cumulative-sum-is-a-perfect-square-expected-value) and [here](https://web.archive.org/web/20230320233310/https://math.stackexchange.com/questions/1176195/would-you-ever-stop-rolling-the-die).

-----

**Problem 1.2:** You and your opponent each roll a dice randomly. While you can see the value of one of the dice, your opponent can see the value of the other. Money equal to the sum of the two dice is placed in a box. You and your opponent take turns bidding to buy the box. The question is, who should bid first? Additionally, assuming that you know the value you can see on your dice, what is the optimal bidding strategy for you?

This problem was discussed [here](https://web.archive.org/web/20230320233056/https://math.stackexchange.com/questions/3406588/optimal-strategy-of-a-dice-game).

-----

**Problem 1.3:** What is the first date from today onwards that has eight unique digits in its `YYYY:MM:DD` format?

-----

**Problem 1.4:** You are playing a game with a 100-sided dice. Each time you roll the dice, you can either earn money equal to the number on the dice or pay one dollar to roll again. What is the expected value of this game?

If the dice is replaced with a 1000-sided dice, how would the expected value change? Would it be larger or smaller than 10 times the expected value of the 100-sided dice?

Alternatively, if you have 10 10-sided dice instead, what would be the expected value range of this game?

-----

**Problem 1.5:** What is the expected number of rolls required to obtain all six numbers when rolling a dice? What is the probability that the product of two dice is a perfect square?

-----

**Problem 1.6:** You have four coins. What is the probability of obtaining two heads? What is the probability of obtaining an even number of heads?

-----

**Problem 1.7:** What is the smallest integer whose digits multiply to 96?

This problem was discussed [here](https://web.archive.org/web/20230320233451/https://www.glassdoor.com/Interview/I-am-thinking-of-a-number-that-does-not-contain-the-digit-1-The-product-of-the-digits-is-96-What-is-the-smallest-possible-QTN_129149.htm#:~:text=The%20smallest%20number%20we%20can,96%20and%20contain%20no%201's.).

-----

**Problem 1.8:** Suppose you flip four coins. What is the probability of obtaining an even number of heads? If you flip nine coins, what is the probability of obtaining an even number of heads? If you flip an odd number of coins, what is the probability of obtaining an even number of heads? Lastly, what is the probability of obtaining an even number of heads when flipping an even number of coins?

# `SFJU`

**Problem 2.1:** Consider the sum of 100 dice rolls, denoted by $X$. Find $\Pr(X\ge 400)$. 

*Solution 2.1:* We can use the normal approximation of $X$, which is $X'\sim \mathcal{N}(100\cdot 3.5, 100\cdot \frac{6^2-1}{12})=\mathcal{N}(350, 875/3)$. Then, we have:

$$
\Pr(X\ge 400)\approx \Pr(X'\ge 400) = 1-\Pr(X'<400) = 1-\Phi\left(\frac{400-\mu}{\sigma}\right) = 1- \Phi\left(\frac{2\sqrt{105}}{7}\right) \approx 0.001707,
$$

where $\mu = 350$ and $\sigma^2 = 875/3$. Here, $\Phi$ is the CDF of the standard normal distribution.

-----

**Problem 2.2:** Consider the sum of rolling a die 100 times, denoted by $X$, and flipping a coin 600 times and counting the number of heads, denoted by $Y$. Find the probability that $X$ is greater than $Y$, i.e., find $\Pr(X > Y)$.

*Solution 2.2:* Let us again use normal approximation. The normal approximation for $X$ is $X' \sim \mathcal{N}(100\cdot 3.5, 100\cdot \frac{6^2-1}{12})$, and that for $Y$ is $Y' \sim \mathcal{N}(600\cdot 1/2, 600\cdot 1/2 \cdot 1/2)$. Simplifying them, we get $X' \sim \mathcal{N}(350, 875/3)$ and $Y' \sim \mathcal{N}(300, 150)$. We have $Z = X' - Y' \sim \mathcal{N}(350 - 300,875/3 + 150)$. Therefore,

$$
\Pr(X > Y) \approx \Pr(X' > Y') = \Pr(Z > 0) = 1 - \Phi\left(\frac{-50}{\sqrt{875/3 + 150}}\right) \approx 0.9913.
$$

-----

**Problem 2.3:** Suppose we conduct two linear regressions on the sets of paired data $(x_i, y_i)$. One regression generates an equation in the form of $y = ax + b_1$, while the other regression results in an equation of the form $x = by + b_2$. What is the relationship between the coefficients $a$ and $b$ in these equations?

-----

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

-----

**Problem 2.5:** A white cube with sides measuring 3 units each is painted red on all of its surfaces. The cube is then cut into 27 smaller cubes, each measuring 1 unit on each side. If someone randomly selects one of the smaller cubes and throws it onto a table, and five of its faces are observed to be white, what is the probability that the last face is red?

*Solution 2.5:* We are given a set of small cubes, each of which can have at most three red faces. Out of the 27 cubes, one has no red faces, six have one red face, twelve have two red faces, and eight have three red faces.

Let $O$ denote the event of observing that five of the faces of a randomly selected cube are white. Let $R$ denote the number of red faces of the chosen cube. We know that $\Pr(O\mid R=0)=1$, $\Pr(O\mid R=1)=1/6$ (the probability that the face that lands on the table is the red one and hence remains invisible), and $\Pr(O\mid R=2) = \Pr(O\mid R=3) = 0$. 

Given the observation $O$, the event that the last face is red is exactly the event that $R=1$. Using Bayes' theorem, we have:

$$
\Pr(R=1\mid O) = \frac{\Pr(O\mid R=1)\Pr(R=1)}{\sum_{i=0}^3 \Pr(O\mid R=i)\Pr(R=i)} = \frac{1/6\cdot 6/27}{1/6\cdot 6/27+1\cdot 1/27} = \frac{1}{2}.
$$

Therefore, the probability that the last face of the cube is red, given that five of its faces are observed to be white, is $1/2$.

-----

**Problem 2.6:** On a circular table, $x$ individuals from country A and $y$ individuals from country B are seated randomly. Each person from the same country shakes hands with only those from their own country, and only with the person to their left or right. What is the probability of two randomly chosen individuals shaking hands?

*Solution 2.6:* The probability of a handshake is the product of the probability of two people sitting next to each other and the probability of them being from the same country. The probability of two people sitting next to each other is $\frac{x+y}{\binom{x+y}{2}}$, and the probability of them being from the same country is $\frac{\binom{x}{2}+\binom{y}{2}}{\binom{x+y}{2}}$. Therefore, the probability of a handshake is:

$$
\frac{x+y}{\binom{x+y}{2}}\cdot \frac{\binom{x}{2}+\binom{y}{2}}{\binom{x+y}{2}} = \frac{2(x(x-1)+y(y-1))}{(x+y)(x+y-1)^2}.
$$