---
title: 'Probability Interview Problem Set 1'
date: 2023-03-20
permalink: /posts/2023/03/probability_problem_set1/
comments: true
tags:
  - interview
---

In this blog post, I will present a list of interview problems related to probability. I intend to publish solutions for some of these problems in the future. I hope you find these problems interesting and enjoyable! I will continue to update this blog post by including additional problems and their respective solutions. The most recent update to this post was made on March 29, 2023.

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