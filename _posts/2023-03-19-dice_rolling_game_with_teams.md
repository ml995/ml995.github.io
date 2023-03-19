---
title: 'Dice Rolling Game with Teams and Payoffs'
date: 2023-03-19
permalink: /posts/2023/03/dice_rolling_game_with_teams/
comments: true
tags:
  - probability
---

**Problem**: Four players take turns rolling a fair 30-sided die once. The four numbers rolled are all different (if a later player rolls the same number as a previous player, they must re-roll until they get a different number). The highest and lowest numbers are paired together, as are the second-highest and third-highest numbers. The team with the lower average score is the loser, and the loser must pay the winner an amount equal to the average of the winning team's score.

As the first player to roll the dice, what number do you most want to roll?

**Solution**: We consider the continuous version of this problem. That is, four 
players draw numbers from the uniform distribution on the interval 
$[0,1]$. Let $X_{(1)}, X_{(2)}, X_{(3)}, X_{(4)}$ be the order 
statistics of these four numbers, sorted from smallest to largest. 
Suppose we are the first player and we draw a number $x$. If $x$ is 
known to be the smallest of the four numbers, then our expected payoff 
$\mathbb{E}[(X_{(1)}+X_{(4)})-(X_{(2)}+X_{(3)})\mid X_{(1)}=x]/2$ is 
equal to

$$
[(x+(x+\frac{3}{4}(1-x)))-((x+\frac{1}{4}(1-x))+(x+\frac{2}{4}(1-x)))]/2=0
$$

Similarly, if $x$ is the largest of the four numbers, the expected payoff is also 0.

Next, we consider the case where $x$ is the second largest or the third largest number. First, the probability that $x$ is the second largest is $\Pr[X_{2}=x]=x(1-x)^2$. In this case, our expected payoff is $\mathbb{E}[(X_{(2)}+X_{(3)})-(X_{(1)}+X_{(4)})\mid X_{(2)}=x]/2$, which equals:

$$
[(x+(x+\frac{1-x}{3}))-(\frac{x}{2}+(x+\frac{2(1-x)}{3})]/2=\frac{5x-2}{12}
$$

Similarly, the probability that $x$ is the third largest is 
$\Pr[X_{3}=x]=x^2(1-x)$. In this case, our expected payoff is 
$\mathbb{E}[(X_{(2)}+X_{(3)})-(X_{(1)}+X_{(4)})\mid X_{(2)}=x]/2$, which
 equals:

$$
[(x+\frac{2}{3}x)-(\frac{1}{3}x+\frac{1+x}{2})]/2=\frac{5x-3}{12}
$$

Therefore, our overall expected payoff is:

$$
f(x)=x(1-x)^2\cdot \frac{5x-2}{12}+x^2(1-x)\cdot \frac{5x-3}{12}=-x/6 + x^2/2 - x^3/3
$$

The following is a graph of $f(x)$. Its maximum value is achieved at $x=1/2+1/(2\sqrt{3})$, and the minimum value is achieved at $x=1/2-1/(2\sqrt{3})$. For the original problem, we compute $30\cdot (1/2+1/(2\sqrt{3}))\approx 23.66$ and $30\cdot 
(1/2-1/(2\sqrt{3}))\approx 6.34$. Therefore, for the original problem, due to rounding errors when converting from the continuous problem to the discrete problem, the maximum value should be achieved at either 23 or 24. After conducting simulations for the original problem, we found that the maximum value is achieved at 24 and the minimum value is achieved at 7.

![](/images/dice_rolling_f.svg)