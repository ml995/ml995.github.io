---
title: 'Fairness of a Dice Rolling Game: Three 6-Sided Dice vs One 20-Sided Die'
date: 2023-03-19
permalink: /posts/2023/03/fairness_of_a_dice_rolling_game_three_6_sided_dice_vs_one_20_sided_die/
comments: true
tags:
  - probability
---

**Problem**: In this game, Player A rolls three six-sided dice, while Player B rolls one twenty-sided die. Player A's score is the sum of the values on their three dice, whereas Player B's score is simply the face value of their die. The question is, is this a fair game?

**Solution**: At first glance, it's tempting to look at the expectations of the players' dice rolls. Let's denote the face value of Player A's three dice by $X_1$, $X_2$, and $X_3$, and the face value of Player B's die by $Y$. We have $X_1$, $X_2$, and $X_3$ uniformly distributed on $[6]$, and $Y$ uniformly distributed on $[20]$.

Computing the expectations, we find that $\mathbb{E}[X_1+X_2+X_3] = \frac{1+6}{2} \cdot 3 = \frac{21}{2}$, and $\mathbb{E}[Y] = \frac{1+20}{2} = \frac{21}{2}$. Surprisingly, these are the same! However, we can't conclude that the game is necessarily fair based on this alone. 

To determine if the game is fair, it's better to compute the winning probabilities of both players. Let's start with Player A. The probability that their total score beats Player B's score is given by:

$$
\Pr[X_1+X_2+X_3>Y] = \mathbb{E}\left[\frac{X_1+X_2+X_3-1}{20}\right] = \frac{19}{40}.
$$

The probability of a tie is:

$$
\Pr[X_1+X_2+X_3=Y] = \frac{1}{20}.
$$

Finally, the probability that Player B wins is:

$$
\Pr[X_1+X_2+X_3<Y] = 1 - \Pr[X_1+X_2+X_3>Y] - \Pr[X_1+X_2+X_3=Y] = \frac{19}{40}.
$$

Therefore, the game is fair.