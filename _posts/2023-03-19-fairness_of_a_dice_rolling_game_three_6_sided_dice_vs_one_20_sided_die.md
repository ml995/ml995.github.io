---
title: 'Fairness of a Dice Rolling Game: Three 6-Sided Dice vs One 20-Sided Die'
date: 2023-03-19
permalink: /posts/2023/03/fairness_of_a_dice_rolling_game_three_6_sided_dice_vs_one_20_sided_die/
comments: true
tags:
  - probability
---

**Problem 1**: In this game, Player A rolls three six-sided dice, while Player B rolls one twenty-sided die. Player A's score is the sum of the values on their three dice, whereas Player B's score is simply the face value of their die. The question is, is this a fair game?

*Solution 1*: At first glance, it's tempting to look at the expectations of the players' dice rolls. Let's denote the face value of Player A's three dice by $X_1$, $X_2$, and $X_3$, and the face value of Player B's die by $Y$. We have $X_1$, $X_2$, and $X_3$ uniformly distributed on $[6]$, and $Y$ uniformly distributed on $[20]$.

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

**Problem 2**: In this game, Player A rolls three six-sided dice, while Players B and C each roll one twenty-sided die. Player A's score is the total of the values on their three dice, while Players B and C score based on the face value of their respective dice. The question is whether this game is fair.

*Solution 2*: The problem is not very challenging by itself, given sufficient time. However, it may be challenging to calculate every detail correctly under time pressure during an interview. Let $A$, $B$, and $C$ denote the score of players A, B, and C, respectively. Let us compute the winning probability of A first.

Since $A = X_1 + X_2 + X_3$ and $X_i \sim \mathrm{Unif}([6])$, we have $\mathrm{Var}[X_i] = (6^2 - 1)/12 = 35/12$ and $\mathrm{Var}[A] = 35/12 \cdot 3 = 35/4$, $\mathbb{E}[A] = (1 + 6)/2 \cdot 3 = 21/2$, $\mathbb{E}[A^2] = \mathrm{Var}[A] + \mathbb{E}[A]^2 = 35/4 + (21/2)^2 = 119$. Therefore, we have

$$
\Pr(A > \max\{B,C\}) = \Pr(A > B \wedge A > C) = \mathbb{E}[\Pr(B < A \mid A)\Pr(C < A \mid A)] = \mathbb{E}[(\frac{A-1}{20})^2] = \frac{1}{400}\mathbb{E}[A^2-2A+1] = (119-21+1)/400 = 99/400.
$$

It is obvious that the winning probability is the same for players B and C. Let us compute the winning probability of player B.

We have

$$
\mathbb{E}[\mathbb{E}[\max\{A,C\} \mid A]] = \mathbb{E}[\frac{A}{20}\cdot A + (1 - \frac{A}{20})\frac{20+A+1}{2}] = \mathbb{E}[\frac{A^2-A+420}{40}] = \frac{119-21/2+420}{40} = 1057/80.
$$

Thus, 

$$
\Pr(B > \max\{A,C\}) = \mathbb{E}[\frac{20-\max\{A,C\}}{20}] = 1 - \frac{1}{20}\mathbb{E}[\max\{A,C\}\mathbb{E}[\mid A]] = 1 - 1057/1600 = 543/1600.
$$

So the game is not fair in the sense of different winning probabilities. The problem was also discussed [here](https://web.archive.org/web/20230401010602/https://math.stackexchange.com/questions/2610668/is-this-a-fair-game-three-players-throw-dice).
