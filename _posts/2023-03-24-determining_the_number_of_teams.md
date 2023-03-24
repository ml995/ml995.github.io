---
title: 'Determining the Number of Teams'
date: 2023-03-24
permalink: /posts/2023/03/determining_the_number_of_teams/
comments: true
tags:
  - brain-teaser
---

**Problem:** Suppose there are $N$ teams that play against each other once. A team earns 2 points for a win, 0 points for a loss, and 1 point for a tie. Given that each team earns half of their total points from matches against the bottom 10 teams, what is the value of $N$?

**Solution:** We begin by considering the bottom 10 teams. If we only consider the scores from matches between these bottom 10 teams (which we will call "internal" matches), the total score for these matches is $\binom{10}{2}\times 2 = 90$. Note that half of their total score comes from matches against the bottom 10 teams, which means that these 90 points represent half of their total score. Therefore, their total score (including internal and external matches) is 180.

There are a total of $N$ teams, and the total score for all teams is $\binom{N}{2}\times 2 = N(N-1)$. The total score for teams that are not among the bottom 10 is $N(N-1)-180$, and the score they earn from matches against the bottom 10 teams is $(N(N-1)-180)/2$.

We can also compute the score earned by teams that are not among the bottom 10 from matches against the bottom 10 teams in a different way. First, the total score for these matches is $10(N-10)\cdot 2$. As we computed earlier, the total score for internal matches among the bottom 10 teams is 90, which means that the external matches between the bottom 10 teams and the other teams contribute 90 points to the bottom 10 teams' total score. Therefore, the score earned by teams that are not among the bottom 10 from matches against the bottom 10 teams is $10(N-10)\cdot 2-90$.

Equating the two expressions for the score earned by teams that are not among the bottom 10 from matches against the bottom 10 teams gives:

$$
(N(N-1)-180)/2 = 10(N-10)\cdot 2-90.
$$

Solving for N yields $N=16,25$.