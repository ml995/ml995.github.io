---
title: 'Expected Value of Balls Remaining in Second Jar in Random Selection Process'
date: 2023-04-15
permalink: /posts/2023/04/expected_value_of_balls_in_second_jar/
comments: true
tags:
  - probability
---

**Problem**: Suppose we have two jars, each containing $n$ balls. We randomly select one of the jars with equal probability and draw a ball from it. We repeat this process until one of the jars becomes empty. Find the expected value of the number of balls remaining in the other jar.

**Solution**: The problem at hand is related to Banach's matchbox problem, but with a slight difference. In Banach's matchbox problem, the process stops when one of the jars is found to be empty. To find the expected value of the number of balls remaining in the other jar, we only need to find the expected value of $T$, where $T$ is the time taken to empty one jar.

Suppose that when we empty the first jar, the other jar has $k$ balls remaining ($k=0,1,2,\dots,n-1$). This event happens with probability

$$
\Pr(K=k) = \binom{n+k-1}{n-1}\left(\frac{1}{2}\right)^{n+k},
$$

since the process stops when either jar becomes empty, $\Pr(T=k)$ is twice the above probability, i.e.,

$$
\Pr(T=k) = 2\Pr(K=k) = \binom{n+k-1}{n-1}\left(\frac{1}{2}\right)^{n+k-1}.
$$

Thus, the expected value of $T$ is

$$
\begin{aligned}
\mathbb{E}[T] &= \sum_{k=0}^{n-1} k \Pr(T=k) \\
&= \sum_{k=0}^{n-1} k \binom{n+k-1}{n-1}\left(\frac{1}{2}\right)^{n+k-1}.
\end{aligned}
$$

Therefore, the quantity of interest is

$$
\begin{aligned}
n-\mathbb{E}[T] &= n - \sum_{k=0}^{n-1} k \binom{n+k-1}{n-1}\left(\frac{1}{2}\right)^{n+k-1} \\
&= \frac{(2n-1)!}{4^{n-1}(n-1)!^2}.
\end{aligned}
$$

We verify the result using the Python simulation below.

```python
import random

def subtract_until_zero(n):
    x1 = n
    x2 = n
    while x1 > 0 and x2 > 0:
        if random.random() < 0.5:
            x1 -= 1
        else:
            x2 -= 1
    return x1 + x2

num_simulations = 10000

for n in range(1, 11):
    total_result = 0
    for i in range(num_simulations):
        result = subtract_until_zero(n)
        total_result += result
    avg_result = total_result / num_simulations
    print(f"n = {n}, avg result = {avg_result}")

"""
    n = 1, avg result = 1.0
    n = 2, avg result = 1.4974
    n = 3, avg result = 1.8796
    n = 4, avg result = 2.1881
    n = 5, avg result = 2.4436
    n = 6, avg result = 2.7139
    n = 7, avg result = 2.9298
    n = 8, avg result = 3.1538
    n = 9, avg result = 3.3549
    n = 10, avg result = 3.5019
"""


import math

for n in range(1, 11):
    numerator = math.factorial(2*n - 1)
    denominator = 4**(n-1) * (math.factorial(n-1))**2
    result = numerator / denominator
    print(f"n={n}: {result}")


"""
    n=1: 1.0
    n=2: 1.5
    n=3: 1.875
    n=4: 2.1875
    n=5: 2.4609375
    n=6: 2.70703125
    n=7: 2.9326171875
    n=8: 3.14208984375
    n=9: 3.338470458984375
    n=10: 3.5239410400390625
"""


from scipy.special import binom

for n in range(1, 11):
    result = 0
    for k in range(n):
        numerator = k * binom(n + k - 1, n - 1)
        denominator = 2**(n + k - 1)
        term = numerator / denominator
        result += term
    print(f"n={n}: {n - result}")

"""
    n=1: 1.0
    n=2: 1.5
    n=3: 1.875
    n=4: 2.1875
    n=5: 2.4609375
    n=6: 2.70703125
    n=7: 2.9326171875
    n=8: 3.14208984375
    n=9: 3.338470458984375
    n=10: 3.5239410400390625
"""
```