---
title: 'Expected Number of Moves for an Ant to Traverse a Triangle and Tetrahedron'
date: 2023-03-14
permalink: /posts/2023/03/traverse_triangle/
comments: true
tags:
  - probability
---

**Problem**: Consider a triangle $ABC$, where an ant starts at point $A$. At each moment, the ant moves with equal probability to either of the other two points. What is the expected number of moves until the ant has visited all three edges for the first time? Additionally, what is the expected number of moves for a tetrahedron $ABCD$ in which the ant starts at vertex $A$ and moves with equal probability to any of the other three vertices at each moment, until it has visited all six edges for the first time?

**Solution 1**: ![Illustration of the five states](/images/5triangles.svg)

After excluding symmetry, this problem can be represented by five states, as shown in the figure. Dashed lines represent unvisited edges, solid lines represent visited edges, and slightly larger dots indicate the current location of the ant. Below, we describe each of these five states:

1. Initial state: No edges have been visited. The ant starts at vertex $A$.

2. The ant has visited one edge and stopped at one of its vertices.

3. The ant has visited two edges and stopped at a vertex of the unvisited edge.

4. The ant has visited two edges and stopped at the common vertex of the two visited edges. Note that state 3 must occur before state 4.

5. All three edges have been visited. This is the termination state.

We define $f(i)$ as the expected time it takes to reach state 5 starting from state $i$. We set $f(5)=0$. Note that from state 4, the ant must move to state 3 next, so we have $f(4)=1+f(3)$. When starting from state 3, there are two possibilities. In our example, if the ant visits the unvisited edge (in our example, the ant goes from point $C$ to point $A$), we enter state 5 with a probability of $1/2$. Otherwise, with a probability of $1/2$, we move from state 3 to state 4. Thus, we get $f(3)=1+1/2\cdot (f(5)+f(4))$. Similarly, state 2 has a $1/2$ probability of moving to state 3 and a 1/2 probability of returning to state 2, so $f(2)=1+1/2\cdot (f(2)+f(3))$. State 1 must move to state 2 next, so $f(1)=1+f(2)$.

To summarize the equations: $f(5)=0$, $f(4)=1+f(3)$, $f(3)=1+1/2\cdot (f(5)+f(4))$, $f(2)=1+1/2\cdot (f(2)+f(3))$, $f(1)=1+f(2)$. Solving these equations gives us $f(1)=6$, so the answer is 6.

**Solution 2**

We can also apply the geometric distribution to analyze this problem. After starting from state 1, the next state must be state 2, which takes 1 unit of time. Then, the time it takes to transition from state 2 to state 3 follows a geometric distribution with parameter $p=1/2$, with an expected value of 2.

From state 3, there are two possible outcomes. With a probability of $1/2$, the ant returns to state 3, which takes 2 units of time since it has to go through state 4. Alternatively, with a probability of 1/2, the ant proceeds to state 5. Therefore, the time it takes starting from state 3 is given by $2(\mathrm{Geom}(1/2)-1)+1$, which has an expected value of 3.

Thus, the expected total time to traverse all edges is the sum of the expected times to transition from state 1 to state 2, from state 2 to state 3, and from state 3 to state 5, which is $1+2+3=6$. This result is consistent with the one obtained above.

Here is a [link](https://gist.github.com/lchen91/7cb8adad9ede50aa41150bb73a7c31fd) to the code for the simulation.