---
title: 'How Long Will It Take for Three Frogs to Meet on a Triangle?'
date: 2023-03-26
permalink: /posts/2023/03/three_frogs_meet/
comments: true
tags:
  - probability
---

**Problem:** Three frogs jump between the vertices of an equilateral triangle. A vertex can hold more than one frog, and the frogs jump randomly to one of the other two vertices each minute. The frogs are independent and each vertex starts with one frog. How long does it take on average for all the frogs to meet at the same vertex?

**Solution:** The problem at hand involves a Markov chain with three states, namely $S_1$, $S_2$, and $S_3$. We define $S_1$ to be the state where all three frogs are on distinct vertices, which is the initial state. $S_2$ is the state where two frogs are on one vertex and the third frog is on another vertex. $S_3$ is the terminal state where all three frogs are on the same vertex. In this problem, the goal is to quickly determine the transition probabilities during an interview under time pressure.

We use $(n_1,n_2,n_3)$ to represent the number of frogs on vertices $A$, $B$, and $C$, respectively. For instance, $S_1$ can be represented by $(1,1,1)$, while $S_3$ can be represented by $(3,0,0)$, $(0,3,0)$, or $(0,0,3)$.

When in state $S_1=(1,1,1)$, we can only transition to $S_1$ or $S_2$. To find $\Pr(S_1\to S_1)$, we consider where the frog on vertex $A$ jumps. Without loss of generality, let us assume that the frog on $A$ jumps to $B$. To return to $S_1$, the frog on $B$ must jump to $C$, and the frog on $C$ must jump to $A$. Therefore, $\Pr(S_1\to S_1)=1/4$, and thus $\Pr(S_1\to S_2)=3/4$.

Next, we find $\Pr(S_2\to S_3)$ and $\Pr(S_2\to S_1)$. Let's suppose the current state is $(2,1,0)$. To reach $S_3$, all frogs must jump to vertex $C$, which happens with probability $1/8$. Therefore, $\Pr(S_2\to S_3)=1/8$. To reach $S_1$, the frog on $B$ must jump to $A$, and the two remaining frogs must both jump to $B$ or both jump to $C$. Thus, $\Pr(S_2\to S_1)=1/2\cdot (1/2\cdot 1/2\cdot 2)=1/4$. Consequently, $\Pr(S_2\to S_2)=1-1/4-1/8=5/8$.

Let $x_1$ and $x_2$ denote the expected number of minutes required to reach $S_3$ when starting at $S_1$ and $S_2$, respectively. We can express these quantities as follows:

$$
\begin{align}
x_1={}& 1+1/4\cdot x_1 + 3/4\cdot x_2  \\
x_2={}& 1 + 1/4 \cdot x_1 + 5/8\cdot x_2 + 1/8\cdot 0
\end{align}
$$
  
Solving the above system of equations yields $x_1=12$. To verify this result, we can use the following Python code:

```python
import random

def simulate():
    # Initialize the vertices with one frog each
    vertices = [0, 1, 2]
    time = 0
    
    # Continue looping until all the frogs are on the same vertex
    while not (vertices[0] == vertices[1] == vertices[2]):
        # Increment the time
        time += 1
        
        # Move each frog to a random vertex (excluding the current vertex)
        for i in range(3):
            choices = [j for j in range(3) if j != vertices[i]]
            vertex = random.choice(choices)
            vertices[i] = vertex
    
    # Return the time it took for all the frogs to meet at the same vertex
    return time

# Run the simulation 10,000 times and calculate the average time
num_simulations = 10000
total_time = sum(simulate() for _ in range(num_simulations))
average_time = total_time / num_simulations

print(f"On average, it takes {average_time} minutes for all the frogs to meet at the same vertex.")
 ```