---
title: 'Finding the Optimal Fixed Point Location for a Thin Wooden Stick to Swing from Horizontal to Vertical in Minimum Time'
date: 2023-03-20
permalink: /posts/2023/03/optimal_fixed_point_for_wooden_stick_swing/
comments: true
tags:
  - physics
---

**Problem**: Consider a thin and uniform wooden stick of length $L$, fixed at one point such that it can rotate without any friction around that point. If the stick is released from a horizontal position while at rest, it will swing downwards due to the force of gravity. What is the distance between the fixed point and the midpoint of the stick that would allow the stick to reach the vertical position in the shortest time possible?

**Solution**: This is supposedly a physics question asked in an interview by a high-frequency trading firm. Assuming that the distance between the fixed point and the midpoint of the stick is $x$, and the angle of rotation from the horizontal is $\theta$, the linear mass density of the stick is $\rho$, and the angular velocity is $\omega$. The moment of inertia of the stick is given by

$$
\frac{1}{3}\rho (L/2-x)^3+\frac{1}{3}\rho(L/2+x)^3
$$

When the stick rotates through an angle of $\theta$, its kinetic energy (on the left-hand side of the equation below) should be equal to the gravitational potential energy lost (on the right-hand side of the equation below).

$$
\frac{1}{2}\left(\frac{1}{3}\rho (L/2-x)^3+\frac{1}{3}\rho(L/2+x)^3\right)\omega^2 = \rho L gx\sin \theta
$$

The kinetic energy on the left-hand side can also be obtained by integration without using the moment of inertia, although the moment of inertia is ultimately obtained by integration. To maximize the angular velocity, the quantity below should be minimized:

$$
\frac{(L/2-x)^3+(L/2+x)^3}{x}
$$

After simplification, the above quantity is equal to $\frac{L^3}{4x}+3Lx$. Its minimum value is obtained at $x=L/\sqrt{12}$.