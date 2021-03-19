$\def\abs#1{\vert #1 \vert}$

1. 若$v$, $w$是两个向量，$G$是一个随机矩阵（一般是Stieltjes变换或者Stieltjes变换的某些函数，比如导数等），则可以通过local law of random matrices将这个矩阵换成确定的。具体见此[幻灯片](http://www.mathphys.org/Venice17/slides/knowles.pdf)。这条local law具体在[此文](https://arxiv.org/pdf/1110.6449.pdf)中证明。
2. [此文](https://arxiv.org/pdf/2101.09612.pdf)改进了ReLU神经网络的收敛结果。
3. 不等式：如果$\Sigma\in \mathbb{R}^{d\times d}$，$\Vert\Sigma^{-1}\Vert\le \frac{\Vert\Sigma\Vert^{d-1}}{\|\det(\Sigma)\|}$。 [相关推特](https://twitter.com/miniapeur/status/1356026324733874181?s=20)。
4. Hoffman-Wielandt不等式：如果$A,B$是两个Hermitian矩阵，则$\sum_{i=1}^n (\lambda_i(A)-\lambda_i(B))^2\le \operatorname{Tr}(A-B)^2 $。注意：$A,B$一般不满足交换律，$\lambda_i(A)-\lambda_i(B)$一般而言不是$A-B$的特征值。这个不等式告诉我们$\lambda_i(A)$这个函数是Lipschitz的，因此也是连续的。若要证明连续性，还有一个办法是注意到特征值是多项式$\det(zI-A)$的根。可以看[这篇文章](https://arxiv.org/pdf/1710.10792.pdf)的Section 3。
5. Polish空间是可分的完备可度量化的拓扑空间。可分的意思是即存在可数的稠密子集。
6. 介绍Fixed-Parameter Tractability的[简明材料]([ac-tr-21-004.pdf (tuwien.ac.at)](https://www.ac.tuwien.ac.at/files/tr/ac-tr-21-004.pdf))，见其中的第17.2.1节。
7. VC维数、PAC学习的[演示文稿]([lec23_24_handout.pdf (toronto.edu)](https://www.cs.toronto.edu/~jlucas/teaching/csc411/lectures/lec23_24_handout.pdf))。
8. 对于超图，有$\alpha$-无环，依次包含$\beta$-无环，$\gamma$-无环，和Berge-无环。
9. 很简明清楚的关于核化（Kernelization）的[材料](https://simons.berkeley.edu/sites/default/files/docs/4006/lossy-kernel.pdf)。Daniel Marx的[演示文稿](http://cs.bme.hu/~dmarx/papers/marx-warsaw-fpt1)更全面。
10. Fourier–Motzkin elimination是一种在线形不等式组中消元的算法。

