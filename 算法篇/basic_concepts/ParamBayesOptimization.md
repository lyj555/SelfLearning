[TOC]

# Bayes Optimization

## 1. 介绍

本章节**贝叶斯优化**用于机器学习模型调参使用，由J.Snoek(2012)提出，主要思想是给定优化的目标函数（只需要指定输入和输出即可，无需知道内部结构以及数学性质），通过不断添加样本点来更新目标函数的后验分布(posterior distribution)，该过程相当于是**高斯过程**（通俗点说就是每次使用参数均均考虑之前参数的相关信息，从而更好的调整当前的参数）。

与常规的网格搜索或者随机搜索的区别是：

- 贝叶斯调参采用高斯过程，考虑之前的信息，不断的更新先验；网格搜索活随机搜索未考虑之前的信息
- 贝叶斯调参迭代次数相对较少，速度快；
- 贝叶斯调参相较于网格搜索，对非凸问题依然稳健，而网格搜索容易得到局部最优解。

下面主要主要分两部分介绍，分别是高斯过程和贝叶斯调参过程。

- 高斯过程，用于拟合优化目标函数
- 贝叶斯优化，包括"开采"和"勘探"，用以花最少的代价找到最优值

## 2. 高斯过程

高斯过程是一系列随机变量的集合，有限个联合高斯分布， 高斯过程可以用于非线性回归、非线性分类和参数的寻优等。

### 2.1 多元高斯分布

- 高斯分布

  定义$k$维随机变量$\boldsymbol{x}=[X_1, X_2, \ldots, X_k]$,其高斯分布为，则其高斯分布为
  $$
  \boldsymbol{x} \sim N_k(\boldsymbol{\mu}, \Sigma)
  $$
  其中，$\boldsymbol{\mu}=[E(X_1), E(X_2), \ldots, E(X_k)]$, $\Sigma=[Cov(X_i, X_j)], i=1,2,\ldots, k; j=1,2,\ldots,k$



- 高斯分布概率密度函数

$$
f(x_1, x_2, \ldots, x_k)=\frac{1}{\sqrt{(2\pi)^k|\Sigma|}}\rm{exp}(-\frac{1}{2}(\boldsymbol{x}-\mu)^T\Sigma^{-1}(\boldsymbol{x}-\mu))
$$

- 条件分布

  令$\boldsymbol{f} \sim N(\boldsymbol{\mu}, \Sigma)$，

  $\boldsymbol{f}=[\boldsymbol{f_1}, \boldsymbol{f_2}]^T, \boldsymbol{\mu}=[\boldsymbol{\mu_1}, \boldsymbol{\mu_2}]^T, \Sigma=\begin{bmatrix} \Sigma_{11}&\Sigma_{12}\\ \Sigma_{21}&\Sigma_{22} \end{bmatrix}$

  $\boldsymbol{f_1} \sim N(\boldsymbol{\mu_1}, \Sigma_{11}), \boldsymbol{f_2} \sim N(\boldsymbol{\mu_2}, \Sigma_{22})$

  $\boldsymbol{f_2}|\boldsymbol{f_1} = N(\boldsymbol{\mu_2}+\Sigma_{21}\Sigma_{11}^{-1}(\boldsymbol{f_1}-\boldsymbol{\mu_1}), \Sigma_{22}-\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12})$

  > - 整体服从多维高斯分布，那么其中一部分也服从高斯分布，均值和协方差可唯一确定
  > - 整体服从多维高斯分布，那么条件分布也服从高斯分布，均值和协方差可以唯一确定

### 2.2 高斯过程和高斯分布

高斯分布基于**向量**层面，高斯过程是基于**函数**层面。

高斯过程完全是由它的均值函数$m( . )$和协方差函数$k(., .)$所决定,
$$
f(\boldsymbol{x}) \sim GP(m(\boldsymbol{x}), k(\boldsymbol{x}, \boldsymbol{x}^{\prime}))
$$
其中变量$\boldsymbol{x}和\boldsymbol{x}^{\prime}$均服从高斯分布。

### 2.3 Gaussian process regression

假设训练数据为$X=[\boldsymbol{x_1}, \boldsymbol{x_2, \ldots, \boldsymbol{x_n}}]^T, \boldsymbol{y}=[y_1, y_2, \ldots, y_n ]$

目标是对一个新的数据$\boldsymbol{x}^*$，来预测对应的值$y^{*}$.

往往是需要对$p(\boldsymbol{y}|X)$进行建模，预测时，只需求$p(y^*|\boldsymbol{x}^*)$即可。

在高斯过程中，额外考虑了$\boldsymbol{y}$和$y^*$的关系，即求$p(y^*|\boldsymbol{x}^*,\boldsymbol{y})$.

- 先验分布

  高斯过程假设$\boldsymbol{y}$值服从联合正态分布，即服从零均值的多元高斯分布
  $$
  \boldsymbol{y} \sim N(\boldsymbol{0}, \boldsymbol{K})
  $$
  其中$\boldsymbol{K}=\begin{bmatrix} &k(x_1, x_1) &k(x_1, x_2) &\cdots &k(x_1, x_n) \\ &k(x_2, x_1) &k(x_2, x_1) &\cdots &k(x_2, x_n) \\ &\vdots &\vdots &\ddots &\vdots \\ &k(x_n, x_1) &k(x_n,x_2) &\cdots &k(x_n,x_n) \end{bmatrix}$, $k$为核函数

  而根据训练数据求得最优的核矩阵$\boldsymbol{K}$,为后验估计做准备

- 后验分布

  同样$[\boldsymbol{y}, y^*]^T$服从高斯分布，
  $$
  \begin{bmatrix} \boldsymbol{y} \\ y^*\end{bmatrix} \sim N(\boldsymbol{0}, \begin{bmatrix}\boldsymbol{K} & \boldsymbol{K}_*^T \\ \boldsymbol{K}_* &\boldsymbol{K}_{**} \end{bmatrix})
  $$
  其中$\boldsymbol{K}_*=[k(x_*, x_1)\ \ k(x_*, x_2) \ \cdots \ k(x^*,x_n)], \boldsymbol{K}_{**}=k(x_*,x_*)$

  则对应的后验分布
  $$
  y_*|\boldsymbol{y}\sim N(\boldsymbol{0}+\boldsymbol{K}_*\boldsymbol{K}^{-1}\boldsymbol{y}, \boldsymbol{K}_{**}-\boldsymbol{K}_*\boldsymbol{K}^{-1}\boldsymbol{K}_*^T)
  $$
  可以看出$y^*$的均值是$\boldsymbol{y}$的一个加权平均。

  > 后验分布中，利用了训练样本中的信息，因为假设训练和测试样本遵循一个高斯过程，
  >
  > 所以如果训练样本中有$\boldsymbol{x}$对于测试样本非常接近，那么测试样本的预测值应该和$\boldsymbol{x}$对应的值接近。

- 模型训练

  模型训练的过程就是求核函数$k$中的参数。关于函数的求解，有两种方式，

  - Maximum Likelihood(ML)

  - Monte Carlo

    暴力破解，强力计算后验分布

## 3. 贝叶斯优化

贝叶斯优化是一种逼近思想，当计算非常复杂、迭代次数较高时能起到很好的效果，多用于超参数确定

### 3.1 基本思想

基于贝叶斯定理来估计目标函数的后验分布，然后根据分布选择下一个采样的超参数组合，它充分利用之前采样点的信息，其优化的工作方式是通过对目标函数形状的学习，来找到是全局效果提升最大的参数。

在上面介绍的高斯过程中，我们可以看出，它的主要工作是对目标函数进行建模，得到其的后验分布。

通过高斯分布得到后验分布后，可以根据分布进行抽样计算，而贝叶斯优化很容易在局部最优解上不断采样，这就设计到了**开发**和**探索**之间的权衡。

- 开发（exploitation）

  根据后验分布，在最可能出现全局最优解的区域进行采样，相当于均值

- 探索（exploration）

  在还未取样的区域获取采样点，相当于方差

如何进行高效的采样，即开发和探索，或者说在某组参数下，使得分布的均值和方差之和最大，均值越大表示模型效果越好，而方差越大则表示该区域拟合的分布不是很准确，需要额外的取样来校正。

贝叶斯优化中，定义了一个**Acquisition Function**的函数，通过此函数来确定下一组参数。

### 3.2 Acquisition Function

Acquisition Function(Utility Function)是关于$x$(超参数)的函数，映射到实数空间，表示该点的功效，或者说我们想选取一个候选点，使得Acquisition Function最大。其主要有以下几种形式，

- POI(probability of improvement)
  $$
  POI(X)=P(f(X) \geq f(X^+)+\xi)=\Phi(\frac{\mu(x)-f(X^+)-\xi}{\sigma(x)})
  $$
  其中，$f(X)$为$X$的目标函数值，$f(X^+)$为目前为止最优的目标函数值，$\mu(x)$和$\sigma(x)$代表高斯过程学得的均值和方差（$f(X)$的后验分布），$\xi$可以看做为trade-off系数，如果没有该系数，则POI函数倾向选择$X^+$周围的点，即倾向exploit而不是explore，因此需要该参数来权衡。

- EI(Expected Improvement)

  POI是一个概率函数，因此只考虑了$f(X)$比$f(X^+)$大的概率，而EI则考虑了$f(X)$比$f(X^+)$大多少，

  首先获取$X$,
  $$
  X = \arg \max_xE(max(0, f_{t+1}(x)-f(X^+))|D_t)
  $$
  其中$D_t$表示为前$t$个样本，在正态分布的假设下，有
  $$
  EI(X)=\begin{cases} \mu(x)-f(X^+)\Phi(\frac{\mu(x)-f(X^+)}{\sigma(x)})+\sigma(x)\Phi(\frac{\mu(x)-f(X^+)}{\sigma(x)}),\quad if\ \sigma(x)>0 \\ 0,\quad if\ \sigma(x)=0 \end{cases}
  $$

- Confidence bound criteria

  $LCB(x)=\mu(x)-k\sigma(x)$

  $UCB(x)=\mu(x)+k\sigma(x)$
  
  

## References

- [如何通俗易懂地介绍 Gaussian Process？](https://www.zhihu.com/question/46631426)
- [强大而精致的机器学习调参方法：贝叶斯优化](https://www.cnblogs.com/yangruiGB2312/p/9374377.html)

- []()