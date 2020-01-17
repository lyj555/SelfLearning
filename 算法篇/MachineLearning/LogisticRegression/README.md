[TOC]

# 逻辑回归模型

## 1. 模型算法

### 1.1 模型简介
逻辑回归（Logistic Regression）是机器学习中的一种分类模型，由于算法的简单和高效，在实际中应用非常广泛。本文逻辑回归模型假设`label`服从伯努利分布，通过极大化似然函数的方法，运用梯度下降法进行参数的求解，从而解决二分类问题。

### 1.2 模型假设

- `label`服从伯努利分布
假设正样本的概率为$p$，负样本的概率则为$1-p$，那么整个逻辑回归模型可以描述为如下，
$$
h_\theta(\theta;x)=p
$$
- 概率假设
逻辑回归模型假设正样本的概率$p$为如下，
$$
p=\frac{1}{1+e^{-\theta^Tx}}
$$
相当于对于特征做了线性加权，之后对于加权值做了一个非线性变换(**sigmoid**函数)

> 上面函数将**截距项**默认添加到$x$中

## 2.  模型求解
### 2.1 损失函数
假设我们的训练样本为$(x_i, y_i)|_{i=1}^{m}$，根据上面的概率假设，我们可以得到样本的最大似然函数$L$，
$$
L=\prod_{i=1}^m p_i^{y_i}.(1-p_i)^{1-y_i}
$$
其对数似然函数为
$$
\ln L=\sum_{i=1}^m y_i\ln{p_i}+(1-y_i)\ln{(1-p_i)}
$$
在机器学习领域，经常遇到的是损失函数的概念，其衡量的是模型预测错误的程度。

最大化似然函数对应最小化负的似然函数，那么对似然函数取负数即可得到逻辑回归模型的损失函数，如下，
$$
\rm{Loss}(y, p)=- \sum_{i=1}^m y_i\ln{p_i}+(1-y_i)\ln{(1-p_i)}
$$
其中上式也称之为**交叉熵损失函数**,由熵的定义($H(p)=-\sum_{i=1}^m p_i\log{p_i}$)演化得到。

> 熵：在信息论与统计中，熵是表示随机变量的不确定性的度量，值越小，不确定越小


### 2.2 损失函数求解
模型求解的过程即通过最小化损失函数，从而求得参数$\theta$的过程，即
$$
\theta^*=\underset{\theta}{\arg \min}\ \rm{Loss}
$$
通常会在损失函数项中加入正则项($L_1$, $L_2$)，具体形式分别如下，
$$
\begin{cases}
\rm{Loss}_{l1} = \underset{\theta,b}{\min}||\theta||_1+C \cdot \rm{Loss}\\
\rm{Loss}_{l2} = \underset{\theta,b}{\min}\frac{1}{2}\theta^T\theta+C \cdot \rm{Loss}
\end{cases}
$$

其中$C$表示正则项的控制力度，$C$越小表示强的正则强度

此时问题表示为一个优化问题，存在多种求解方法（逻辑回归的损失函数是凸函数，加入正则项后是严格凸函数），求解方法如下表（参考[sklearn中逻辑回归算法](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)的求解方式），

| case         | solver                                |
| :----------- | :------------------------------------ |
| 不添加正则项 | `lbfgs` `newton-cg` `sag` `saga`      |
| $L_1$正则项  | `CD`or`saga`                          |
| $L_2$正则项  | `CD` `lbfgs` `newton-cg` `sag` `saga` |
| 大数据集     | `sag` `saga`                          |


### 2.3 数据实例  

下面以**梯度下降**解法为例，进行求解，假设数据为，
$$
\begin{array}{ccc|c}
x_1 & {x_2} & {x_3} & {label} \\
\hline
8 & 5& 1 & 0\\
1 & 4 & 3& 1 \\
3 & 2& 2& 0 \\
3 & 3& 0& 0 \\
3 & 10 & 1& 1 \\
3 & 9 & 8& 0 
\end{array}
$$
则损失函数可定义为
$$
Loss(\theta_j)=-\sum_{i=1}^{6}y_i\ln p_i+(1-y_i)\ln (1-p_i)
$$
其中$p_i=\frac{1}{1+e^{-\theta^Tx}}=\frac{1}{1+e^{\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_3}}$
然后通过梯度下降对损失函数进行优化。

损失函数$\rm{Loss}(\theta_j)$关于$\theta_j$ 的偏导数为如下，
$$
\begin{align*}
\frac{\partial\ \rm{Loss}(\theta_j)}{\partial\ \theta_j} & = -\sum_{i=1}^6y_i\cdot\frac{1}{p_i}\cdot \frac{\partial\ p_i}{\partial\ \theta_j}+(1-y_i)\cdot\frac{1}{1-p_i}\cdot (-1) \cdot \frac{\partial\ p_i}{\partial\ \theta_j} \\
 & = -\sum_{i=1}^{6} y_i\cdot \frac{1}{p_i}\cdot p_i\cdot(1-p_i)\cdot x_j^{(i)} + (1-y_i)\cdot\frac{1}{1-p_i}\cdot (-1) \cdot p_i\cdot(1-p_i)\cdot x_j^{(i)}\\ 
 & =   -\sum_{i=1}^{6} y_i\cdot(1-p_i)\cdot x_j^{(i)} + (1-y_i)\cdot (-1) \cdot p_i\cdot x_j^{(i)}\\
 & = -\sum_{i=1}^{6}(p_i-y_i)\cdot x_j^{(i)} \\ 
\end{align*}
$$
其中$\frac{\partial\ p_i}{\partial\ \theta_j}=p_i\cdot(1-p_i)\cdot x_j^{(i)}$.

求解的过程如下，
1. 随机初始化参数${\theta}$, $\theta_0=1,\theta_1=1,\theta_2=1, \theta_3=1$

2. 依次更新$\theta_0,\theta_1,\theta_2,\theta_3$，更新方式如下，
$$\theta_j := \theta_j-\alpha \cdot \left. \frac{\partial\ \rm{Loss}(\theta_j)}{\partial\ \theta_j} \right|_{\theta_j},j=0,1,2,3$$
其中$\alpha$为预先设定的学习率

   > 更新$\theta_0$时，将$\theta_1, \theta_2, \theta_3$的值带入式中（相当于已知），然后依次更新$\theta_1, \theta_2, \theta_3$,直到所有参数更新完成后，此时一轮的学习完成


3. 不断重复上述步骤2，直到参数值收敛或者达到预定训练次数。  

那么最终的模型可以表示为,
$$
h(\theta,x) = \frac{1}{1+e^{\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_3}}
$$


## 3. 逻辑回归模型疑问

- 损失函数为什么使用最大似然估计而不是最小二乘法？    

  - 最优化的角度

    如果选用最小二乘法，那么损失函数可以写为
    $$\rm Loss=\sum_{i=1}^m(p_i-y_i)^2$$
    那么$\rm Loss$是关于参数$\theta$的非凸函数，难以求得最优解。
  - 模型假设
    逻辑回归模型假设`label`服从伯努利分布，往往数据`label`服从高斯分布（正态分布）时使用平方损失。
- 逻辑回归模型在训练时，如果有很多特征高度相关，那么会有什么影响？    
在最终收敛的情况下，即使存在多个线性相关的特征，最终的模型精度一样。整体模型训练的速度会变慢。
- 如果有特征存在量纲不一致的情况，比如某个特征取值范围在0-1之间，另外一个特征取值在1-100000之间，最终有什么影响？  
在保证训练次数的前提下，最终的模型会收敛，模型精度一致。不过模型整体训练的速度会变慢。

## 4. 延伸

### 4.1 逻辑回归和高斯贝叶斯模型

逻辑回归是一种判别模型，表为对条件概率$p(y|x)$建模，而不关心背后的数据分布$p(x, y)$;而高斯贝叶斯模型（Gaussian Naive Bayes）是一种生成模型，先对数据的联合分布建模，再通过贝叶斯公式来计算样本属于各个类别的后验概率，即，
$$
p(y|x)=\frac{p(x|y)p(y)}{\sum{p(x|y)p(y)}}
$$
通常假设$p(x|y)$是高斯分布，$p(y)$是多项式分布，相应的参数可以通过最大似然求得。

如果考虑二分类问题，通过简单的变换可以得到：
$$
\begin{aligned}
\log \frac{p(y=1|x)}{p(y=0|x)}
&=\log \frac{p(x|y=1) \cdot p(y=1)}{p(x|y=0) \cdot p(y=0)}\\
&=\log \frac{p(x|y=1)}{p(x|y=0)} + \log \frac{p(y=1)}{p(y=0)}\\
&=-\frac{(x-\mu_1)^2}{2\sigma_1^2}+\frac{(x-\mu_0)^2}{2\sigma_0^2}+\theta_0
\end{aligned}
$$

如果$\sigma_1=\sigma_2$,二次项会抵消，我们得到一个简单的线性关系：
$$
\log \frac{p(y=1|x)}{p(y=0|x)}=\theta^Tx
$$
对上面的式子进一步变换，即得到我们上面讨论的逻辑回归模型，
$$
p(y=1|x)=\frac{e^{\theta^{T}x}}{1+e^{\theta^{T}x}}=\frac{1}{1+e^{-\theta^{T}x}}
$$
这种情况下，GNB和LR会学习到同一个模型，**实际上，在更一般的假设（$p(x|y)$的分布属于指数分布簇）下，都可以得到类似的结论**。

> 指数族分布，对于随机变量$x$，在给定参数$\eta$下，其概率分布满足如下形式$p(x|\eta)=h(x)g(\eta)exp\{\eta^Tu(x)\}$，其中$x$可以是向量或者标量，可以使离散值后者连续值，$\eta$是自然参数，$g(\eta)$是归一化系数，$h(x)$、$u(x)$是$x$的某个函数。
>
> 常见的几个指数分布族，正态分布，伯努力分布，二项分布，泊松分布，伽马分布

### 4.2 多分类

如果标签$y$中取值不是0和1,而是$K(K>2)$个类别，这时问题就变为一个多份类问题。往往有两种方式处理该问题，

- 对每个类别训练一个二元分类器（one-vs-all）

  当$K$个类别不是互斥的时候，比如用户会购买哪种品类，比较合适

- softmax

  当$K$个类别是互斥的时候，即$y=i$的时候不能取其他的值，比如预测用户的年龄段，这种情况下，softmax更加合适。

softmax回归是对逻辑回归在多分类的推广，相应的模型也可以称之为**多元逻辑回归**（Multinomial Logistic Regression），模型通过softmax函数来对概率进行建模，具体形式如下，
$$
p(y=i|x,\theta)=\frac{e^{\theta_i^Tx}}{\sum_{j=1}^Ke^{\theta_j^Tx}}
$$
最终决策函数为$y^*={\arg \max}_i p(y=i|x,\theta)$

多元逻辑回归所对应的损失函数如下，
$$
\rm{Loss}(\theta)=-\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^K sign[y_i=j]\cdot \log \frac{e^{\theta_i^Tx}}{\sum_{k=1}^Ke^{\theta_k^Tx}}
$$

## References

- [逻辑回归总结点](https://www.jianshu.com/p/ece51cf6aa36\)
- [LRvsSVM](https://www.jianshu.com/p/ace5051d0023?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation)
- [Logistic Regression 模型简介](https://tech.meituan.com/2015/05/08/intro-to-logistic-regression.html)