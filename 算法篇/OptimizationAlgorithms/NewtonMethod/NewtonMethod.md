# Newton Method

一般来说，牛顿法主要应用在两个方面，1：求方程的根；2：最优化。


  ## 求方程的根

目标是当$f(x) = 0$求$x$。

对于一些复杂的函数来说，求根往往没有解析解，所以往往通过迭代进行求解。

推导如下：

将函数$f(x)$在$x=x_0$处一阶泰勒展开，
$$
f(x) \approx f(x_0) + f^{\prime}(x_0)(x - x_0)
$$
令$f(x)=0$也即$f(x_0) + f^{\prime}(x_0)(x - x_0) = 0$，求得

$x = x_0 - \frac{f(x_0)}{f^{\prime}(x_0)}$，当然此时的解只是近似解，需要基于此继续迭代，由此有迭代公式为
$$
x_{n+1} = x_{n} - \frac{f(x_n)}{f^{\prime}(x_n)}
$$

## 优化

目标是最小化$f(x), x\in R^n$，显然这是无约束最优化问题。

设$f(x)$有二阶连续偏导数，若第$k$次迭代值为$x^{(k)}$，则将$f(x)$在$x=x^{(k)}$进行二阶泰勒展开，
$$
f(x) \approx f(x^{(k)}) + g(x^{(k)})^{T}\cdot (x-x^{(k)}) + \frac{1}{2} (x - x^{(k)})^T \cdot H(x^{(k)}) \cdot (x - x^{(k)})
$$
其中$g(x^{(k)}) = \frac{\partial f(x)}{\partial x^{(k)}} = \nabla f(x^{(k)})$，$H(x^{(k)}) = [\frac{\partial^2 f}{\partial x_i \partial x_j}]_{n \times n}$，$g(x^{(k)})$是一个向量，$H(x^{(k)})$是一个矩阵，也就是常说的黑塞矩阵(Hessian Matrix)



## References

- [梯度下降法、牛顿法和拟牛顿法](https://zhuanlan.zhihu.com/p/37524275)