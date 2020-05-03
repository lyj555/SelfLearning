目录
---
- [1. RNN简介](#1-RNN简介)
	- [1.1 RNN基本结构](#11-RNN基本结构)
	- [1.2 RNN的设计模式](#12-RNN的设计模式)
		- [1.2.1 每个时间步有输出,隐藏单元有循环连接](#121-每个时间步有输出,隐藏单元有循环连接)
		- [1.2.2 每个时间步有输出,隐藏单云无循环连接](#122-每个时间步有输出,隐藏单云无循环连接)
		- [1.2.3 隐藏单元有循环连接,最后一个时间步有输出](#123-隐藏单元有循环连接,最后一个时间步有输出)
	- [1.3 RNN Application](#13-RNN-Application)
	- [1.4 RNN Extensions](#14-RNN-Extensions)
	- [1.5 参考资源](#15-参考资源)
- [2. RNN-BPTT](#2-RNN-BPTT)	
	- [2.1 BPTT推导](#21-BPTT推导)
	- [2.2 梯度消失/爆炸](#22-梯度消失/爆炸)
	- [2.3 参考资源](#23-参考资源)
- [3. LSTM(1997)](#3-lstm1997)
	- [3.1 LSTM内部结构](#31-LSTM内部结构)
	- [3.2 LSTM如何实现长短期记忆(遗忘门输入门的作用)](#32-LSTM如何实现长短期记忆(遗忘门输入门的作用))
	- [3.3 LSTM中的激活函数是否可以随意更换](#33-LSTM中的激活函数是否可以随意更换)
	- [3.4 伪代码](#34-伪代码)
- [4. GRU(2014)](#4-GRU2014)
	- [4.1 GRU 内部结构](#41-gru内部结构)
	- [4.2 GRU伪代码](42-GRU伪代码)




## 1. RNN简介
循环神经网络(RNN, recurrent neural network)，是一类具有内部**环(loop)**的神经网络。不同于密集连接神经网络和卷积神经网络，RNN可以存在”记忆“，它可以记录输入和输入间的状态；它不同于前馈神经网络（将序列数据整体转换为向量然后一次性处理），它处理序列数据的方式是遍历所有的序列元素，并保存一个**状态(state)**，包含与已查看内容相关的序列。

### 1.1 RNN基本结构
RNN本质是一个**递推函数**, $h^{(t)}=f(h^{(t-1)};\theta)$,结合当前输入$x^{(t)}$即为$h^{(t)}=f(h^{(t-1)}, x^{(t)};\theta)$.    
以下为标准RNN计算图，
![](../../../pics/RNN.jpg) 
RNN的整体表达如下，
$$
\begin{aligned}
s_t &= Ux_t+Ws_{t-1} \\ 
h_t &= \rm{f}(s_t) \\ 
o_t &= Vh_t \\
\end{aligned}
$$
其中$\boldsymbol{W},\boldsymbol{U},\boldsymbol{V}$为权重，$h^{(t)}$为t时刻的输出，函数$f$表示激活函数
> Note:一般$s_0$初始化为0向量，使用`tanh`作为激活函数

### 1.2 RNN的设计模式
RNN通常有三种设计模式，  
- 每个时间步均有输出，且隐藏单元间有循环连接   
- 每个时间步都有输出，但是隐藏单元之间没有循环连接，只有当前时刻的输出到下个时刻的隐藏单元之间有循环连接   
- 隐藏单元之间有循环连接，但只有最后一个时间步有输出   

#### 1.2.1 每个时间步有输出,隐藏单元有循环连接
这种模式即通常所说的RNN结构，这种结构在每个时间步均会有输出，所以其经常应用于**Seq2Seq**任务中，比如序列标注、机器翻译等，如下图所示
![](../../../pics/RNN-设计模式1.png)

#### 1.2.2 每个时间步有输出,隐藏单云无循环连接
这种模式的表达能力弱于第一种，这种结构在每个时间步均有输出，但其隐藏单元不再有连接。   
正是因为这种设计，该模式的每个时间步可以与其他时间步单独训练，从而实现并行化，如下图所示
![](../../../pics/RNN-设计模式2.png)

#### 1.2.3 隐藏单元有循环连接,最后一个时间步有输出  
不同于上面的两种模式（每个时间步均有输出），该模式只有最后一个时间步有输出，这种网络一般用于概括序列。具体来说，就是产生固定大小的表示，用于下一步处理，在一些**Seq2One**中简单任务中，这种网络用的比较多,因为这些任务只需要关注序列的全局特征。

> Note: 前两种RNN被称为Elman Network和Jordan Network，通常说的RNN指前者 
>
> - Elman RNN   
    $$
    h^{(t)} = tanh(\boldsymbol{W}_hx^{(t)}+\boldsymbol{U_h}h^{(t-1)}+b_h) \\
     y^{(t)} = softmax(\boldsymbol{W}_yh^{(t)}+b_y)
    $$
> - Jordan RNN   
    $$
    h^{(t)} = tanh(\boldsymbol{W}_hx^{(t)}+\boldsymbol{U_h}y^{(t-1)}+b_h) \\
     y^{(t)} = softmax(\boldsymbol{W}_yh^{(t)}+b_y)
    $$

### 1.3 RNN Application  
- Language Modeling and Generating Text   
- Machine Translation  
Machine Translation is similar to language modeling in that our input is a sequence of words in our source language (e.g. German). We want to output a sequence of words in our target language (e.g. English). A key difference is that our output only starts after we have seen the complete input, because the first word of our translated sentences may require information captured from the complete input sequence.  
![](../../../pics/rnn_translation.png)

- Speech Recognition   
- Generating Image Descriptions  

### 1.4 RNN Extensions
- Bidirectional RNNs   
- Deep (Bidirectional) RNNs   
- LSTM networks  

### 1.5 参考资源   
- [Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)

- [RNN基本结构](https://github.com/imhuay/Algorithm_Interview_Notes-Chinese/blob/master/A-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/B-%E4%B8%93%E9%A2%98-RNN.md#rnn-%E7%9A%84%E5%9F%BA%E6%9C%AC%E7%BB%93%E6%9E%84)


## 2. RNN-BPTT

BPTT(Backpropagation Through Time),基于时间反向梯度传播算法，RNN是一种基于时序数据的的神经网络模型，因此传统的BP算法不适用该模型的优化，因此BPTT算法应运而生。

### 2.1 BPTT推导
简记RNN结构如下，
$$
\begin{aligned}
s_t &= Ux_t+Ws_{t-1} \\ 
h_t &= \rm{tanh}(s_t) \\ 
z_t &= Vh_t \\
\hat{y_t} &= \rm{softmax}(z_t) 
\end{aligned}
$$
以`cross entrophy loss`定义损失，
$$
\begin{aligned}
E_t(y_t, \hat{y_t}) &= -y_t \rm{log}\hat{y_t} \\
E(y, \hat{y}) &= \sum\limits_{t} E_t(y_t, \hat{y_t}) \\
&= -\sum\limits_{t}y_t \log \hat{y_t} 
\end{aligned}
$$
其中$y_t$是$t$步的真实值，$\hat{y_t}$是预测值。通常完整句子表示一个训练样本，所以整体损失就是每一步损失之和，也就是$E(y,\hat{y})$

BPTT算法的整体优化思路与BP算法类似，RNN中，我们目的求得$\frac{\partial E}{\partial U}, \frac{\partial E}{\partial W}, \frac{\partial E}{\partial V}$，在上面提到整体损失是每个时间步损失的和，所以说$\frac{\partial E}{\partial U}=\sum_t\frac{\partial E_t}{\partial U}, \frac{\partial E}{\partial W}=\sum_t\frac{\partial E_t}{\partial W}, \frac{\partial E}{\partial V}=\sum_t\frac{\partial E_t}{\partial V}$
接下来只需要求每个时刻的偏导数即可。  

- $\frac{\partial E_t}{\partial V}$的计算   
根据链式法则，有
$$
\frac{\partial E_t}{\partial V} = \frac{\partial E_t}{\partial \hat{y_t}} \cdot \frac{\partial \hat{y_t}}{\partial z_t} \cdot \frac{\partial z_t}{\partial V} 
$$
- $\frac{\partial E_t}{\partial W}$的计算  
由于$W$是各个时刻共享的，所以$t$时刻之前每个时刻$U$的变化对$E_t$均有贡献。  
$$
\frac{\partial E_t}{\partial W} = \sum\limits_{k=0}^t {\frac{\partial E_k}{\partial s_k} \cdot \frac{\partial s_k}{\partial W}}
$$
- $\frac{\partial E_t}{\partial U}$的计算  
计算方式类似$\frac{\partial E_t}{\partial W}$  
$$
\frac{\partial E_t}{\partial U} = \sum\limits_{k=0}^t {\frac{\partial E_k}{\partial s_k} \cdot \frac{\partial s_k}{\partial U}}
$$

### 2.2 梯度消失/爆炸

上面$\frac{\partial E_t}{\partial W}$计算中，按照链式法则展开后有

$$
\begin{aligned}
\frac{\partial E_t}{\partial W} &= \sum\limits_{k=0}^t {\frac{\partial E_k}{\partial \hat{y_t}} \cdot \frac{\partial \hat{y_t}}{\partial z_t} \cdot \frac{\partial z_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial s_t} \cdot \frac{\partial s_t}{\partial s_k} \cdot \frac{\partial s_k}{\partial W}}  \\
&= \sum\limits_{k=0}^t {\frac{\partial E_k}{\partial \hat{y_t}} \cdot \frac{\partial \hat{y_t}}{\partial z_t} \cdot \frac{\partial z_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial s_t} \cdot \left(\prod\limits_{j=k+1}^t{\frac{\partial s_j}{\partial s_{j-1}}} \right) \cdot \frac{\partial s_k}{\partial W}}
\end{aligned}
$$

上式中，注意到$\frac{\partial s_j}{\partial s_{j_1}}$是对向量进行求偏导，所以结果是一个矩阵(Jacobian matrix)。因为tanh激活函数将值映射到(-1, 1)，导数范围(0, 1)，sigmoid激活函数将值映射到(0, 1)，导数范围(0, 0.25)，可以证明矩阵的二阶范数的上界是1. 一旦当矩阵中的值接近饱和，当矩阵相乘时，其值就会指数级别下降，造成梯度消失，换言之，这种现象导致RNN不能学习到长期的依赖关系。对于前馈神经网络来说当层数非常深时，也会面临同样的问题，梯度消失。  
**解决方式之一便是替换激活函数，比如换为Relu，但这样虽然可以避免梯度消失的问题，**但是存在梯度爆炸问题（问题本质是各个单元的参数共享，还是存在矩阵连乘的问题），所以一个改进的方式是将参数$W$初始化为单位矩阵。  

> 为什么CNN中使用Relu较少出现上面的问题
> 主要原因是CNN中每层的参数$W$不同，且在初始化时，是独立同分布的，可以在一定程度上可以相互抵消，即使多层之后较小可能出现上面的问题

### 2.3 参考资源 
- [BPTT的详细推导](https://www.cnblogs.com/wacc/p/5341670.html)   
- [Recurrent Neural Networks Tutorial, Part 3 – Backpropagation Through Time and Vanishing Gradients](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)


## 3. LSTM(1997)
LSTM(Long Short-Term Memory)，长短期记忆神经网络，是循环神经网络的一种。在上面标准RNN结构中，由于存在梯度消失的问题，所起其难以学习到长期的依赖，LSTM的设计的**门机制**可以很大程度上避免梯度的消失，而学习到长期的依赖关系。    

LSTM网络的框架仍然标准的RNN框架（下图），而和标准RNN框架不同的是其计算隐含层的状态。标准的RNN计算计算隐藏层是
$s_t = f(x_t, s_{t-1}) = \rm{tanh}(Ux_t+Ws_{t-1})$，其中$U,W$是参数，$x_t$是第$t$步的输入，$s_{t-1}$是$t-1$步的隐藏层计算的状态，而LSTM只是改进了函数$f$，可以理解$s_t = LSTM(x_t, s_{t-1})$，接下俩看LSTM的具体计算模式。
<div> <img src="../../../pics/gru-lstm.png" style="zoom:80%" width="700px" /></div>
### 3.1内部结构  
- LSTM在标准RNN结构上加入了**门控机制**来限制信息的流动。 
传统的LSTM（下）和标准RNN（上）内部比较  

<div align="center"><img src="../../../pics/LSTM3-SimpleRNN.png" style="zoom:70%" width="700px" /></div>
<div align="center"><img src="../../../pics/LSTM3-chain.png" style="zoom:70%" width="700px" /></div>
- 总体来说，LSTM中加入了三个门：**遗忘门f**,**输入门i**,**输出门o**，以及一个**内部记忆状态C** 
    - 遗忘门f   
    遗忘门控制前一步记忆状态中有多少可以被遗忘，相当于之前的记忆状态多大比例遗忘，如下所示  
    <div> <img src="../../../pics/LSTM3-focus-f.png" style="zoom:80%" width="700px" /></div>
    - 输入门i   
    输入门控制当前的状态多大程度可以更新至记忆状态，相当于从目前的记忆中抽取一定的比例添加至记忆状态中，如下所示  
    <div> <img src="../../../pics/LSTM3-focus-i.png" style="zoom:80%" width="700px" /></div>
    - 记忆状态C   
    记忆状态由遗忘门和输入门共同决定，相当于遗忘门以往掉一部分信息，然后输入门再添加一些信息，如下所示   
    <div> <img src="../../../pics/LSTM3-focus-C.png" style="zoom:80%" width="700px" /></div>
    - 输出门o   
    控制当前的输出多大程度上取决于当前的记忆状态，相当于从目前记忆状态中提取多大比例输出
    <div> <img src="../../../pics/LSTM3-focus-o.png" style="zoom:80%" width="700px" /></div>
    
### 3.2 实现长短期记忆的实现
LSTM主要通过**遗忘门和输入门来实现长短期的记忆**   
- 如果当前时间点的状态中没有重要信息，遗忘门 f 中各分量的值将接近 1（f -> 1）；输入门 i 中各分量的值将接近 0（i -> 0）；此时过去的记忆将会被保存，从而实现**长期记忆**；   

- 如果当前时间点的状态中出现了重要信息，且之前的记忆不再重要，则 f -> 0，i -> 1；此时过去的记忆被遗忘，新的重要信息被保存，从而实现**短期记忆**；

- 如果当前时间点的状态中出现了重要信息，但旧的记忆也很重要，则 f -> 1，i -> 1    

### 3.3 激活函数是否可以随意更换
- 在 LSTM 中，所有控制门都使用 sigmoid 作为激活函数（遗忘门、输入门、输出门）  
- 在计算候选记忆或隐藏状态时，使用双曲正切函数 tanh 作为激活函数   

- **sigmoid的饱和性**   
    - 所谓饱和性，即输入超过一定范围后，输出几乎不再发生明显变化了
    - sigmoid 的值域为 (0, 1)，符合门控的定义
        - 当输入较大或较小时，其输出会接近 1 或 0，从而保证门的开或关；
        - 如果使用非饱和的激活函数，将难以实现门控/开关的效果。
    - sigmoid 是现代门控单元中的共同选择。

- **tanh的作用**   
    - 使用 tanh 作为计算状态时的激活函数，主要是因为其值域为 (-1, 1)  
        - 一方面，这与多数场景下特征分布以 0 为中心相吻合；
        - 另一方面，可以避免在前向传播的时候发生数值问题（主要是上溢）
    -  此外，tanh 比 sigmoid 在 0 附近有更大的梯度，通常会使模型收敛更快。
> 早期，使用 h(x) = 2*sigmoid(x) - 1 作为激活函数，该激活函数的值域也是 (-1, 1)

### 3.4 参数量的计算

总接来看整体分为为四部分的，如下：
$$
\begin{cases}
i_t = \sigma(W_i[x_t; h_{t-1}]+b_i) \\
f_t = \sigma(W_f[x_t; h_{t-1}]+b_f) \\
o_t = \sigma(W_o[x_t; h_{t-1}]+b_o) \\
\tilde{C_t} = \sigma(W_c[x_t; h_{t-1}]+b_c) \\
\end{cases}
$$

$$
\begin{cases}
C_t = i_t \ast \tilde{C_t} + f_t \ast C_{t-1} \\ 
h_t = o_t \ast tanh(C_t)
\end{cases}
$$

参数整体分为四部分，每部分均为$W$和$b$。

假设$x_t$的向量维度为$R^{n \times 1}$，$h_t$的向量维度为$R^{h \times 1}$。

在文本中，其实$n$对应着embedding的维度，也即词向量的维度；$h$代表隐向量的维度。

其中$[x_t; h_{t-1}]$代表他们纵向拼接，相当于此部分的维度为$R^{(n+h) \times 1}$。

那么对应$W$的维度为$R^{h \times (h+h)}$，$b$对应的维度为$R^{h \times 1}$。

所以整体的参数量为$4 \times (h*(n+h) + h)$

### 3.5 bi-LSTM的pytorch代码

```python
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNN, self).__init__()       
        # [0-10001] => [100] 编码10000个词，每个词有100个特征
        self.embedding = nn.Embedding(vocab_size, embedding_dim) #[10000,100]
        # [100] => [256] 使用bi-lstm,使用dropout防止过拟合
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,                       
                           bidirectional=True, dropout=0.5)
        # [256*2] => [1]
        self.fc = nn.Linear(hidden_dim*2, 1) 
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        x: [seq_len, b] vs [b, 3, 28, 28]
        """
        # [seq, b, 1] => [seq, b, 100] 先编码再dropout
        embedding = self.dropout(self.embedding(x))
 
        # output: [seq, b, hid_dim*2]
        # c/h: [num_layers*2, b, hid_dim]
        output, (hidden, cell) = self.rnn(embedding) #[h,c]默认为0所以输入省了
        
        # 取ht,和ht2做一次连接
        # [num_layers*2, b, hid_dim] => 2 of [b, hid_dim] => [b, hid_dim*2]
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # [b, hid_dim*2] => [b, 1]
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out
```



### 3.6 伪代码 
```python
def LSTM_CELL(prev_ct, prev_ht, input):
    '''
    cell of lstm network
    prev_ct: 前一时刻记忆状态 
    prev_ht: 前一时刻的输出的状态
    input: 当前时刻的输入
    :return 
      ht: 当前时刻隐藏层的状态
      Ct:当前时刻记忆状态
    '''
    ft = forget_layer(prev_ht, input)  # 遗忘门，遗忘比例  
    
    it = input_layer(prev_ht, input)  # 输入门，更新比例  
    candidate = candiate_layer(prev_ht, input)  # 当前输入和之前隐藏层所有内容
    
    Ct = ft*prev_ct + it*candidate  # 更新当前时刻的记忆状态  
    
    ot = output_layer(prev_ht, input)  # 输出门，输出比例 
    ht = ot*tanh(Ct)  # 当前时刻的内容
    return ht, Ct
```
### 3.7 参考资源

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)   
- [LSTM内部结构以及问题](https://github.com/imhuay/Algorithm_Interview_Notes-Chinese/blob/master/A-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/B-%E4%B8%93%E9%A2%98-RNN.md#lstm-%E7%9A%84%E5%86%85%E9%83%A8%E7%BB%93%E6%9E%84)
- [实战PyTorch（一）：Bi-LSTM 情感分类实战](https://blog.csdn.net/sleepinghm/article/details/105121339)

## 4. GRU(2014)
GRU(Gated Recurrent Unit)，和LSTM类似，也是带有门机制的循环神经网络。GRU中设计两个门，分别是更新门和重置门，其参数量要少于LSTM。  

### 4.1 GRU 内部结构
- GRU内部结构  
GRU中某个CELL的示意图如下，
<div> <img src="../../../pics/LSTM3-var-GRU.png" style="zoom:80%" width="700px" /></div>
相比较于LSTM，GRU将遗忘门和输入门合并为**更新门(update)z**，使用**重置门(reset)**r代替输出门。

- 重置门   
更新门用于控制前一时刻隐含层输出状态的比例，如下， 
$$
\begin{aligned}
r_t &= \sigma(W_r\cdot [h_{t-1},x_t]) \\
\tilde{h_t} &= \rm{tanh} (W \cdot [r_t*h_{t-1}, x_t])
\end{aligned}
$$

- 更新门   
控制前一时刻状态信息融合到当前信息的比例，如下，
$$
\begin{aligned}
z_t &= \sigma(W_z\cdot [h_{t-1},x_t]) \\
h_t &= (1-z_t)*h_{t-1} + z_t*\tilde{h_t}
\end{aligned}
$$



### 4.2 GRU伪代码  

```python
def GRU_CELL(prev_ht, input):
    '''
    cell of GRU.
    input param:
        prev_ht: 前一时刻的输出的状态 
        input: 当前时刻输入
    return: 
        ht: 当前层输出的状态
    '''
    rt = reset_layer(prev_ht, x_t)  # 重置门，确定更新比例
    
    hht = candidate_layer(rt*prev_ht, x_t)  # 当前层的计算内容
    
    zt = update_layer(prev_ht, x_t)  # 更新门，确定更新比例 
    ht = zt*hht + (1-zt)*prev_ht  # 当前层输出的状态
    return ht
```


