# BERT

BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder，因为decoder是不能获取要预测的信息。模型的主要创新点都在pre-train方法上，即用了Masked LM和Next Sentence Prediction，两种方法分别捕捉词语和句子级别的representation，在整体网络模式上和Open AI GPT类似，分为预训练和微调两个阶段。

BERT的整体网络结构由多层的Transformer Encoder组成，如下图所示，

![img](../../../../pics/bert.jpg?lastModify=1614408943)

## 1. 模型架构

BERT的整体是一个多层、双向的Transformer Encoder的结构。

论文中提到的模型结构有两个，分别为BERT-base和BERT-large。

- BERT-base

  L=12，H=768，A=12，Total Parameters=110M

- BERT-large

  L=24，H=1024，A=16，Total Parameters=340M

> L: Transformer block的层数，H: hidden size，A: self-attention head number

**输入表示**

BERT的输入的编码向量（长度是512）是3个嵌入特征的单位和，如下图，这三个词嵌入特征是：

1. word piece嵌入（WordPiece Embedding），word piece是指将单词划分为一组有限的公共字词单元能在单词的有效性和字符的灵活性之间取得一个折中的平衡。例如图4的示例中‘playing’被拆分成了‘play’和‘ing’；（在中文应该是字粒度，没有这一说了）
2. 位置嵌入（Position Embedding），位置嵌入是指将单词的位置信息编码成特征向量，位置嵌入是模型中引入单词位置关系的至关重要的一环。
3. 分割嵌入（Segment Embedding），用于区分两个句子，例如B是否是A的下文（对话场景，问答场景等）。对于句子对，第一个句子的特征值是0，第二个句子的特征值是1。

![img](../../../../pics/bert_input_embedding.jpg?lastModify=1614408943)

> 上图中，有两个特殊符号[CLS]和[SEP]，其中[CLS]是special classification token，代表对序列的表示；[SEP]表示分句符号，用于断开输入语料中的两个sentence。

### 1.1 预训练任务

BERT是一个多任务模型，它的任务是由两个自监督任务组成，即MLM和NSP。

**Task 1: Masked LM**

为了训练一个比较深的双向表示，将输入进行了随机的mask，然后mask的内容作学习的目标。最终从输入中随机选择15%mask，将这些token作为预测目标。

在BERT的实验中，15%的WordPiece Token会被随机Mask掉。在训练模型时，一个句子会被多次喂到模型中用于参数学习，但是BERT并没有在每次都mask掉这些单词，而是在确定要Mask掉的单词之后，制定一个策略是，该词80%的时候将其替换为[Mask]，10%的时候将其替换为其它任意单词，10%的时候会保留原始Token。

- 80%：`my dog is hairy -> my dog is [mask]`

- 10%：`my dog is hairy -> my dog is apple`

- 10%：`my dog is hairy -> my dog is hairy`

  > 原因：如果句子中某个token100%mask掉，在fine-tuning的时候模型就会有一些没有见过的单词；加入随机token的原因就是想保持对每个输入token的分布式表征，否则模型就会记住这个[mask]为token 'hairy'，当然也会带来一些负面影响，不过一个单词随机替换的概率为15%*10%=1.5%，比例很小，可以忽略。

  > 论文提到这样做的原因是保持预训练和微调阶段的一致性，因为真实在微调阶段没有[MASK]标记。

**Task 2: Next Sentence Prediction**

Next Sentence Prediction（NSP）的任务是判断句子B是否是句子A的下文。如果是的话输出`IsNext`，否则输出`NotNext`。训练数据的生成方式是从平行语料中随机抽取的连续两句话，其中50%保留抽取的两句话，它们符合IsNext关系，另外50%的第二句话是随机从预料中提取的，它们的关系是NotNext的。这个关系保存在上图中的`[CLS]`符号中。形如下，

  ```
Input = [CLS] the man went to [MASK] store [SEP]
he bought a gallon [MASK] milk [SEP]
Label = IsNext
  
Input = [CLS] the man [MASK] to the store [SEP]
penguin [MASK] are flight ##less birds [SEP]
Label = NotNext  
  ```

### 1.2 微调任务

因为Transformer中存在的self-attention机制，使得BERT可以对接绝大多数的下游任务（无论输入是单块文本或者多块文本）。

下面为NLP中任务的微调使用模式，

  - 句子关系类任务

    这类任务往往输入包含两段文本，输出为一个值，如语义相似度任务，句对推理任务等。

    - 输入端

      输入往往包含两部分，将两部分内容用[SEP]分割，然后放进BERT。形如"[CLS]text1[SEP]text2"

    - 输出端

      在[CLS]位置上面串接一个softmax分类层或者再加其他网络结构。

  - 单句分类任务

    这类任务的输入包含一段文本，输出为一个值，。

    - 输入端

      形如"[CLS]text[SEP]"

    - 输出端

      在[CLS]位置上面串接一个softmax分类层或者再加其他网络结构。

  - 阅读理解任务

    这类任务输入包含两段文本，输出为一段文本。

    - 输入端

      和句子关系类任务类似，形如"[CLS]text1[SEP]text2"，如果以阅读理解为例，则text1部分为question，text2为passage

    - 输出端

      如果以span extraction为例，则基于text2找到其start position和end_position（对应答案的起始位置和结束位置）

  - 序列标注任务

    这类任务输入为一段文本，输出也为一段文本，且出入和输出等长。

    - 输入端

      形如"[CLS]text[SEP]"

    - 输出端

      基于text的每一个token进行操作

如下图所示，

![](../../../../pics/bert_use2.jpg)

整体来看，NLP四大类任务（序列标注，分类，句子关系判断，生成式任务），除生成式任务外，都可以比较方便地改造成BERT能够接受的方式。这意味着它几乎可以做任何NLP的下游任务。

## 2. 词向量

  主要是两种表征方式，一个是微调预训练模型，另外一个是直接通过预训练模型产生词向量（feature-based）。

  - 微调预训练模型（这种是主要的形式）

    这个就是上面所说的方式，基于预训练的模型，根据目标任务建立目标函数，微调BERT模型。

  - 直接产生词向量（featuer-based）

    相当于不需要微调模型，直接基于预训练的模型产生词向量，然后根据目标任务后面接其它的层。

    > 论文提到这种方式可以实现不错的效果，一些任务上要比微调的方式差些

## 3. 总结

  - 优点

    - 提出了双向Transformer encoder表示，语言模型中利用了预测词的context（GPT是单向，ELMo虽然是双向但是分开训练，两个目标函数），学习到的表征能够融合两个方向上的context。
    - 预训练时采用MLM和NSP联合构建构建学习的目标函数（可以通俗理解为玩型填空和句对预测）。

    > 个人理解，加入MLM机制主要原因是想训练深层的语言模型，其次是单词的双向语言表征，如果不加入MLM，每个词预测的话，显然不适用深度模型；加入NSP为了增加两个句子之间的理解，实验证明这对QA和NLI任务有很大的帮助。

  - 缺点

    - 在MLM中，随机mask一些token，默认词之间相互独立，这会损失一些信息，这个促使了XLNet的诞生。
    - 预训练阶段因为采取引入[Mask]标记来Mask掉部分单词的训练模式，而微调阶段是看不到这种被强行加入的Mask标记的，所以两个阶段存在使用模式不一致的情形，这可能会带来一定的性能损失

## 4. Q&A

  - 倘若不加MLM的话，也就是每个单词进行预测，来构建深层双向模型，会存在什么问题？

    过拟合

  - 使用MLM的作用是啥？

    个人总结：充分学习一个词汇的context信息（双向信息），倘若不做MLM的，模型非常容易记住预测目标词汇，造成过拟合现象，且非常难学习到词汇的context信息，反之通过MLM机制增加模型的泛化学习能力；另外就是这种MLM的设计机制可以使得网络结构比较大（15%的mask词汇中挑选80%作为预测，10%随机填充，10%真实词汇，这部分的设计主要是缓解预训练和微调阶段的不一致），给模型增加了一些负面信息，让模型更加充分学习周围词汇的相关信息（或者说通过增加扰动迫使模型去学习更加深层的信息）。

    > 先整体进行介绍，然后再介绍具体的内部机制
    >
    > 1. 学习context信息，仅mask 15%
    > 2. 弥补训练和微调的不一致（10%真实词汇）
    > 3. 更加充分的词汇的学习（10%随机词汇）

  - 简单介绍BERT模型？

    首先BERT（Bidirectional Encoder From Transformer）是基于一个双向编码的且基于Transformer的语言模型。其中双向是指同时考虑预测词汇的前文信息和后文信息。整体模式是预训练和微调。在预训练阶段创新性的采用了MaskedLM（学习预测词的上下文语义）和NSP（句子级别信息），模型搭建上，基于transformer网络进行了深层的堆叠，来捕获文本的深层语义，整体效果在NLP各个领域基本达到了SOTA效果。
    
- 不考虑多头的原因，self-attention中词向量不乘QKV参数矩阵，会有什么问题？

  Self-Attention的核心是**用文本中的其它词来增强目标词的语义表示**，从而更好的利用上下文的信息。

  self-attention中，sequence中的每个词都会和sequence中的每个词做点积去计算相似度，也包括这个词本身。

  对于 self-attention，一般会说它的 q=k=v，这里的相等实际上是指它们来自同一个基础向量，而在实际计算时，它们是不一样的，因为这三者都是乘了QKV参数矩阵的。那如果不乘，每个词对应的q,k,v就是完全一样的。

  在相同量级的情况下，qi与ki点积的值会是最大的（可以从“两数和相同的情况下，两数相等对应的积最大”类比过来）。

  那在softmax后的加权平均中，该词本身所占的比重将会是最大的，使得其他词的比重很少，无法有效利用上下文信息来增强当前词的语义表示。

  而乘以QKV参数矩阵，会使得每个词的q,k,v都不一样，能很大程度上减轻上述的影响。

  当然，QKV参数矩阵也使得多头，类似于CNN中的多核，去捕捉更丰富的特征/信息成为可能。

- 为什么BERT选择mask掉15%这个比例的词，可以是其他的比例吗？

  BERT采用的Masked LM，会选取语料中所有词的15%进行随机mask，论文中表示是受到完形填空任务的启发，但其实**与CBOW也有异曲同工之妙**。

  从CBOW的角度，这里p=15%有一个比较好的解释是：在一个大小为$1/p=100/15\approx 7$的窗口中随机选一个词，类似CBOW中滑动窗口的中心词，区别是这里的滑动窗口是非重叠的。

- 为什么BERT在第一句前会加一个[CLS]标志?

  BERT在第一句前会加一个[CLS]标志，最后一层该位对应向量可以作为整句话的语义表示，从而用于下游的分类任务等。

  为什么选它呢，因为与文本中已有的其它词相比，这个无明显语义信息的符号会**更“公平”地融合文本中各个词的语义信息**，从而更好的表示整句话的语义。

  一种是get_pooled_out()，就是上述[CLS]的表示，输出shape是[batch size,hidden size]。

  一种是get_sequence_out()，获取的是整个句子每一个token的向量表示，输出shape是[batch_size, seq_length, hidden_size]

- Transformer在哪里做了权重共享，为什么可以做权重共享？

  Transformer在两个地方进行了权重共享：

  **（1）**Encoder和Decoder间的Embedding层权重共享；

  **（2）**Decoder中Embedding层和FC层权重共享。

- BERT非线性的来源在哪里？

  前馈层的gelu激活函数和self-attention，self-attention是非线性的

- BERT的三个Embedding直接相加会对语义有影响吗？

  这是一个非常有意思的问题，苏剑林老师也给出了回答，真的很妙啊：

  > Embedding的数学本质，就是以one hot为输入的单层全连接。
  > 也就是说，世界上本没什么Embedding，有的只是one hot。

  BERT的三个Embedding相加，本质可以看作一个特征的融合，强大如 BERT 应该可以学到融合后特征的语义信息的。

- BERT如何解决长文本问题？

  这个问题现在的解决方法是用Sliding Window（划窗），主要见于诸阅读理解任务（如Stanford的SQuAD)。Sliding Window即把文档分成**有重叠**的若干段，然后每一段都当作独立的文档送入BERT进行处理。最后再对于这些独立文档得到的结果进行整合。

  Sliding Window可以只用在Training中。因为Test之时不需要Back Propagation，亦不需要large batchsize，因而总有手段将长文本塞进显存中（如torch.no*grad, batch*size=1）。

- Layer Normalization？

  LayerNorm实际就是对隐含层做层归一化，即对某一层的所有神经元的输入进行归一化。（每hidden_size个数求平均/方差）
   1、它在training和inference时没有区别，只需要对当前隐藏层计算mean and variance就行。不需要保存每层的moving average mean and variance。
   2、不受batch size的限制，可以通过online learning的方式一条一条的输入训练数据。
   3、LN可以方便的在RNN中使用。
   4、LN增加了gain和bias作为学习的参数。

  公式如下，
  $$
  LayerNorm(x) = \alpha \cdot \frac{x-\mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
  $$

  > 其中$\alpha$和$\beta$是可学习参数

  LayerNorm和BatchNorm对比，

  batch normalization的缺点：因为统计意义，在batch_size较大时才表现较好；不易用于RNN；训练和预测时用的统计量不同等。
  layer normalization就比较适合用于RNN和单条样本的训练和预测。但是在batch_size较大时性能时比不过batch normalization的。
