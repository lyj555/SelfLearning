# -*- coding: utf-8 -*-

import sys
import numpy as np
from keras.utils import plot_model
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths

from keras_bert.datasets import get_pretrained, PretrainedList
model_path = get_pretrained(PretrainedList.chinese_base)  # download chinese pre-trained model

paths = get_checkpoint_paths(model_path)
model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, seq_len=10)
model.summary(line_length=120)
plot_model(model, to_file="keras_bert.png", show_shapes=True)  # loss确定 SEP 标记？

token_dict = load_vocabulary(paths.vocab)

tokenizer = Tokenizer(token_dict)
text = '语言模型'
tokens = tokenizer.tokenize(text)
print('Tokens:', tokens)
indices, segments = tokenizer.encode(first=text, max_len=10)
print("indices:", indices)
print("segments:", segments)

predicts = model.predict([np.array([indices]), np.array([segments])])[0]
for i, token in enumerate(tokens):
    print(token, predicts[i].tolist()[:5])  # extract word embedding


# load and predict

model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, training=True, seq_len=None)
plot_model(model, to_file="keras_bert_training.png", show_shapes=True)

token_dict_inv = {v: k for k, v in token_dict.items()}
text = '数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科'
tokens = tokenizer.tokenize(text)
tokens[1] = tokens[2] = '[MASK]'
print('Tokens:', tokens)

indices = np.array([[token_dict[token] for token in tokens]])
segments = np.array([[0] * len(tokens)])
masks = np.array([[0, 1, 1] + [0] * (len(tokens) - 3)])

predicts = model.predict([indices, segments, masks])[0].argmax(axis=-1).tolist()

print('Fill with: ', list(map(lambda x: token_dict_inv[x], predicts[0][1:3])))

sentence_1 = '数学是利用符号语言研究數量、结构、变化以及空间等概念的一門学科。'
sentence_2 = '从某种角度看屬於形式科學的一種。'
print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))

indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2)
masks = np.array([[0] * len(indices)])

predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]

print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))

sentence_2 = '任何一个希尔伯特空间都有一族标准正交基。'
print('Tokens:', tokenizer.tokenize(first=sentence_1, second=sentence_2))
indices, segments = tokenizer.encode(first=sentence_1, second=sentence_2)
masks = np.array([[0] * len(indices)])

predicts = model.predict([np.array([indices]), np.array([segments]), masks])[1]
print('%s is random next: ' % sentence_2, bool(np.argmax(predicts, axis=-1)[0]))

from keras_bert import Tokenizer

token_dict = {
    '[CLS]': 0,
    '[SEP]': 1,
    'un': 2,
    '##aff': 3,
    '##able': 4,
    '[UNK]': 5,
}
tokenizer = Tokenizer(token_dict)
print(tokenizer.tokenize('unaffable'))  # 分词结果是：`['[CLS]', 'un', '##aff', '##able', '[SEP]']`

indices, segments = tokenizer.encode('unaffable')
print(indices)   # 词对应的下标：`[0, 2, 3, 4, 1]`
print(segments)  # 段落对应下标：`[0, 0, 0, 0, 0]`

print(tokenizer.tokenize(first='unaffable', second='钢'))
# 分词结果是：`['[CLS]', 'un', '##aff', '##able', '[SEP]', '钢', '[SEP]']`
indices, segments = tokenizer.encode(first='unaffable', second='钢', max_len=10)
print(indices)   # 词对应的下标：`[0, 2, 3, 4, 1, 5, 1, 0, 0, 0]`
print(segments)  # 段落对应下标：`[0, 0, 0, 0, 0, 1, 1, 0, 0, 0]`


#  训练和使用

import keras
from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs


# 随便的输入样例：
sentence_pairs = [
    [['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']],
    [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']],
    [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']],
]


# 构建自定义词典
token_dict = get_base_dict()  # 初始化特殊符号，如`[CLS]`
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())  # Used for selecting a random word

# 构建和训练模型
model = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
)
compile_model(model)
model.summary()

plot_model(model, to_file="model.png", show_shapes=True)


def _generator():
    while True:
        yield gen_batch_inputs(
            sentence_pairs,
            token_dict,
            token_list,
            seq_len=20,
            mask_rate=0.3,
            swap_sentence_rate=1.0,
        )


model.fit_generator(
    generator=_generator(),
    steps_per_epoch=1000,
    epochs=100,
    validation_data=_generator(),
    validation_steps=100,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ],
)


inputs, output_layer = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
    training=False,      # 当`training`是`False`，返回值是输入和输出
    trainable=False,     # 模型是否可训练，默认值和`training`相同
    output_layer_num=4,  # 最后几层的输出将合并在一起作为最终的输出，只有当`training`是`False`有效
)
plot_model(model, to_file="model.png", show_shapes=True)


from keras_bert import extract_embeddings, POOL_NSP, POOL_MAX

# model_path = 'xxx/yyy/uncased_L-12_H-768_A-12'
texts = [
    ('all work and no play', 'makes jack a dull boy'),
    ('makes jack a dull boy', 'all work and no play'),
]

embeddings = extract_embeddings(model_path, texts, output_layer_num=4, poolings=[POOL_NSP, POOL_MAX])


