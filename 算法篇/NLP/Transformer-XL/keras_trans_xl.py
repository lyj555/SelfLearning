# -*- coding: utf-8 -*-

import keras
from keras.utils import plot_model
import numpy as np
from keras_transformer_xl import MemorySequence, build_transformer_xl


class DummySequence(keras.utils.Sequence):

    def __init__(self):
        pass

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return np.ones((3, 5 * (index + 1))), np.ones((3, 5 * (index + 1), 3))


model = build_transformer_xl(
    units=4,
    embed_dim=4,
    hidden_dim=4,
    num_token=3,
    num_block=3,
    num_head=2,
    batch_size=3,
    memory_len=20,
    target_len=10,
)
plot_model(model, to_file="model.png", show_shapes=True)

seq = MemorySequence(
    model=model,
    sequence=DummySequence(),
    target_len=10,
)

pred = model.predict([model, seq], verbose=True)

