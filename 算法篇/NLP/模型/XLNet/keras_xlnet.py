# -*- coding: utf-8 -*-

import os
from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI

from keras.utils import plot_model

checkpoint_path = './xlnet_cased_L-12_H-768_A-12'

tokenizer = Tokenizer(os.path.join(checkpoint_path, 'spiece.model'))
model = load_trained_model_from_checkpoint(
    config_path=os.path.join(checkpoint_path, 'xlnet_config.json'),
    checkpoint_path=os.path.join(checkpoint_path, 'xlnet_model.ckpt'),
    batch_size=16,
    memory_len=512,
    target_len=128,
    in_train_phase=False,
    attention_type=ATTENTION_TYPE_BI,
)
model.summary()

plot_model(model, to_file="xlnet.png", show_shapes=True)
