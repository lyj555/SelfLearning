# -*- coding: utf-8 -*-

import os
from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI

checkpoint_path = '.../xlnet_cased_L-24_H-1024_A-16'

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


import os
import sys

import numpy as np

from keras_xlnet import PretrainedList, get_pretrained_paths
from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint
from keras_xlnet import ATTENTION_TYPE_UNI, ATTENTION_TYPE_BI


checkpoint_path = get_pretrained_paths(PretrainedList.en_cased_base)
vocab_path = checkpoint_path.vocab
config_path = checkpoint_path.config
model_path = checkpoint_path.model

