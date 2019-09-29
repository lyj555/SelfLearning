# -*- coding: utf-8 -*-

import keras
import keras.backend as K


class TrigPosEmbedding(keras.layers.Layer):
    """Position embedding use sine and cosine functions(Transformer).
    non trainable params
    Expand mode:
        # Input shape
            2D tensor with shape: `(batch_size, sequence_length)`.
        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
    Add mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
    Concat mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim + output_dim)`.
    """
    MODE_EXPAND = 'expand'
    MODE_ADD = 'add'
    MODE_CONCAT = 'concat'

    def __init__(self,
                 mode=MODE_ADD,
                 output_dim=None,
                 **kwargs):
        """
        :param output_dim: The embedding dimension.
        :param kwargs:
        """
        if mode in [self.MODE_EXPAND, self.MODE_CONCAT]:
            if output_dim is None:
                raise NotImplementedError('`output_dim` is required in `%s` mode' % mode)
            if output_dim % 2 != 0:
                raise NotImplementedError('It does not make sense to use an odd output dimension: %d' % output_dim)
        self.mode = mode
        self.output_dim = output_dim
        self.supports_masking = True
        super(TrigPosEmbedding, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'mode': self.mode,
            'output_dim': self.output_dim,
        }
        base_config = super(TrigPosEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        if self.mode == self.MODE_EXPAND:
            return input_shape + (self.output_dim,)
        if self.mode == self.MODE_CONCAT:
            return input_shape[:-1] + (input_shape[-1] + self.output_dim,)
        return input_shape

    def call(self, inputs, mask=None):
        input_shape = K.shape(inputs)
        if self.mode == self.MODE_ADD:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], input_shape[2]
            pos_input = K.tile(K.expand_dims(K.arange(0, seq_len), axis=0), [batch_size, 1])
        elif self.mode == self.MODE_CONCAT:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], self.output_dim
            pos_input = K.tile(K.expand_dims(K.arange(0, seq_len), axis=0), [batch_size, 1])
        else:  # expand mode
            output_dim = self.output_dim
            pos_input = inputs  # batch_size * seq_len
        if K.dtype(pos_input) != K.floatx():
            pos_input = K.cast(pos_input, K.floatx())
        evens = K.arange(0, output_dim // 2) * 2
        odds = K.arange(0, output_dim // 2) * 2 + 1
        even_embd = K.sin(
            K.dot(
                K.expand_dims(pos_input, -1),
                K.expand_dims(1.0 / K.pow(
                    10000.0,
                    K.cast(evens, K.floatx()) / K.cast(output_dim, K.floatx())
                ), 0)
            )
        )
        odd_embd = K.cos(
            K.dot(
                K.expand_dims(pos_input, -1),
                K.expand_dims(1.0 / K.pow(
                    10000.0, K.cast((odds - 1), K.floatx()) / K.cast(output_dim, K.floatx())
                ), 0)
            )
        )
        embd = K.stack([even_embd, odd_embd], axis=-1)
        output = K.reshape(embd, [-1, K.shape(inputs)[1], output_dim])
        if self.mode == self.MODE_CONCAT:
            output = K.concatenate([inputs, output], axis=-1)
        if self.mode == self.MODE_ADD:
            output += inputs
        return output


if __name__ == "__main__":
    import numpy as np

    seq_len = np.random.randint(1, 10)
    embd_dim = np.random.randint(1, 20) * 2
    indices = np.expand_dims(np.arange(seq_len), 0)  # batch_size * seq_len

    # test expand mode
    model = keras.models.Sequential()
    model.add(TrigPosEmbedding(
        input_shape=(seq_len,),
        mode=TrigPosEmbedding.MODE_EXPAND,
        output_dim=embd_dim,
        name='Pos-Embd',
    ))
    model.compile('adam', 'mse')
    model.summary()

    # model_path = os.path.join(tempfile.gettempdir(), 'test_trig_pos_embd_%f.h5' % np.random.random())
    # model.save(model_path)
    # model = keras.models.load_model(model_path, custom_objects={'TrigPosEmbedding': TrigPosEmbedding})
    # model.summary()
    predicts = model.predict(indices)[0].tolist()
    for i in range(seq_len):
        for j in range(embd_dim):
            actual = predicts[i][j]
            if j % 2 == 0:
                expect = np.sin(i / 10000.0 ** (float(j) / embd_dim))
            else:
                expect = np.cos(i / 10000.0 ** ((j - 1.0) / embd_dim))
            assert round(actual, 3) == round(expect, 3), f"{expect} != {actual}"


