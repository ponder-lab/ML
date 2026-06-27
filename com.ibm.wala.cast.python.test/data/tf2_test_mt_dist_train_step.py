import numpy as np
import tensorflow as tf


def consume(z):
    pass


# Vendored from MusicTransformer-tensorflow2.0 (master <-> hybridize refactoring diff). The
# refactoring decorates `__dist_train_step` (model.py); its `inp` parameter is fed the training
# batch from `Data.seq2seq_batch` (data.py), which is `np.array(...)`. `__dist_train_step` is a
# plain method (not a Keras `call`), so this isolates the result from the implicit-`call()`
# blocker (wala/ML#106).
#
# `inp` lands at ⊥ (not tensor-classified): `np.array`'s tensor type is not propagated to a callee
# parameter (wala/ML#598), whereas a `tf.constant` feed recovers concretely. This pins that
# coverage loss. Once wala/ML#598 lands, `inp` should type and `consume` flips to a tensor
# parameter.
def seq2seq_batch():
    return np.array([[1, 2, 3], [4, 5, 6]])


class MusicTransformer:
    @tf.function
    def dist_train_step(self, inp):
        consume(inp)
        return inp


MusicTransformer().dist_train_step(seq2seq_batch())
