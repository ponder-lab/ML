import numpy as np
import tensorflow as tf

# wala/ML#907 reduction: a Keras Layer whose `call` takes two type-hinted tensor parameters,
# invoked through `__call__`, once with numpy arrays built from the (2, 20) literal and once with
# tf.constant arrays. Separate layers and sinks so the two call parameters do not union across
# contexts. The question is whether the call parameter carries a RANK (not an extent) on each path.


def consume_numpy_call(t):
    pass


def consume_const_call(t):
    pass


def consume_numpy_helper(t):
    pass


def consume_const_helper(t):
    pass


def numpy_helper(data):
    # The module-level function the call passes one parameter into (the _gather_elements_along_row
    # role): if the call parameter has no rank, neither does what it hands on.
    consume_numpy_helper(data)
    return data


def const_helper(data):
    consume_const_helper(data)
    return data


class NumpyLayer(tf.keras.layers.Layer):
    def call(self, logits: tf.Tensor, labels: tf.Tensor):
        consume_numpy_call(logits)
        return numpy_helper(logits)


class ConstLayer(tf.keras.layers.Layer):
    def call(self, logits: tf.Tensor, labels: tf.Tensor):
        consume_const_call(logits)
        return const_helper(logits)


logits_shape = (2, 20)
rng = np.random.RandomState(42)
np_logits = rng.uniform(size=logits_shape).astype(np.float32)
np_labels = rng.permutation(np.eye(*logits_shape).T).T.astype(np.float32)
NumpyLayer()(np_logits, np_labels)

const_logits = tf.constant(1.0, shape=(2, 20))
const_labels = tf.constant(1.0, shape=(2, 20))
ConstLayer()(const_logits, const_labels)
