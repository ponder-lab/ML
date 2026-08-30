import tensorflow as tf

# wala/ML#832: a COUPLED user transform — a three-cycle axis rotation — whose repeated
# application is what the unknown-length rule's second comparison exists to catch: after one
# application axis 1 agrees with the input, after two it does not, so an agreeing-axes rule that
# applied the transform once would emit a concrete extent the runtime moves. The transpose hop
# computes the permutation in place from the body's own literal.


def consume_cycle_pair(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (1, 3, 2, 2), x.shape
    assert x.dtype == tf.float32


class Cycle(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.transpose(inputs, perm=[0, 2, 3, 1])


pair = tf.keras.Sequential([Cycle(), Cycle()])
result = pair(tf.ones((1, 2, 2, 3)))
assert result.shape == (1, 3, 2, 2), result.shape
consume_cycle_pair(result)


def consume_cycle_repeated(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (1, 3, 2, 2), x.shape
    assert x.dtype == tf.float32


repeated = []
for _ in range(2):
    repeated.append(Cycle())
consume_cycle_repeated(tf.keras.Sequential(repeated)(tf.ones((1, 2, 2, 3))))
