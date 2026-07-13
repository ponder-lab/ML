import tensorflow as tf


def consume(t):
    pass


# A reshape target built from `tf.shape` extractions (wala/ML#722): element `i`
# of `tf.shape(x)` is the runtime size of `x`'s axis `i`, so each extraction
# carries the axis's own classification. `x`'s batch axis is graph-time `None`
# (dynamic evidence), so `batch` must type *dynamic*; the sequence axis is the
# concrete 4 (TensorFlow constant-folds `tf.shape` over statically known
# axes), so `seq` must fold to the constant, not degrade to *unresolved*.
inp = tf.keras.Input(shape=(4, 6))

batch = tf.shape(inp)[0]
seq = tf.shape(inp)[1]
d = tf.shape(inp)[-1]

h = tf.reshape(inp, [-1, d])
w = tf.ones((6, 3))
out = tf.matmul(h, w)
out = tf.reshape(out, [batch, seq, 3])

model = tf.keras.Model(inp, out)
result = model(tf.ones((2, 4, 6)))

assert result.shape == (2, 4, 3)
assert result.dtype == tf.float32

consume(out)
