import tensorflow as tf


def consume_batchnorm(x):
    pass


def consume_relu(x):
    pass


def consume_chain(x):
    pass


# Applied directly, without a surrounding model: each of these layers returns its input's shape
# and dtype unchanged (wala/ML#840).
bn = tf.keras.layers.BatchNormalization()
bn_out = bn(tf.ones((32, 10)))
assert bn_out.shape == (32, 10), bn_out.shape
assert bn_out.dtype == tf.float32, bn_out.dtype
consume_batchnorm(bn_out)

relu = tf.keras.layers.ReLU()
relu_out = relu(tf.ones((32, 10)))
assert relu_out.shape == (32, 10), relu_out.shape
consume_relu(relu_out)

# A shape-preserving layer between a Dense and its consumer: the declared width must survive it,
# which is the case the issue is about.
fc = tf.keras.layers.Dense(10)
norm = tf.keras.layers.BatchNormalization()
act = tf.keras.layers.ReLU()
chain_out = act(norm(fc(tf.ones((32, 784)))))
assert chain_out.shape == (32, 10), chain_out.shape
consume_chain(chain_out)
