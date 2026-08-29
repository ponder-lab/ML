import numpy as np
import tensorflow as tf

# The wala/ML#855 reachability measurement: every candidate layer fed from a DESTRUCTURED tuple
# dataset element, the configuration whose element read has an implicit pointer key, so a
# generator reading its input by a bare points-to walk sees nothing while the value's dataflow
# state is exact (the wala/ML#845 mechanism). Each consumer pins one candidate's result.


def consume_bn(x):
    pass


def consume_relu(x):
    pass


def consume_leaky(x):
    pass


def consume_softmax(x):
    pass


def consume_masking(x):
    pass


def consume_activation(x):
    pass


def consume_pool2d(x):
    pass


def consume_zeropad(x):
    pass


def consume_convt(x):
    pass


def consume_flatten(x):
    pass


def consume_gap1d(x):
    pass


def consume_gmp1d(x):
    pass


rows = 8
imgs = tf.constant(np.ones((rows, 16, 16, 3), dtype=np.float32))
seqs = tf.constant(np.ones((rows, 20, 32), dtype=np.float32))
flags = tf.constant(np.ones((rows, 10), dtype=np.float32))
labels = tf.constant(np.ones((rows,), dtype=np.float32))

loaded = tf.data.Dataset.from_tensor_slices((imgs, seqs, flags, labels)).batch(
    4, drop_remainder=True
)

for X_img, X_seq, X_flags, Y in loaded:
    bn = tf.keras.layers.BatchNormalization()(X_img)
    assert bn.shape == (4, 16, 16, 3), bn.shape
    consume_bn(bn)

    relu = tf.keras.layers.ReLU()(X_img)
    assert relu.shape == (4, 16, 16, 3), relu.shape
    consume_relu(relu)

    leaky = tf.keras.layers.LeakyReLU()(X_img)
    assert leaky.shape == (4, 16, 16, 3), leaky.shape
    consume_leaky(leaky)

    soft = tf.keras.layers.Softmax()(X_img)
    assert soft.shape == (4, 16, 16, 3), soft.shape
    consume_softmax(soft)

    masked = tf.keras.layers.Masking(mask_value=-1.0)(X_img)
    assert masked.shape == (4, 16, 16, 3), masked.shape
    consume_masking(masked)

    act = tf.keras.layers.Activation("relu")(X_img)
    assert act.shape == (4, 16, 16, 3), act.shape
    consume_activation(act)

    pooled = tf.keras.layers.MaxPool2D(2)(X_img)
    assert pooled.shape == (4, 8, 8, 3), pooled.shape
    consume_pool2d(pooled)

    padded = tf.keras.layers.ZeroPadding2D()(X_img)
    assert padded.shape == (4, 18, 18, 3), padded.shape
    consume_zeropad(padded)

    convt = tf.keras.layers.Conv2DTranspose(5, 3)(X_img)
    assert convt.shape == (4, 18, 18, 5), convt.shape
    consume_convt(convt)

    flat = tf.keras.layers.Flatten()(X_img)
    assert flat.shape == (4, 768), flat.shape
    consume_flatten(flat)

    gap = tf.keras.layers.GlobalAveragePooling1D()(X_seq)
    assert gap.shape == (4, 32), gap.shape
    consume_gap1d(gap)

    gmp = tf.keras.layers.GlobalMaxPooling1D()(X_seq)
    assert gmp.shape == (4, 32), gmp.shape
    consume_gmp1d(gmp)
