# Regression guard (wala/ML#548): a `tf.reshape` on the path to a `@tf.function` parameter must not
# degrade the inferred parameter. Mirrors the cifar10 path in
# TensorFlow2.0-Examples/3-Neural_Network_Architecture/main.py: tf.reshape(labels) feeds a
# tf.data.Dataset whose iteration binds the @tf.function's `labels` parameter. `consume(labels)`
# pins the type. See ponder-lab/Input-Signature-Inference-Paper#49 (corpus audit).
import tensorflow as tf

(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
y_train = tf.reshape(y_train, (-1,))

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)


def consume(t):
    pass


@tf.function
def train_step(images, labels):
    consume(labels)


for images, labels in train_ds:
    # Runtime truth mirrored by `testReshapeToParam`: the `tf.reshape`-fed `labels`
    # parameter stays an exact `uint8` vector — `(32,)` for the full batches and
    # `(16,)` for the final partial batch (50000 % 32 == 16).
    assert labels.dtype == tf.uint8
    assert labels.shape == (32,) or labels.shape == (16,)
    train_step(images, labels)
