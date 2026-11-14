# Test `ContainerElements` where a function the call graph does not reach resizes a program's list
# stored as an attribute of a Keras model, `model.items.append(...)`: Keras keeps the list's
# storage, and only TensorFlow's module itself resizes no list passed to its functions, so no list
# reports a length (wala/ML#1182).
import threading

import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 3


xs = [x, y]
model = tf.keras.Sequential()
model.items = xs


def grow():
    model.items.append(x)


thread = threading.Thread(target=grow)
thread.start()
thread.join()
sink_list(xs)
