import tensorflow as tf
from tensorflow.keras import backend as K


def consume_gathered(g):
    pass


def consume_backend_gathered(b):
    pass


# `tf.gather` selects whole slices along the first axis, so the result is indexed by the
# indices' shape with each entry a row of the table. It is NOT the table's own shape, which
# is what a pass-through model would report.
table = tf.ones((5000, 8), dtype=tf.float32)
indices = tf.ones((2, 256), dtype=tf.int32)

gathered = tf.gather(table, indices)
assert gathered.shape == (2, 256, 8) and gathered.dtype == tf.float32
consume_gathered(gathered)

# The Keras backend alias reaches the same operation, which is the form the corpus uses.
backend_gathered = K.gather(table, indices)
assert backend_gathered.shape == (2, 256, 8) and backend_gathered.dtype == tf.float32
consume_backend_gathered(backend_gathered)
