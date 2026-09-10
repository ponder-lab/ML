# A `tf.function` with a declared `input_signature`. The declaration pins the parameter's RANK and
# dtype; the extents are `None`, so rank 2 is the recoverable fact and concrete extents are not.
import numpy as np
import tensorflow as tf

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="Inputs"),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="Targets"),
]


def consume_declared(a):
    pass


def consume_helper(a):
    pass


def helper(seq):
    consume_helper(seq)
    return tf.cast(tf.math.equal(seq, 0), tf.float32)


@tf.function(input_signature=train_step_signature)
def train_step(inputs, targets):
    consume_declared(inputs)
    return helper(inputs)


out = train_step(tf.ones((4, 7), dtype=tf.int32), tf.ones((4, 7), dtype=tf.int32))
assert out.shape == (4, 7)


def consume_opaque_declared(a):
    pass


def consume_opaque_helper(a):
    pass


def opaque_helper(seq):
    consume_opaque_helper(seq)
    return tf.cast(tf.math.equal(seq, 0), tf.float32)


@tf.function(input_signature=train_step_signature)
def opaque_step(inputs, targets):
    consume_opaque_declared(inputs)
    return opaque_helper(inputs)


# The condition the subject is in: the ARGUMENT is unresolvable, so the declaration is the only
# source of a rank. A permutation of a transpose loses its rank for the analysis (wala/ML#910).
rng = np.random.RandomState(0)
opaque = tf.constant(rng.permutation(np.ones((7, 4)).T), dtype=tf.int32)
assert opaque.shape == (4, 7)

out2 = opaque_step(opaque, opaque)
assert out2.shape == (4, 7)


# A SECOND decorated function with a DIFFERENT declared rank, also called on an unresolvable
# argument. The two declarations differ (rank 2 vs rank 3), so a recognizer that keyed on anything
# shared — the signature list, or "some input_signature in the file" — would hand one function the
# other's rank. Under per-function keying each gets its own, which rank 2 vs rank 3 makes unmissable.
other_signature = [
    tf.TensorSpec(shape=(None, None, None), dtype=tf.int32, name="Volume"),
]


def consume_other(a):
    pass


@tf.function(input_signature=other_signature)
def other_step(x):
    consume_other(x)
    return x


opaque3 = tf.constant(rng.permutation(np.ones((4, 5, 6)).T), dtype=tf.int32)
assert opaque3.shape == (6, 5, 4)

out3 = other_step(opaque3)
assert out3.shape == (6, 5, 4)
