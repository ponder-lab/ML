# wala/ML#481: `tf.cast(x, y.dtype)` with a bool input. The cast target is another value's
# `.dtype` attribute; the result must carry THAT dtype (or honestly unknown), never the bool
# input's, or the downstream subtraction imposes bool onto its float partner and the wrong
# answer becomes self-consistent through the very attribute being read.
import tensorflow as tf

MIN_FLOAT = -1e9


def remove_accidental(logits, labels, identifiers):
    identifiers = tf.expand_dims(identifiers, 1)
    positive_indices = tf.math.argmax(labels, axis=1)
    positive_identifier = tf.gather(identifiers, positive_indices)
    duplicate = tf.equal(positive_identifier, tf.transpose(identifiers))
    duplicate = tf.cast(duplicate, labels.dtype)
    consume_cast(duplicate)
    duplicate = duplicate - labels
    return logits + duplicate * MIN_FLOAT


def consume_cast(t):
    assert t.dtype == tf.float32
    assert t.shape == (4, 4)


def consume(t):
    assert t.dtype == tf.float32
    assert t.shape == (4, 4)


def consume_int_target(t):
    # A SECOND `.dtype`-target cast whose source dtype differs from the first one's. Both casts
    # reach the same summary, so this pins that each recovers ITS OWN target rather than the
    # union of both (wala/ML#481).
    assert t.dtype == tf.int64
    assert t.shape == (4, 4)


def int_target(values, ints):
    duplicate = tf.equal(values, tf.transpose(values))
    duplicate = tf.cast(duplicate, ints.dtype)
    consume_int_target(duplicate)
    return duplicate


def consume_unresolved_target(t):
    # The cast target does not resolve statically, but it is certainly NOT the bool input's
    # dtype: the result must read an unknown dtype rather than the input pass-through, or the
    # subtraction below would impose bool on its float partner (wala/ML#481).
    assert t.dtype == tf.float32
    assert t.shape == (4, 4)


def unresolved_target(logits, labels, dtype_name):
    duplicate = tf.equal(labels, tf.transpose(labels))
    # A dynamic attribute lookup: the target is a real dtype at run time and unresolvable
    # statically, which is the case the honest unknown terminal exists for.
    duplicate = tf.cast(duplicate, getattr(tf, dtype_name))
    consume_unresolved_target(duplicate)
    return logits + duplicate


logits = tf.zeros((4, 4))
labels = tf.one_hot([0, 1, 2, 3], 4)
identifiers = tf.constant([7, 8, 9, 7], dtype=tf.int64)
out = remove_accidental(logits, labels, identifiers)
assert out.dtype == tf.float32
assert out.shape == (4, 4)
consume(out)

unresolved = unresolved_target(logits, labels, "float32")

ints = tf.constant([[1, 2, 3, 4]], dtype=tf.int64)
int_out = int_target(labels, ints)
assert int_out.dtype == tf.int64
assert unresolved.dtype == tf.float32
assert unresolved.shape == (4, 4)
