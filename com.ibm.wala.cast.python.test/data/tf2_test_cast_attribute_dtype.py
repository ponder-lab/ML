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
    pass


logits = tf.zeros((4, 4))
labels = tf.one_hot([0, 1, 2, 3], 4)
identifiers = tf.constant([7, 8, 9, 7], dtype=tf.int64)
out = remove_accidental(logits, labels, identifiers)
assert out.dtype == tf.float32
assert out.shape == (4, 4)
consume(out)
