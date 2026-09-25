import tensorflow as tf


def consume_none(l):
    assert l.shape == (2, 3) and l.dtype == tf.float32


def consume_scalar(l):
    assert l.shape == () and l.dtype == tf.float32


def consume_perplexity(p):
    assert p.shape == () and p.dtype == tf.float32


def consume_fed(l):
    assert l.shape == (2, 3) and l.dtype == tf.float32


class Head(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.w = self.add_weight("w", shape=[4, 10], dtype=tf.float32)
        super(Head, self).build(input_shape)

    def call(self, x):
        return tf.matmul(x, self.w)


logits = tf.ones((2, 3, 10))
labels = tf.zeros((2, 3), dtype=tf.int32)

per_token = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)
loss_ = per_token(labels, logits)
consume_none(loss_)

mean_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
consume_scalar(mean_loss(labels, logits))

consume_perplexity(tf.exp(tf.reduce_mean(loss_)))

head = Head()
consume_fed(per_token(labels, head(tf.ones((2, 3, 4)))))


def consume_sum(l):
    assert l.shape == () and l.dtype == tf.float32


summed = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="sum"
)
consume_sum(summed(labels, logits))


def consume_unresolved(l):
    # At run time the environment is unset, so the reduction is "none" and the loss is per-token;
    # the analysis cannot read the environment, so it must not assert either shape.
    assert l.shape == (2, 3) and l.dtype == tf.float32


import os

mode = os.environ.get("LOSS_REDUCTION", "none")
configured = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction=mode
)
consume_unresolved(configured(labels, logits))


def consume_two_instances(l):
    # Two instances with different reductions reach one call; the shape is one of two.
    assert l.shape == (2, 3) and l.dtype == tf.float32


def consume_two_literals(l):
    # One instance whose reduction is one of two literals; the shape is one of two.
    assert l.shape == (2, 3) and l.dtype == tf.float32


flag = len(os.environ.get("LOSS_FLAG", "")) == 0
chosen = per_token if flag else mean_loss
consume_two_instances(chosen(labels, logits))

either = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none" if flag else "sum"
)
consume_two_literals(either(labels, logits))
