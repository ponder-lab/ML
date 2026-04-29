import tensorflow as tf


def f(a):
    pass


# Real-world usage: model_fn returns an EstimatorSpec carrying loss/train_op/etc.
loss_tensor = tf.constant(1.0)

spec = tf.estimator.EstimatorSpec(
    mode="train",
    predictions=None,
    loss=loss_tensor,
    train_op=None,
    eval_metric_ops={},
    export_outputs=None,
)

# spec.loss must resolve to the original loss_tensor allocation
# (a scalar float32). If EstimatorSpec is mis-modeled as "return one of the
# inputs" instead of "fresh allocation with field sets," this read won't
# round-trip correctly.
read_loss = spec.loss
f(read_loss)
