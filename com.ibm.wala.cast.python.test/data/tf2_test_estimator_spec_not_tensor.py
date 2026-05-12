import tensorflow as tf


def f(a):
    pass


loss_tensor = tf.constant(1.0)
assert isinstance(loss_tensor, tf.Tensor)

spec = tf.estimator.EstimatorSpec(
    mode=tf.estimator.ModeKeys.EVAL,
    predictions=None,
    loss=loss_tensor,
    train_op=None,
    eval_metric_ops={},
    export_outputs=None,
)

# spec is a namedtuple-like object, NOT a tensor. The static analyzer must
# allocate it as `Ltensorflow/estimator/EstimatorSpec`, not as a tensor class
# (wala/ML#523). If misclassified as Tensor, `f` would have 1 tensor parameter
# at value number 2; with the correct allocation class, `f` has 0 tensor
# parameters.
assert not isinstance(spec, tf.Tensor)
f(spec)
