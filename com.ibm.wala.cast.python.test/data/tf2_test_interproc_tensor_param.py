import tensorflow as tf


class Model:
    def call(self, x):
        return x * 2.0

    def get_loss(self, real, pred):
        # Both parameters receive tensors at the call site in `train_step`.
        assert real.shape == (3,)
        assert real.dtype == tf.float32
        assert pred.shape == (3,)
        assert pred.dtype == tf.float32
        return tf.reduce_mean(tf.square(pred - real))

    def train_step(self, inputs, targets):
        predictions = self.call(inputs)
        return self.get_loss(targets, predictions)


Model().train_step(tf.constant([1.0, 2.0, 3.0]), tf.constant([1.0, 1.0, 1.0]))
