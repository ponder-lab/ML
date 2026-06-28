import tensorflow as tf


class Model(tf.keras.Model):
    def get_loss(self, real, pred):
        return tf.reduce_mean(tf.square(pred - real))

    def train_step(self, inputs, targets):
        # `inputs`/`targets` are `padded_batch` elements threaded through `fit`. See wala/ML#623.
        assert inputs.shape == (2, 2)
        assert inputs.dtype == tf.int32
        assert targets.shape == (2, 2)
        assert targets.dtype == tf.int32
        return self.get_loss(targets, inputs)

    def fit(self, dataset):
        for inputs, targets in dataset:
            self.train_step(inputs, targets)


ds = tf.data.Dataset.from_tensor_slices(
    (
        tf.constant([[1, 2], [3, 4]], dtype=tf.int32),
        tf.constant([[1, 1], [1, 1]], dtype=tf.int32),
    )
).padded_batch(2)
Model().fit(ds)
