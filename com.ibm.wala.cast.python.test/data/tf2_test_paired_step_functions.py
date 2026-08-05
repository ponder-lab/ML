import tensorflow as tf


def consume_train(a, b):
    pass


def consume_test(a, b):
    pass


# `gpt-2-tensorflow2.0`'s `Gpt2.fit` in miniature. Two step methods are handed out as a
# tuple of function values, destructured by the caller, and invoked indirectly. The
# arguments come from two datasets, and the second dataset is field 1 of a destructuring
# whose field 0 rebinds the parameter it is destructuring.
class Stepper:
    def _train_step(self, inputs, targets):
        consume_train(inputs, targets)
        return inputs

    def _test_step(self, inputs, targets):
        consume_test(inputs, targets)
        return targets

    def get_train_test_function(self):
        train_fuc = self._train_step
        test_fuc = self._test_step
        return train_fuc, test_fuc

    def fit(self, train_dataset):
        # The rebinding destructure: the parameter becomes its own field 0, and the second
        # dataset is field 1 of the same tuple.
        train_dataset, test_dataset = train_dataset
        train_func, test_func = self.get_train_test_function()

        for _, (inputs, targets) in enumerate(train_dataset):
            assert inputs.shape == (8, 207) and inputs.dtype == tf.int32
            assert targets.shape == (8, 207) and targets.dtype == tf.int32
            train_func(inputs, targets)

            for _, (test_inputs, test_targets) in enumerate(test_dataset):
                assert test_inputs.shape == (8, 207) and test_inputs.dtype == tf.int32
                assert test_targets.shape == (8, 207) and test_targets.dtype == tf.int32
                test_func(test_inputs, test_targets)


def make_dataset():
    features = tf.ones((8, 207), dtype=tf.int32)
    labels = tf.ones((8, 207), dtype=tf.int32)
    return tf.data.Dataset.from_tensor_slices((features, labels)).batch(8)


stepper = Stepper()
stepper.fit((make_dataset(), make_dataset()))
