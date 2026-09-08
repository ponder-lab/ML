# Diagnostic fixture (wala/ML#875): a local name bound twice, where the second binding is a
# `tf.reshape` carrying a `-1`, and the rebound name is then passed as an argument.
#
# The question this file is built to answer is how many members reach `consume`'s parameter:
#   - one member, the reshape result, if the argument feed reads only the live binding; or
#   - two members, the reshape result *and* the overwritten `Dense` result, if the feed unions
#     every binding of the name.
#
# It also separates that question from a second one. The `-1` here is fully determined --
# 8 * 10 * 46 divided by 8 * 10 is exactly 46 -- so a reshape that can read its input shape must
# fold it to a numeric extent. A `?` on the trailing axis therefore says the fold did not see the
# input, independently of how many members arrive.
import tensorflow as tf


def consume(t):
    pass


class Rebind(tf.keras.Model):
    def __init__(self):
        super(Rebind, self).__init__()
        self.batch_size = 8
        self.maxlen = 10
        self.label_size = 46
        self.dense = tf.keras.layers.Dense(self.label_size)

    def call(self, inputs):
        predict = self.dense(inputs)
        predict = tf.reshape(predict, [self.batch_size, self.maxlen, -1])
        # Runtime truth: both bindings hold the same shape here, and only the second one is live
        # at the call below.
        assert predict.shape == (8, 10, 46)
        assert predict.dtype == tf.float32
        consume(predict)
        return predict


model = Rebind()
model.build(input_shape=(8, 10, 768))
model(tf.ones((8, 10, 768)))
