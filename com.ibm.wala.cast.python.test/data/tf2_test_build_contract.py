import tensorflow as tf


def consume(t):
    pass


def get_shape_list(tensor):
    return tensor.shape.as_list()


def opaque(t):
    return tf.reshape(t, get_shape_list(t)[0 : t.shape.ndims])


# NLPGNN's ALBERT entry idiom (wala/ML#717): the model's input contract is
# declared explicitly and statically via `model.build`, while the runtime
# inputs come from an opaque pipeline; the declared contract seeds the `call`
# input, and the split/squeeze/cast chain carries it to the pieces.
class StackedInput(tf.keras.Model):
    def call(self, inputs, training=None):
        input_ids, token_type_ids, input_mask = tf.split(inputs, 3, 0)
        input_ids = tf.cast(tf.squeeze(input_ids, axis=0), tf.int32)
        return input_ids


batch_size = 8
maxlen = 100

model = StackedInput()
model.build(input_shape=(3, batch_size, maxlen))

X = opaque(tf.ones((batch_size, maxlen)))
token_type_id = opaque(tf.zeros((batch_size, maxlen)))
input_mask = opaque(tf.ones((batch_size, maxlen)))

predict = model([X, token_type_id, input_mask])

assert predict.shape == (batch_size, maxlen)
assert predict.dtype == tf.int32

consume(predict)
