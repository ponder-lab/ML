# Driver for `TextCNN.predict` from `kyzhouhzau/NLPGNN/nlpgnn/models/TextCNN.py` (the
# wala/ML#618 row): `inputs` receives an integer token-ID tensor, and `predict` forwards it to
# the model through `self(inputs, training)`. The `TextCNN` model is vendored verbatim from
# upstream; only this driver is bespoke.
import numpy as np
import tensorflow as tf

from nlpgnn.models.TextCNN import TextCNN

maxlen = 5
vocab_size = 10
embedding_dims = 8
class_num = 3
batch = 2

inputs = tf.constant(np.ones((batch, maxlen), dtype=np.int32))
model = TextCNN(
    maxlen=maxlen,
    vocab_size=vocab_size,
    embedding_dims=embedding_dims,
    class_num=class_num,
)
result = model.predict(inputs, training=False)

assert inputs.shape == (batch, maxlen)
assert inputs.dtype == tf.int32
assert result.shape == (batch, class_num)
assert result.dtype == tf.float32
