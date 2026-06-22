import tensorflow as tf
import numpy as np

from nlpgnn.models.TextCNN import TextCNN

# Driver for `TextCNN.call` from `kyzhouhzau/NLPGNN/nlpgnn/models/TextCNN.py`, a
# real-world text-classification utility (a convolutional sentence encoder:
# embedding lookup, parallel Conv1D kernels, global average pooling,
# concatenation, batch normalization, and a softmax dense head), for tensor-type
# inference coverage. The `TextCNN` model is vendored verbatim from upstream;
# only this driver is bespoke. Unlike `gcn_proj`/`gat_proj`, the decorated
# parameter `inputs` is an integer token-ID tensor (an embedding-lookup index),
# so this exercises int-dtype parameter recovery.
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
result = model(inputs, training=False)

assert inputs.shape == (batch, maxlen)
assert inputs.dtype == tf.int32
assert result.shape == (batch, class_num)
assert result.dtype == tf.float32
