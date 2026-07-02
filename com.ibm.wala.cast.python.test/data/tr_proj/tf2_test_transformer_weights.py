# Regression guard for wala/ML#669. The model topology is adapted from
# `LongmaoTeamTf/deep_recommenders` (`examples/train_transformer_on_imdb_keras.py`, `build_model`
# inlined at module level): a functional `tf.keras.Model` whose weight graph walks back through
# `Dense` and the transformer's `K.dot` kernels. Resolving those weight shapes
# (`Model.getWeightShapes`) meets a WALA 1.8.0 `ScopeMappingInstanceKey` in the `units` slot,
# which crashed `getConstantValues` with `IllegalStateException` on 0.52.12; it must instead
# degrade to "not statically resolvable".
import tensorflow as tf

from deep_recommenders.keras.models.nlp import Transformer

vocab_size = 5000
max_len = 128
model_dim = 8


def consume(t):
    pass


encoder_inputs = tf.keras.Input(shape=(max_len,), name="encoder_inputs")
decoder_inputs = tf.keras.Input(shape=(max_len,), name="decoder_inputs")
outputs = Transformer(
    vocab_size,
    model_dim,
    n_heads=2,
    encoder_stack=2,
    decoder_stack=2,
    feed_forward_size=50,
)(encoder_inputs, decoder_inputs)
outputs = tf.keras.layers.GlobalAveragePooling1D()(outputs)
outputs = tf.keras.layers.Dense(2, activation="softmax")(outputs)
model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)

for w in model.weights:
    consume(w)

# The classifier head's kernel: the transformer outputs vocab-size logits, so after pooling the
# head consumes a 5000-dim feature.
assert any(tuple(w.shape) == (5000, 2) for w in model.weights)
assert all(w.dtype == tf.float32 for w in model.weights)
