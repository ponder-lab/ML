# Fixture mirroring the linear-chain CRF functions from
# `kyzhouhzau/NLPGNN/nlpgnn/metrics/crf.py` (`crf_sequence_score`,
# `crf_unary_score`, `crf_binary_score`, `crf_log_norm`, `crf_forward`,
# `crf_decode_forward`). These real-world functions exercise CRF-specific
# tensor ops (`tf.gather`, `tf.sequence_mask`, `tf.scan`, `tf.slice`,
# `tf.reshape` with runtime-derived dimensions, and a custom RNN cell). The
# fixture lets `TestTensorflow2Model` pin their parameter types and exercise
# tensor-type inference on this op mix; see wala/ML#567, wala/ML#568, and
# wala/ML#406 for gaps it surfaced.
import tensorflow as tf


def crf_sequence_score(inputs, tag_indices, sequence_lengths, transition_params):
    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
    unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
    binary_scores = crf_binary_score(tag_indices, sequence_lengths, transition_params)
    sequence_scores = unary_scores + binary_scores
    return sequence_scores


def crf_unary_score(tag_indices, sequence_lengths, inputs):
    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    batch_size = tf.shape(inputs)[0]
    max_seq_len = tf.shape(inputs)[1]
    num_tags = tf.shape(inputs)[2]

    flattened_inputs = tf.reshape(inputs, [-1])

    offsets = tf.expand_dims(tf.range(batch_size) * max_seq_len * num_tags, 1)
    offsets += tf.expand_dims(tf.range(max_seq_len) * num_tags, 0)
    if tag_indices.dtype == tf.int64:
        offsets = tf.cast(offsets, tf.int64)
    flattened_tag_indices = tf.reshape(offsets + tag_indices, [-1])

    unary_scores = tf.reshape(
        tf.gather(flattened_inputs, flattened_tag_indices),
        [batch_size, max_seq_len],
    )

    masks = tf.sequence_mask(
        sequence_lengths, maxlen=tf.shape(tag_indices)[1], dtype=tf.float32
    )

    unary_scores = tf.reduce_sum(unary_scores * masks, 1)
    return unary_scores


def crf_binary_score(tag_indices, sequence_lengths, transition_params):
    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    num_tags = tf.shape(transition_params)[0]
    num_transitions = tf.shape(tag_indices)[1] - 1

    start_tag_indices = tf.slice(tag_indices, [0, 0], [-1, num_transitions])
    end_tag_indices = tf.slice(tag_indices, [0, 1], [-1, num_transitions])

    flattened_transition_indices = start_tag_indices * num_tags + end_tag_indices
    flattened_transition_params = tf.reshape(transition_params, [-1])

    binary_scores = tf.gather(flattened_transition_params, flattened_transition_indices)
    masks = tf.sequence_mask(
        sequence_lengths, maxlen=tf.shape(tag_indices)[1], dtype=tf.float32
    )
    truncated_masks = tf.slice(masks, [0, 1], [-1, -1])
    binary_scores = tf.reduce_sum(binary_scores * truncated_masks, 1)
    return binary_scores


def crf_log_norm(inputs, sequence_lengths, transition_params):
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
    first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
    first_input = tf.squeeze(first_input, [1])
    rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])

    alphas = crf_forward(
        rest_of_input, first_input, transition_params, sequence_lengths
    )
    log_norm = tf.reduce_logsumexp(alphas, [1])

    return log_norm


def crf_forward(inputs, state, transition_params, sequence_lengths):
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    last_index = tf.maximum(
        tf.constant(0, dtype=sequence_lengths.dtype), sequence_lengths - 1
    )
    inputs = tf.transpose(inputs, [1, 0, 2])
    transition_params = tf.expand_dims(transition_params, 0)

    def _scan_fn(_state, _inputs):
        _state = tf.expand_dims(_state, 2)
        transition_scores = _state + transition_params
        new_alphas = _inputs + tf.reduce_logsumexp(transition_scores, [1])
        return new_alphas

    all_alphas = tf.transpose(tf.scan(_scan_fn, inputs, state), [1, 0, 2])
    all_alphas = tf.concat([tf.expand_dims(state, 1), all_alphas], 1)

    idxs = tf.stack([tf.range(tf.shape(last_index)[0]), last_index], axis=1)
    return tf.gather_nd(all_alphas, idxs)


class CrfDecodeForwardRnnCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, transition_params, **kwargs):
        super(CrfDecodeForwardRnnCell, self).__init__(**kwargs)
        self._transition_params = tf.expand_dims(transition_params, 0)
        self._num_tags = transition_params.shape[0]

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def build(self, input_shape):
        super(CrfDecodeForwardRnnCell, self).build(input_shape)

    def call(self, inputs, state):
        state = tf.expand_dims(state[0], 2)
        transition_scores = state + self._transition_params
        new_state = inputs + tf.reduce_max(transition_scores, [1])
        backpointers = tf.argmax(transition_scores, 1)
        backpointers = tf.cast(backpointers, dtype=tf.int32)
        return backpointers, new_state


def crf_decode_forward(inputs, state, transition_params, sequence_lengths):
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
    mask = tf.sequence_mask(sequence_lengths, tf.shape(inputs)[1])
    crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params)
    crf_fwd_layer = tf.keras.layers.RNN(
        crf_fwd_cell, return_sequences=True, return_state=True
    )
    return crf_fwd_layer(inputs, state, mask=mask)


# Driver: concrete linear-chain CRF inputs (batch=2, seq_len=3, num_tags=4).
inputs = tf.constant(
    [
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
        [[1.3, 1.4, 1.5, 1.6], [1.7, 1.8, 1.9, 2.0], [2.1, 2.2, 2.3, 2.4]],
    ],
    dtype=tf.float32,
)
tag_indices = tf.constant([[0, 1, 2], [1, 2, 3]], dtype=tf.int32)
sequence_lengths = tf.constant([3, 2], dtype=tf.int32)
transition_params = tf.constant(
    [
        [0.0, 0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6, 0.7],
        [0.8, 0.9, 1.0, 1.1],
        [1.2, 1.3, 1.4, 1.5],
    ],
    dtype=tf.float32,
)

assert inputs.shape == (2, 3, 4) and inputs.dtype == tf.float32
assert tag_indices.shape == (2, 3) and tag_indices.dtype == tf.int32
assert sequence_lengths.shape == (2,) and sequence_lengths.dtype == tf.int32
assert transition_params.shape == (4, 4) and transition_params.dtype == tf.float32

unary = crf_unary_score(tag_indices, sequence_lengths, inputs)
assert unary.shape == (2,) and unary.dtype == tf.float32

binary = crf_binary_score(tag_indices, sequence_lengths, transition_params)
assert binary.shape == (2,) and binary.dtype == tf.float32

seq_score = crf_sequence_score(inputs, tag_indices, sequence_lengths, transition_params)
assert seq_score.shape == (2,) and seq_score.dtype == tf.float32

log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)
assert log_norm.shape == (2,) and log_norm.dtype == tf.float32

# `crf_forward` is exercised through `crf_log_norm` above (its only caller in
# NLPGNN), so it is not invoked directly here. `crf_decode_forward` is invoked
# directly, mirroring its use from `CrfLogLikelihood.crf_decode`.
first_input = inputs[:, 0, :]
rest_of_input = inputs[:, 1:, :]
dec_fwd, _state = crf_decode_forward(
    rest_of_input, first_input, transition_params, sequence_lengths
)
assert dec_fwd.dtype == tf.int32
