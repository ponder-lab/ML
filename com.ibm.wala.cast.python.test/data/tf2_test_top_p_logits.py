# Minimal fixture for the input-signature-inference empirical pass
# (ponder-lab/Input-Signature-Inference-Paper#22). Mirrors `top_p_logits` from
# akanyaani/gpt-2-tensorflow2.0/sample.py — a function the previous paper's
# Hybridize tool refactored with `@tf.function`. This fixture's purpose is to
# let Ariadne's `TestTensorflow2Model` test pin the inferred parameter type
# for `logits` and observe whether ops in the body (tf.sort, tf.cumsum,
# tf.stack, tf.range, tf.gather_nd, tf.where) are load-bearing on the dtype
# axis for input-signature emission.
import tensorflow as tf


def top_p_logits(logits, p):
    """Taken from OpenAI GPT-2 implementation."""
    batch = tf.shape(logits)[0]
    sorted_logits = tf.sort(logits, direction="DESCENDING", axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack(
        [
            tf.range(0, batch),
            tf.maximum(
                tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0
            ),
        ],
        axis=-1,
    )
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


# Driver: build a known logits tensor and pass it through.
logits = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=tf.float32)
p = 0.9

# Verify Python runtime types match what we expect Ariadne to infer.
assert logits.shape == (1, 5), f"logits shape was {logits.shape}"
assert logits.dtype == tf.float32

result = top_p_logits(logits, p)
assert result.shape == (1, 5)
assert result.dtype == tf.float32
