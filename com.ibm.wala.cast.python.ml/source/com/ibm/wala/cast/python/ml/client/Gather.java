package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

/**
 * Generator for {@code tf.gather} at its default axis. Output dtype is inherited from the {@code
 * params} argument, and output shape is {@code indices.shape + params.shape[1:]}: each index
 * selects a full slice along the first axis, so the leading {@code indices.shape} indexes the
 * result and the trailing {@code params.shape[1:]} is the selected slice.
 *
 * <p>That is the same computation {@link EmbeddingLookup} performs, because {@code
 * tf.nn.embedding_lookup} is a gather along the first axis; this subclass exists to give the two
 * APIs distinct dispatch while sharing one rule rather than two that can drift apart.
 *
 * <p>Modeled previously as a pass-through, which returned the table's own type. A subject reading
 * {@code K.gather(self.embeddings, inputs)} therefore typed the result as the {@code (vocab,
 * model_dim)} table rather than the {@code (batch, sequence, model_dim)} lookup, which is a rank
 * error rather than an imprecision, and a signature written from it rejects every call. See
 * wala/ML#815.
 *
 * <p>Scope: the default first-axis gather. A non-default {@code axis} or a non-zero {@code
 * batch_dims} rearranges the result differently, and is left to the inherited behavior rather than
 * answered wrongly.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/gather">tf.gather</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Gather extends EmbeddingLookup {

  public Gather(PointsToSetVariable source) {
    super(source);
  }

  public Gather(CGNode node) {
    super(node);
  }
}
