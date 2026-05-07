package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.nn.embedding_lookup}. Output dtype is inherited from the {@code params}
 * argument (the embedding table). Output shape is left at ⊤ for now: the precise shape is {@code
 * ids.shape + params.shape[1:]} (each id selects a full row of the embedding table, then the result
 * is reshaped around {@code ids.shape}), which requires combining two inputs' shapes — wala/ML#449
 * (Tier 8) covers refining this once a tier base for "shape derived from multiple inputs" is in
 * place.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup">tf.nn.embedding_lookup</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class EmbeddingLookup extends PassThroughUnaryTensorGenerator {

  public EmbeddingLookup(PointsToSetVariable source) {
    super(source);
  }

  public EmbeddingLookup(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "params";
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
  }
}
