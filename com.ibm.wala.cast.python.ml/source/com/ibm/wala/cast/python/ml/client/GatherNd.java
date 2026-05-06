package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.gather_nd}. Output dtype is inherited from the {@code params} input.
 * Output shape is left at ⊤ for now: the precise shape is {@code indices.shape[:-1] +
 * params.shape[indices.shape[-1]:]} which requires combining two inputs' shapes — wala/ML#449 (Tier
 * 8) covers refining this once a tier base for "shape derived from multiple inputs" is in place.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/gather_nd">tf.gather_nd</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class GatherNd extends PassThroughUnaryTensorGenerator {

  public GatherNd(PointsToSetVariable source) {
    super(source);
  }

  public GatherNd(CGNode node) {
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
