package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.slice(input_, begin, size, name=None)}. Output dtype is inherited from
 * the {@code input_} input. Output shape is left at ⊤ for now: the precise shape is determined by
 * the {@code begin}/{@code size} extents (where a {@code size[i]} of {@code -1} means "all
 * remaining elements along axis {@code i}"), tracked by <a
 * href="https://github.com/wala/ML/issues/569">wala/ML#569</a>. Forwarding only the dtype already
 * recovers it from ⊤ — see <a href="https://github.com/wala/ML/issues/568">wala/ML#568</a>.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/slice">tf.slice</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Slice extends PassThroughUnaryTensorGenerator {

  public Slice(PointsToSetVariable source) {
    super(source);
  }

  public Slice(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "input_";
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
  }
}
