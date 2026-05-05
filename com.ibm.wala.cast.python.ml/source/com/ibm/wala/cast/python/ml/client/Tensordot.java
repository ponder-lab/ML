package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.linalg.tensordot}. Output dtype is inherited from the {@code a} input.
 * Output shape is left at ⊤ for now: the precise shape depends on the {@code axes} argument
 * together with both inputs' shapes (axes can be a scalar, a 1-D list, or a 2-D list), which is
 * non-trivial and not yet covered by the tier framework. See wala/ML#449.
 *
 * <p>Extends {@link PassThroughUnaryTensorGenerator} for the dtype-from-input path only — the
 * shape-passthrough behavior is overridden to ⊤. The base class documents itself as "shape and
 * dtype passthrough"; using it here for *just* the dtype path is an intentional partial reuse,
 * acknowledged because extracting a separate dtype-only base would split a small piece of code
 * across two classes for a single-method gain. If more dtype-only-passthrough ops accumulate, a
 * shared {@code DTypePassThroughUnaryTensorGenerator} base is the natural refactor — tracked as
 * future work.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/linalg/tensordot">tf.linalg.tensordot</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Tensordot extends PassThroughUnaryTensorGenerator {

  public Tensordot(PointsToSetVariable source) {
    super(source);
  }

  public Tensordot(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "a";
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
  }
}
