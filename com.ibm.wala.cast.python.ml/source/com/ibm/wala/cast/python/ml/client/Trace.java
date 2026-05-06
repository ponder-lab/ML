package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.linalg.trace}. Output dtype is inherited from the {@code x} input. Output
 * shape is the leading {@code x.shape[:-2]} (the trace collapses the last two dimensions to a
 * scalar per matrix in the batch); leaving the shape at ⊤ for now since deriving "input shape minus
 * the last two dimensions" needs a small dim-list slice that isn't shared by the existing tier
 * bases. See wala/ML#449.
 *
 * <p>Extends {@link PassThroughUnaryTensorGenerator} for the dtype-from-input path only — see
 * {@link Tensordot}'s class-level Javadoc for the rationale (and the future-refactor note about
 * extracting a shared {@code DTypePassThroughUnaryTensorGenerator} base).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/linalg/trace">tf.linalg.trace</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Trace extends PassThroughUnaryTensorGenerator {

  public Trace(PointsToSetVariable source) {
    super(source);
  }

  public Trace(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "x";
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
  }
}
