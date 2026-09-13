package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.Set;

/**
 * Generator for {@code np.random.randint(low, high, size, dtype)} (wala/ML#909): the {@link
 * NpRandomSizedDraw} signature shape, two distribution parameters then {@code size}, so the shape
 * is the parent's and only the dtype differs. NumPy's default integer dtype is {@code int64}; a
 * supplied {@code dtype} the analysis cannot read degrades to unknown rather than the default.
 *
 * @see <a
 *     href="https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html">numpy.random.randint</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NpRandomIntegerDraw extends NpRandomSizedDraw {

  /**
   * The 0-based positional index of {@code dtype}, after {@code low}, {@code high} and {@code
   * size}.
   */
  private static final int DTYPE_POSITION = 3;

  public NpRandomIntegerDraw(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public NpRandomIntegerDraw(CGNode node) {
    super(node);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return this.dTypeApiDefaultOrUnknown(builder, EnumSet.of(DType.INT64));
  }

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return "dtype";
  }
}
