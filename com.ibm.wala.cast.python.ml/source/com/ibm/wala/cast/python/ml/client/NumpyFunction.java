package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Generator for {@code tf.numpy_function(func, inp, Tout, stateful=True, name=None)}. The result's
 * dtype is the one the {@code Tout} argument states; its shape is whatever {@code func} returns at
 * runtime, which the call does not state, so it is unknown.
 *
 * @see <a href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/numpy_function">
 *     tf.numpy_function</a>
 */
public class NumpyFunction extends TensorTypeAllocator {

  /**
   * Parameter positions and keyword names for {@code tf.numpy_function}. Ordinals match the
   * position in the XML's {@code paramNames} after the implicit {@code self} receiver.
   */
  protected enum Parameters {
    /** The Python function the op runs. */
    FUNC,

    /** The function's inputs. */
    INP,

    /** The result's dtype. */
    TOUT,

    /** Whether the function is stateful. */
    STATEFUL,

    /** Optional debug name for the op. */
    NAME;

    /**
     * The keyword name used when the call site passes the argument by keyword.
     *
     * @return The keyword name.
     */
    public String getName() {
      return this == TOUT ? "Tout" : name().toLowerCase(Locale.ROOT);
    }

    /**
     * The positional index of this parameter, excluding the implicit {@code self} receiver.
     *
     * @return The zero-based positional index.
     */
    public int getIndex() {
      return ordinal();
    }
  }

  public NumpyFunction(PointsToSetVariable source) {
    super(source);
  }

  public NumpyFunction(CGNode node) {
    super(node);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Unknown: the call does not state the shape {@code func} returns.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Unknown, when {@code Tout} does not resolve to a dtype.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.UNKNOWN);
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.TOUT.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.TOUT.getName();
  }
}
