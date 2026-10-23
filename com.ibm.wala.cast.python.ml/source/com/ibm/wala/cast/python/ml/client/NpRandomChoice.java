package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.Set;

/**
 * Generator for {@code np.random.choice(a, size=None, replace=True, p=None)} (<a
 * href="https://github.com/wala/ML/issues/1008">wala/ML#1008</a>). The shape is the {@code size}
 * argument, as for the other sized draws, and a draw without one is a scalar. The elements are
 * drawn from {@code a}, or from {@code np.arange(a)} when {@code a} is an integer, so the dtype is
 * {@code int64} for an integer {@code a} and an array's own dtype otherwise; a scalar draw from an
 * integer {@code a} is a Python {@code int}, encoded as {@code int32} (see {@link NpRandomDraw}). A
 * Python sequence, whose dtype NumPy promotes over its elements, reads as unknown.
 *
 * @see <a
 *     href="https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html">numpy.random.choice</a>
 */
public class NpRandomChoice extends NpRandomSizedDraw {

  /** The positional index of {@code size}, after {@code a}. */
  private static final int SIZE_POSITION = 1;

  public NpRandomChoice(PointsToSetVariable source) {
    super(source);
  }

  public NpRandomChoice(CGNode node) {
    super(node);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> a = this.getArgumentPointsToSet(builder, 0, "a");
    if (a == null || a.isEmpty()) return EnumSet.of(DType.UNKNOWN);
    boolean integer = false;
    boolean array = false;
    for (InstanceKey ik : a)
      if (ik instanceof ConstantKey<?> constant) {
        if (!(constant.getValue() instanceof Integer || constant.getValue() instanceof Long))
          return EnumSet.of(DType.UNKNOWN);
        integer = true;
      } else {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin == null
            || asin.concreteType().getReference().equals(PythonTypes.list)
            || asin.concreteType().getReference().equals(PythonTypes.tuple))
          return EnumSet.of(DType.UNKNOWN);
        array = true;
      }
    // An integer beside an array would need the array members' dtypes apart from the integer's,
    // which the value read does not separate.
    if (integer && array) return EnumSet.of(DType.UNKNOWN);
    // Without a `size`, an integer `a` draws a Python `int`, which this analysis encodes as
    // `int32`,
    // as `NpRandomDraw` encodes a Python `float` as `float32`.
    if (integer) return EnumSet.of(this.isScalarDraw(builder) ? DType.INT32 : DType.INT64);
    Set<DType> elements = this.getDTypesOfValue(builder, a);
    return elements == null || elements.isEmpty() ? EnumSet.of(DType.UNKNOWN) : elements;
  }

  @Override
  protected int getShapeParameterPosition() {
    return SIZE_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return "size";
  }
}
