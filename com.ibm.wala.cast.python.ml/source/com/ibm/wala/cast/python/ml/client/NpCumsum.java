package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code np.cumsum(a, axis=None, dtype=None)} (<a
 * href="https://github.com/wala/ML/issues/954">wala/ML#954</a>). The input is read as {@link
 * NpArray} reads its {@code x}, so a list of Python scalars promotes as {@code np.array} would; the
 * running sum then widens an integer input narrower than the platform integer to int64, and keeps
 * any other dtype. Without {@code axis} the result is the flattened input, rank 1 with the product
 * of the input's extents (unresolved when one is not a constant); with a constant {@code axis} it
 * keeps the input's shape. An input the analysis cannot read gives an unknown dtype and an
 * unresolved extent rather than nothing: the result is an array whatever its elements are.
 */
public class NpCumsum extends NpArray {

  private static final int DTYPE_POSITION = 2;

  public NpCumsum(PointsToSetVariable source) {
    super(source);
  }

  public NpCumsum(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> inputs = super.getDefaultShapes(builder);
    // The running sum along an axis keeps the input's shape whatever the axis is, so only the
    // presence of a non-`None` axis matters, not its value: absent or `None` flattens.
    if (this.isAxisPassed(builder)) return inputs;
    if (inputs == null || inputs.isEmpty())
      return Collections.singleton(List.of(UnresolvedDim.INSTANCE));
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> input : inputs) {
      long product = 1;
      boolean known = true;
      for (Dimension<?> d : input) {
        if (d instanceof NumericDim) product *= ((NumericDim) d).value();
        else known = false;
      }
      ret.add(
          List.of(
              known && product <= Integer.MAX_VALUE
                  ? new NumericDim((int) product)
                  : UnresolvedDim.INSTANCE));
    }
    return ret;
  }

  /**
   * The input's promoted dtype, widened to int64 for an integer narrower than the platform integer:
   * numpy accumulates such inputs in the platform integer, so {@code np.cumsum} of an int32 array
   * is int64 on a 64-bit platform, while int64 and the floating dtypes keep their own.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The result's dtypes.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> promoted = super.getDefaultDTypes(builder);
    if (promoted == null || promoted.isEmpty()) return promoted;
    EnumSet<DType> ret = EnumSet.noneOf(DType.class);
    for (DType dtype : promoted) ret.add(widenedToPlatformInteger(dtype));
    return ret;
  }

  /**
   * Whether a non-{@code None} {@code axis} is passed, positionally (a third use beyond the
   * callable and {@code a}) or by keyword. A keyword bound to a {@code None} literal counts as
   * absent, since {@code axis=None} flattens like no axis; any other value, constant or computed,
   * keeps the input's shape, so the value is never read.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @return {@code true} iff some call site passes a non-{@code None} {@code axis}.
   */
  private boolean isAxisPassed(PropagationCallGraphBuilder builder) {
    PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) return passesAxis(this.getNode(), call);
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode()))
      if (callerInvoke.snd instanceof PythonInvokeInstruction
          && passesAxis(callerInvoke.fst, (PythonInvokeInstruction) callerInvoke.snd)) return true;
    return false;
  }

  private static boolean passesAxis(CGNode node, PythonInvokeInstruction call) {
    int vn = call.getUse("axis");
    if (vn == -1 && call.getNumberOfPositionalParameters() > 2) vn = call.getUse(2);
    if (vn == -1) return false;
    SymbolTable symbols = node.getIR().getSymbolTable();
    return !(symbols.isConstant(vn) && symbols.getConstantValue(vn) == null);
  }

  /**
   * Numpy's accumulator dtype for an input dtype: int64 for an integer narrower than it, the
   * input's own dtype otherwise.
   *
   * @param dtype The input's dtype.
   * @return The accumulator's dtype.
   */
  private static DType widenedToPlatformInteger(DType dtype) {
    switch (dtype) {
      case INT32:
      case BOOL:
        return DType.INT64;
      case UINT8:
        // numpy keeps the signedness: an unsigned input narrower than the platform integer sums as
        // uint64, which `DType` has no constant for, so the honest answer is unknown rather than
        // an int64 a signature would then assert against a uint64 argument.
        return DType.UNKNOWN;
      default:
        return dtype;
    }
  }

  /**
   * No type feed: the accumulator rule maps the input's dtype rather than forwarding it, and no
   * feed kind maps a dtype, so {@link NpArray}'s dtype-only feed would deliver the input's own
   * dtype where the running sum widens it (an int32 input typed only by the dataflow would read
   * int32 where the run time is int64) and would overwrite the unknown this generator answers for
   * an unsigned narrow input. A running sum of an input the substrate cannot type therefore reads
   * an unknown dtype, the sound answer, rather than a borrowed wrong one. The precision cost is on
   * the dtypes the rule preserves (int64 and the floating dtypes): for a dataflow-only input of one
   * of those, the borrowed dtype would have been right and the answer is now unknown. A feed kind
   * carrying a rule over the dtype, as {@link TypeFeedKind#TRANSFORM} carries one over the shape,
   * would recover it; a follow-up only if a value needs it.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@code null}.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE_POSITION;
  }
}
