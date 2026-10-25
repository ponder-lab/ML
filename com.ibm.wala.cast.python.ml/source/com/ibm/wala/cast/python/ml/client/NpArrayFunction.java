package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.withoutClientOnlyKeys;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Shared base for the NumPy functions that read their array-like arguments as {@code np.array}
 * reads its {@code x} and compute a fresh array from them (<a
 * href="https://github.com/wala/ML/issues/1008">wala/ML#1008</a>). It supplies the reads the
 * subclasses share: an argument's NumPy dtypes promoted across several arguments, an array's
 * flattened length, and whether, and to which constant, an optional argument is passed.
 */
abstract class NpArrayFunction extends NpArray {

  NpArrayFunction(PointsToSetVariable source) {
    super(source);
  }

  NpArrayFunction(CGNode node) {
    super(node);
  }

  /**
   * The positions of the arguments whose dtypes the result's dtype is computed from: promoted
   * together, then mapped by {@link #mapDType}.
   *
   * @return The positions, excluding the callable; argument {@code 0} alone by default.
   */
  protected int[] getDTypePositions() {
    return new int[] {0};
  }

  /**
   * The result's dtype for the promoted dtype of its {@linkplain #getDTypePositions() dtype
   * arguments}.
   *
   * @param promoted The arguments' promoted dtype.
   * @return The result's dtype; the promoted one itself by default.
   */
  protected DType mapDType(DType promoted) {
    return promoted;
  }

  /**
   * Declares a {@link TypeFeedKind#DTYPE_ONLY} feed from the one {@linkplain #getDTypePositions()
   * dtype argument} the substrate cannot type, so an argument typed only by the dataflow (a sidecar
   * annotation, a value loaded from a file) still types the result. The fed dtype is not borrowed
   * as it is: it goes through the result's own rule, promoted against the other dtype arguments,
   * whose dtypes were read, and then {@linkplain #mapDType mapped} (wala/ML#1008). With more than
   * one argument the substrate cannot type, or one it types with several dtypes, the promotion has
   * no single partner, and no feed is declared.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The feed, or {@code null}.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    int unread = -1;
    DType others = null;
    for (int position : this.getDTypePositions()) {
      Set<DType> dtypes = this.getNumpyDTypesOfArgument(builder, position);
      if (dtypes.contains(DType.UNKNOWN)) {
        if (unread >= 0) return null;
        unread = position;
      } else if (dtypes.size() != 1) return null;
      else {
        DType dtype = dtypes.iterator().next();
        others = others == null ? dtype : NpConcatenate.promote(others, dtype);
      }
    }
    if (unread < 0) return null;
    int vn = this.getArgumentValueNumber(unread);
    if (vn <= 0) return null;
    PointerKey argument =
        builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(this.getNode(), vn);
    List<PointerKey> operands = new ArrayList<>();
    operands.add(argument);
    addContainerElementOperands(builder, argument, operands);
    DType partner = others;
    return new TypeFeed(
        operands,
        dtype -> this.mapDType(partner == null ? dtype : NpConcatenate.promote(dtype, partner)));
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }

  /**
   * {@inheritDoc}
   *
   * <p>{@code np.array}'s read takes an existing array's dtype only when it is one dtype, since
   * {@code np.array} cannot produce a union. These functions keep each member's dtype, so an array
   * argument that may be either of two dtypes (an {@code np.arange} whose bounds may be integers or
   * floats) reads as both rather than as unknown. A Python scalar or sequence argument is left to
   * {@code np.array}'s read, which alone converts it as NumPy does.
   */
  @Override
  protected Set<DType> getNumpyDTypesOfArgument(PropagationCallGraphBuilder builder, int position) {
    Set<DType> ret = super.getNumpyDTypesOfArgument(builder, position);
    if (!ret.contains(DType.UNKNOWN)) return ret;
    int vn = this.getArgumentValueNumber(position);
    if (vn <= 0) return ret;
    OrdinalSet<InstanceKey> pts =
        withoutClientOnlyKeys(
            builder
                .getPointerAnalysis()
                .getPointsToSet(
                    builder
                        .getPointerAnalysis()
                        .getHeapModel()
                        .getPointerKeyForLocal(this.getNode(), vn)));
    if (pts != null)
      for (InstanceKey ik : pts) {
        if (ik instanceof ConstantKey) return ret;
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null
            && (asin.concreteType().getReference().equals(PythonTypes.list)
                || asin.concreteType().getReference().equals(PythonTypes.tuple))) return ret;
      }
    Set<DType> members;
    try {
      members = this.getDTypes(builder, this.getNode(), vn);
    } catch (IllegalArgumentException e) {
      return ret;
    }
    return members == null || members.isEmpty() ? ret : members;
  }

  /**
   * NumPy's promotion of the dtypes of the arguments at the given positions, over every combination
   * of their possible dtypes.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param positions The arguments' positional indices, excluding the callable.
   * @return The promoted dtypes; {@code {UNKNOWN}} when an argument's dtype cannot be read.
   */
  protected Set<DType> getPromotedDTypesOfArguments(
      PropagationCallGraphBuilder builder, int... positions) {
    Set<DType> ret = null;
    for (int position : positions) {
      Set<DType> argument = this.getNumpyDTypesOfArgument(builder, position);
      if (argument == null || argument.isEmpty()) return EnumSet.of(DType.UNKNOWN);
      if (ret == null) {
        ret = EnumSet.copyOf(argument);
        continue;
      }
      Set<DType> promoted = EnumSet.noneOf(DType.class);
      for (DType a : ret) for (DType b : argument) promoted.add(NpConcatenate.promote(a, b));
      ret = promoted;
    }
    return ret == null ? EnumSet.of(DType.UNKNOWN) : ret;
  }

  /**
   * The number of elements of an array of the given shape, its flattened length.
   *
   * @param shape The shape, or {@code null} when it is unknown.
   * @return The product of the extents when every one is a constant, a {@link DynamicDim} when one
   *     is (the fold stays dynamic), and an {@link UnresolvedDim} otherwise.
   */
  protected static Dimension<?> size(List<Dimension<?>> shape) {
    if (shape == null) return UnresolvedDim.INSTANCE;
    long product = 1;
    boolean dynamic = false;
    boolean known = true;
    for (Dimension<?> d : shape) {
      if (d instanceof NumericDim numeric) product *= numeric.value();
      else {
        known = false;
        if (d instanceof DynamicDim) dynamic = true;
      }
    }
    if (dynamic) return DynamicDim.INSTANCE;
    return known && product <= Integer.MAX_VALUE
        ? new NumericDim((int) product)
        : UnresolvedDim.INSTANCE;
  }

  /**
   * Combines two extents arithmetically, keeping a constant result only when both are constants.
   *
   * @param a One extent.
   * @param b The other extent.
   * @param sum {@code true} to add the extents, {@code false} to multiply them.
   * @return The combined extent: a {@link DynamicDim} when either is one, an {@link UnresolvedDim}
   *     when either is otherwise not a constant.
   */
  protected static Dimension<?> combine(Dimension<?> a, Dimension<?> b, boolean sum) {
    if (a instanceof DynamicDim || b instanceof DynamicDim) return DynamicDim.INSTANCE;
    if (!(a instanceof NumericDim x) || !(b instanceof NumericDim y)) return UnresolvedDim.INSTANCE;
    long value = sum ? (long) x.value() + y.value() : (long) x.value() * y.value();
    return value <= Integer.MAX_VALUE ? new NumericDim((int) value) : UnresolvedDim.INSTANCE;
  }

  /**
   * Whether some call site passes a non-{@code None} argument at a position or by keyword. An
   * argument written as the {@code None} literal counts as absent, since NumPy's optional
   * parameters default to {@code None} or to what {@code None} means.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph lookup.
   * @param position The argument's positional index, excluding the callable.
   * @param name The argument's keyword name.
   * @return {@code true} iff some call site passes it.
   */
  protected boolean isArgumentPassed(
      PropagationCallGraphBuilder builder, int position, String name) {
    PythonInvokeInstruction call = this.getInvokeInstruction();
    if (call != null) return passes(this.getNode(), call, position, name);
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode()))
      if (callerInvoke.snd instanceof PythonInvokeInstruction invoke
          && passes(callerInvoke.fst, invoke, position, name)) return true;
    return false;
  }

  private static boolean passes(
      CGNode node, PythonInvokeInstruction call, int position, String name) {
    int vn = call.getUse(name);
    if (vn == -1 && call.getNumberOfPositionalParameters() > position + 1)
      vn = call.getUse(position + 1);
    if (vn == -1) return false;
    SymbolTable symbols = node.getIR().getSymbolTable();
    return !(symbols.isConstant(vn) && symbols.getConstantValue(vn) == null);
  }

  /**
   * The integer constant an argument always is.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param position The argument's positional index, excluding the callable.
   * @param name The argument's keyword name.
   * @return The constant, or {@code null} when the argument is not one integer constant.
   */
  protected Integer getConstantIntegerArgument(
      PropagationCallGraphBuilder builder, int position, String name) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, position, name);
    if (pts == null || pts.isEmpty()) {
      // A value with no constant in its points-to set, such as `int(8)`, may still resolve by its
      // definitions.
      int vn = this.getArgumentValueNumber(position);
      return vn <= 0
          ? null
          : resolveIntFlowSensitively(
              builder, this.getNode(), vn, new HashSet<>(), FLOW_SENSITIVE_CONSTANT_DEPTH_CAP);
    }
    Integer ret = null;
    for (InstanceKey ik : pts) {
      if (!(ik instanceof ConstantKey<?> constant)
          || !(constant.getValue() instanceof Number number)
          || constant.getValue() instanceof Double
          || constant.getValue() instanceof Float) return null;
      if (ret == null) ret = number.intValue();
      else if (ret != number.intValue()) return null;
    }
    return ret;
  }
}
