package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Modeling of {@code numpy.random.permutation(x)}, and of the {@code RandomState} method of the
 * same name (<a href="https://github.com/wala/ML/issues/858">wala/ML#858</a>).
 *
 * <p>The operation has two arms, and they disagree about the result's shape, so the argument
 * decides which applies:
 *
 * <ul>
 *   <li><b>An array argument.</b> The draw shuffles along the FIRST axis and rearranges nothing
 *       else, so the result has the argument's shape exactly. This is the arm that matters in
 *       practice and the one the subject idiom uses.
 *   <li><b>An integer argument.</b> The result is a shuffled {@code arange}, a rank-one array of
 *       that length with dtype {@code int64}.
 * </ul>
 *
 * <p>The arms have to be distinguished rather than collapsed into a pass-through. An integer's own
 * shape is the scalar {@code ()}, so passing it through would report a scalar where the runtime
 * produces a vector, which is a confidently wrong shape rather than a missing one. Both were
 * checked against NumPy 1.23.5.
 *
 * <p>An argument that resolves to neither leaves the result at ⊤, which is the honest reading: the
 * shape is the argument's and the argument's is unknown.
 */
public class NpPermutation extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(NpPermutation.class.getName());

  /** The 0-based position of the argument, after {@code self}. */
  private static final int X_PARAMETER_POSITION = 0;

  /** The keyword name of the argument. */
  private static final String X_PARAMETER_NAME = "x";

  public NpPermutation(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Manual (node-based) anchor, for producer delegation from the {@code ndarray} this operation
   * allocates.
   *
   * @param node The synthetic {@code do()} node that allocated the value.
   */
  public NpPermutation(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Integer length = this.resolveStaticIntArgument(builder, X_PARAMETER_POSITION, X_PARAMETER_NAME);

    if (length != null) {
      // The integer arm: a shuffled `arange`, so rank one of that length. Reached only when the
      // argument is a statically resolvable integer, which an array argument never is.
      LOGGER.fine(() -> "NpPermutation: integer argument " + length + "; result is rank one.");
      Set<List<Dimension<?>>> ret = HashSetFactory.make();
      ret.add(List.of(new NumericDim(length)));
      return ret;
    }

    OrdinalSet<InstanceKey> argumentPointsToSet =
        this.getArgumentPointsToSet(builder, X_PARAMETER_POSITION, X_PARAMETER_NAME);

    if (isIntegralArgument(argumentPointsToSet)) {
      // Still the integer arm, with a length that does not resolve to one value: the argument is
      // numeric at every site but carries more than one, so `resolveStaticIntArgument` declines.
      // Falling through to the array arm here would map a number to its own scalar shape and
      // report `()` where the runtime gives a vector, which is the confidently wrong shape this
      // class exists to avoid, reached by the other route. The rank is known regardless of the
      // length, so the extent is what degrades.
      LOGGER.fine(() -> "NpPermutation: integral argument of unresolved length; rank one.");
      Set<List<Dimension<?>>> ret = HashSetFactory.make();
      ret.add(List.of(UnresolvedDim.INSTANCE));
      return ret;
    }

    // The array arm: the shape survives the draw unchanged.
    Set<List<Dimension<?>>> preserved = this.getShapesOfValue(builder, argumentPointsToSet);
    LOGGER.fine(() -> "NpPermutation: array argument; preserved shapes " + preserved + ".");

    // ⊤ when the argument does not resolve: the result is still an array shaped like its argument,
    // never ⊥.
    return preserved == null || preserved.isEmpty() ? null : preserved;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Integer length = this.resolveStaticIntArgument(builder, X_PARAMETER_POSITION, X_PARAMETER_NAME);

    // The integer arm produces indices, whatever the integer's own dtype.
    if (length != null) return EnumSet.of(DType.INT64);

    OrdinalSet<InstanceKey> argumentPointsToSet =
        this.getArgumentPointsToSet(builder, X_PARAMETER_POSITION, X_PARAMETER_NAME);

    // The dtype shares the shape's edge: an integral argument whose length does not resolve is
    // still the integer arm, and its result is still indices rather than the argument's own dtype.
    if (isIntegralArgument(argumentPointsToSet)) return EnumSet.of(DType.INT64);

    Set<DType> preserved = this.getDTypesOfValue(builder, argumentPointsToSet);

    return preserved == null || preserved.isEmpty() ? EnumSet.of(DType.UNKNOWN) : preserved;
  }

  /**
   * Whether the argument is an integer at every site it can hold, which selects the integer arm
   * whether or not a single length resolves.
   *
   * <p>Separate from {@link #resolveStaticIntArgument}, which answers a narrower question: that
   * helper declines a multi-valued argument, and declining is not the same as the argument not
   * being an integer. Treating the decline as "not an integer" sends a numeric argument down the
   * array arm, whose pass-through maps a number to the scalar shape the number itself has.
   *
   * @param argumentPointsToSet The argument's points-to set.
   * @return {@code true} iff the set is non-empty and every member is a numeric constant.
   */
  private static boolean isIntegralArgument(OrdinalSet<InstanceKey> argumentPointsToSet) {
    if (argumentPointsToSet == null || argumentPointsToSet.isEmpty()) return false;

    for (InstanceKey instanceKey : argumentPointsToSet) {
      if (!(instanceKey instanceof ConstantKey)) return false;
      if (!(((ConstantKey<?>) instanceKey).getValue() instanceof Number)) return false;
    }

    return true;
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
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }

  /**
   * Returns the producing library of the modeled value: a NumPy draw, so the value is an ndarray
   * (wala/ML#724).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link TensorOrigin#NUMPY}, singleton.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
  }
}
