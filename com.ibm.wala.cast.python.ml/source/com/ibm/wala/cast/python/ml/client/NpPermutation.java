package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
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

    // The array arm: the shape survives the draw unchanged.
    OrdinalSet<InstanceKey> argumentPointsToSet =
        this.getArgumentPointsToSet(builder, X_PARAMETER_POSITION, X_PARAMETER_NAME);
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
    Set<DType> preserved = this.getDTypesOfValue(builder, argumentPointsToSet);

    return preserved == null || preserved.isEmpty() ? EnumSet.of(DType.UNKNOWN) : preserved;
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
