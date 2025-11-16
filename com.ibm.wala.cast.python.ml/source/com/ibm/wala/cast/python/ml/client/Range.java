package com.ibm.wala.cast.python.ml.client;

import static java.util.function.Function.identity;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * A representation of the TensorFlow range operation.
 *
 * <p>This class is used to generate a tensor that contains a sequence of numbers, similar to the
 * range function in Python.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/range">TensorFlow range
 *     documentation</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Range extends TensorGenerator {

  @SuppressWarnings("unused")
  private static final Logger LOGGER = Logger.getLogger(Range.class.getName());

  private static final String FUNCTION_NAME = "tf.range()";

  public Range(PointsToSetVariable source, CGNode node) {
    super(source, node);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    // The shape of a range tensor is always a 1D tensor with the length equal to the number of
    // elements in the range. For example, `tf.range(5)` produces a tensor with shape (5,).

    double start = 0; // Default start value.
    double limit = start; // Default limit value.
    double delta = 1; // Default step value.

    // There are two versions of the `range` function:
    // 1. `tf.range(limit)` - generates a range from 0 to limit
    // 2. `tf.range(start, limit, delta)` - generates a range from start to limit with a step of
    // delta.

    // Decide which version of the `range` function is being called based on the number of numeric
    // arguments.
    // TODO: Handle keyword arguments.
    for (Integer numOfPoisitionArguments : getNumberOfPossiblePositionalArguments(builder))
      if (numOfPoisitionArguments == 1) {
        // it must *just* be `limit`.
        int limitValueNumber =
            this.getNode().getMethod().isStatic()
                ? this.getNode().getIR().getParameter(0)
                : this.getNode().getIR().getParameter(1);

        PointerKey limitPK =
            pointerAnalysis.getHeapModel().getPointerKeyForLocal(this.getNode(), limitValueNumber);
        OrdinalSet<InstanceKey> limitPointsToSet = pointerAnalysis.getPointsToSet(limitPK);

        assert !limitPointsToSet.isEmpty() : "Expected a non-empty points-to set for limit.";

        for (InstanceKey limitIK : limitPointsToSet) {
          limit = ((Number) ((ConstantKey<?>) limitIK).getValue()).doubleValue();
          int shape = (int) Math.ceil((limit - start) / delta);
          ret.add(List.of(new NumericDim(shape))); // Add the shape as a 1D tensor.
        }
      } else if (numOfPoisitionArguments >= 3) {
        // it must be `start`, `limit`, and `delta`.
        int startValueNumber =
            this.getNode().getMethod().isStatic()
                ? this.getNode().getIR().getParameter(0)
                : this.getNode().getIR().getParameter(1);

        PointerKey startPK =
            pointerAnalysis.getHeapModel().getPointerKeyForLocal(this.getNode(), startValueNumber);

        int limitValueNumber =
            this.getNode().getMethod().isStatic()
                ? this.getNode().getIR().getParameter(1)
                : this.getNode().getIR().getParameter(2);

        PointerKey limitPK =
            pointerAnalysis.getHeapModel().getPointerKeyForLocal(this.getNode(), limitValueNumber);

        int deltaValueNumber =
            this.getNode().getMethod().isStatic()
                ? this.getNode().getIR().getParameter(2)
                : this.getNode().getIR().getParameter(3);

        PointerKey deltaPK =
            pointerAnalysis.getHeapModel().getPointerKeyForLocal(this.getNode(), deltaValueNumber);

        OrdinalSet<InstanceKey> startPointsToSet = pointerAnalysis.getPointsToSet(startPK);
        OrdinalSet<InstanceKey> limitPointsToSet = pointerAnalysis.getPointsToSet(limitPK);
        OrdinalSet<InstanceKey> deltaPointsToSet = pointerAnalysis.getPointsToSet(deltaPK);

        assert !startPointsToSet.isEmpty() : "Expected a non-empty points-to set for start.";
        assert !limitPointsToSet.isEmpty() : "Expected a non-empty points-to set for limit.";
        assert !deltaPointsToSet.isEmpty() : "Expected a non-empty points-to set for delta.";

        for (InstanceKey startIK : startPointsToSet) {
          start = ((Number) ((ConstantKey<?>) startIK).getValue()).doubleValue();

          for (InstanceKey limitIK : limitPointsToSet) {
            limit = ((Number) ((ConstantKey<?>) limitIK).getValue()).doubleValue();

            for (InstanceKey deltaIK : deltaPointsToSet) {
              delta = ((Number) ((ConstantKey<?>) deltaIK).getValue()).doubleValue();

              int shape = (int) Math.ceil((limit - start) / delta);
              ret.add(List.of(new NumericDim(shape))); // Add the shape as a 1D tensor.
            }
          }
        }
      } else
        throw new IllegalStateException(
            "Expected either 1 or >= 3 positional arguments for range(), but got: "
                + numOfPoisitionArguments
                + ".");

    return ret;
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // The dtype of the resulting tensor is inferred from the inputs unless it is provided
    // explicitly.

    // TODO: Handle keyword arguments.
    EnumSet<DType> types =
        getNumberOfPossiblePositionalArguments(builder).stream()
            .map(
                numArgs ->
                    IntStream.range(0, numArgs)
                        .filter(i -> i < 3) // only numeric arguments.
                        .map(i -> this.getNode().getIR().getMethod().isStatic() ? i : i + 1)
                        .map(this.getNode().getIR()::getParameter)
                        .mapToObj(val -> getDTypes(builder, val).stream())
                        .flatMap(identity())
                        .distinct())
            .flatMap(identity())
            .collect(Collectors.toCollection(() -> EnumSet.noneOf(DType.class)));

    // FIXME: We can't tell the difference here between varying dtypes in a single call and that of
    // possible varying dtypes values from the points-to graph. Below, we are treating it as these
    // values lie in a single call, but that may not be the case.

    if (types.contains(DType.FLOAT64)) return EnumSet.of(DType.FLOAT64);
    else if (types.contains(DType.FLOAT32)) return EnumSet.of(DType.FLOAT32);
    else if (types.contains(DType.INT64)) return EnumSet.of(DType.INT64);
    else if (types.contains(DType.INT32)) return EnumSet.of(DType.INT32);

    throw new IllegalStateException(
        "Expected at least one numeric dtype for range(), but got: " + types + ".");
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    throw new UnsupportedOperationException(
        "Shapes for range() are derived from mandatory numeric arguments and must be provided"
            + " explicitly.");
  }

  @Override
  protected int getShapeParameterPosition() {
    throw new UnsupportedOperationException(
        "Range does not have a shape argument. Its shape is derived from the numeric arguments.");
  }

  @Override
  protected int getDTypeParameterPosition() {
    // TODO: We need a value number for the dtype argument. Also, that value number can differ
    // depending on the version of the `range` function being called.

    return -1; // Positional dtype argument for range() is not yet implemented.
  }

  @Override
  protected String getSignature() {
    return FUNCTION_NAME;
  }
}
