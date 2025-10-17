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
import com.ibm.wala.util.debug.UnimplementedError;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.StreamSupport;

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

  public Range(PointsToSetVariable source, CGNode node) {
    super(source, node);
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    // The shape of a range tensor is always a 1D tensor with the length equal to the number of
    // elements in the range.
    // For example, `tf.range(5)` produces a tensor with shape (5,).

    double start = 0; // Default start value.
    double limit = start; // Default limit value.
    double delta = 1; // Default step value.

    // There are two versions of the `range` function:
    // 1. `tf.range(limit)` - generates a range from 0 to limit
    // 2. `tf.range(start, limit, delta)` - generates a range from start to limit with a step of
    // delta.

    // First, decide which version of the `range` function is being called based on the number of
    // numeric arguments.j
    // TODO: Handle keyword arguments.

    int numOfNumericPositionalArgs = getNumberOfNumericPositionalArgs(pointerAnalysis);

    if (numOfNumericPositionalArgs == 1) {
      // it must *just* be `limit`.
      PointerKey limitPK = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, 2);
      OrdinalSet<InstanceKey> limitPointsToSet = pointerAnalysis.getPointsToSet(limitPK);

      assert !limitPointsToSet.isEmpty() : "Expected a non-empty points-to set for limit.";

      for (InstanceKey limitIK : limitPointsToSet)
        if (limitIK instanceof ConstantKey) {
          limit = ((Number) ((ConstantKey<?>) limitIK).getValue()).doubleValue();
          int shape = (int) Math.ceil((limit - start) / delta);
          ret.add(List.of(new NumericDim(shape))); // Add the shape as a 1D tensor.
        } else
          throw new IllegalStateException(
              "Expected a " + ConstantKey.class + " for limit, but got: " + limitIK + ".");
    } else
      // TODO: Handle more cases.
      throw new UnimplementedError(
          "Currently cannot handle more than one numeric positional argument for range().");

    return ret;
  }

  private int getNumberOfNumericPositionalArgs(PointerAnalysis<InstanceKey> pointerAnalysis) {
    int ret = 0;
    int explicitArgumentIndex = 2; // Start from the first explicit argument.

    while (true) {
      PointerKey pk =
          pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, explicitArgumentIndex);
      OrdinalSet<InstanceKey> pointsToSet = pointerAnalysis.getPointsToSet(pk);

      if (pointsToSet.isEmpty()) break; // End of positional arguments.

      // Check if the pointsToSet contains numeric values.
      boolean allNumeric =
          StreamSupport.stream(pointsToSet.spliterator(), false)
              .filter(ik -> ik instanceof ConstantKey)
              .map(ik -> (ConstantKey<?>) ik)
              .map(ConstantKey::getValue)
              .allMatch(v -> v instanceof Number); // Check if all values are numeric.

      if (!allNumeric) break; // There's some argument that is not numeric for this argument.

      ret++; // Increment the count of numeric positional arguments.
      explicitArgumentIndex++; // Move to the next explicit argument.
    }

    return ret;
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // The dtype of the resulting tensor is inferred from the inputs unless it is provided
    // explicitly.

    // TODO: Handle keyword arguments.
    int numberOfNumericPositionalArgs =
        getNumberOfNumericPositionalArgs(builder.getPointerAnalysis());

    EnumSet<DType> types =
        IntStream.range(0, numberOfNumericPositionalArgs)
            .map(i -> i + 2) // Positional arguments start at index 2.
            .mapToObj(val -> getDTypes(builder, val).stream())
            .flatMap(identity())
            .distinct()
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
  protected int getValueNumberForShapeArgument() {
    throw new UnsupportedOperationException(
        "Range does not have a shape argument. Its shape is derived from the numeric arguments.");
  }

  @Override
  protected int getValueNumberForDTypeArgument() {
    // TODO: We need a value number for the dtype argument. Also, that value number can differ
    // depending on the version of the `range` function being called.

    return -1; // Positional dtype argument for range() is not yet implemented.
  }
}
