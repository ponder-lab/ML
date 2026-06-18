package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Base for generators of TensorFlow/NumPy APIs that allocate a tensor from an explicit {@code
 * shape} argument plus an optional {@code dtype} argument (e.g. {@code tf.ones}, {@code tf.zeros},
 * {@code tf.keras.Input}, {@code tf.random.uniform}, {@code tf.one_hot}). It owns the {@code
 * SHAPE}/{@code DTYPE} parameter-slot machinery, the float32 default dtype, the "shape is
 * mandatory" contract, and the constant-extraction helper. It carries no value semantics, so it is
 * not itself dispatched: concrete subclasses (one per API) extend it. Introduced to replace {@code
 * extends Ones} code-reuse-not-is-a inheritance (<a
 * href="https://github.com/wala/ML/issues/514">wala/ML#514</a>).
 */
public abstract class ShapeAndDTypeAllocator extends TensorGenerator {

  private static final Logger LOGGER = getLogger(ShapeAndDTypeAllocator.class.getName());

  protected enum Parameters {
    SHAPE,
    DTYPE,
    NAME;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public ShapeAndDTypeAllocator(PointsToSetVariable source) {
    super(source);
  }

  public ShapeAndDTypeAllocator(CGNode node) {
    super(node);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    LOGGER.fine(
        "No dtype specified for source: " + source + ". Using default dtype of: " + FLOAT32 + " .");

    // Use the default dtype of float32.
    return EnumSet.of(FLOAT32);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    throw new UnsupportedOperationException("Shape is mandatory and must be provided explicitly.");
  }

  @Override
  protected int getShapeParameterPosition() {
    return Parameters.SHAPE.getIndex();
  }

  @Override
  protected String getShapeParameterName() {
    return Parameters.SHAPE.getName();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }

  protected static Optional<Integer> getIntValueFromInstanceKey(InstanceKey instanceKey) {
    if (instanceKey instanceof ConstantKey) {
      ConstantKey<?> constantKey = (ConstantKey<?>) instanceKey;
      Object value = constantKey.getValue();

      if (value == null) return Optional.empty();
      return Optional.of(((Number) value).intValue());
    }

    throw new IllegalArgumentException(
        "Cannot get integer value from non-constant InstanceKey: " + instanceKey + ".");
  }
}
