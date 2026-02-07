package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A generator for tensors created by the `reshape()` function in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/reshape">TensorFlow reshape()
 *     API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Reshape extends ZerosLike {

  private static final Logger LOGGER = getLogger(Reshape.class.getName());

  protected enum Parameters {
    TENSOR,
    SHAPE,
    NAME;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public Reshape(PointsToSetVariable source) {
    super(source);
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
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> types =
        this.getDTypesOfValue(
            builder,
            this.getArgumentPointsToSet(
                builder, this.getValueParameterPosition(), this.getValueParameterName()));
    if (!types.isEmpty()) {
      return types;
    }
    return EnumSet.of(FLOAT32);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    throw new UnsupportedOperationException("Shape is mandatory and must be provided explicitly.");
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> explicitShapes = super.getShapes(builder);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    // Get input tensor shapes to resolve symbolic dimensions.
    Set<List<Dimension<?>>> inputShapes =
        this.getShapes(builder, this.getArgumentValueNumber(builder, Parameters.TENSOR.getIndex()));

    for (List<Dimension<?>> reshapeTo : explicitShapes) {
      // Create a temporary TensorType to use helper methods (symbolicDims, concreteSize)
      // DType doesn't matter for shape calculation.
      TensorType reshapeType = new TensorType(FLOAT32.name().toLowerCase(), reshapeTo);
      int ssz = reshapeType.symbolicDims();
      int csz = reshapeType.concreteSize();

      boolean resolved = false;
      if (inputShapes != null) {
        for (List<Dimension<?>> inputShape : inputShapes) {
          TensorType inputType = new TensorType(FLOAT32.name().toLowerCase(), inputShape);

          if (ssz == 1 && inputType.symbolicDims() == 0 && inputType.concreteSize() != -1) {
            int totalSize = inputType.concreteSize();
            int partialSize = 1;
            for (Dimension<?> d : reshapeTo) {
              if (d instanceof NumericDim) {
                partialSize *= ((NumericDim) d).value();
              }
            }

            if (partialSize > 0 && totalSize % partialSize == 0) {
              int missingDim = totalSize / partialSize;
              List<Dimension<?>> newDims = new ArrayList<>();
              for (Dimension<?> d : reshapeTo) {
                if (d instanceof SymbolicDim) {
                  newDims.add(new NumericDim(missingDim));
                } else {
                  newDims.add(d);
                }
              }
              ret.add(newDims);
              resolved = true;
            }
          }
        }
      }

      if (!resolved) {
        ret.add(reshapeTo);
      }
    }
    return ret;
  }
}
