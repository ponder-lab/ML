package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code tf.GradientTape.gradient}. Returns a fresh tensor whose shape and dtype
 * match the {@code sources} argument — the gradient of a function w.r.t. a tensor has the same
 * shape and dtype as that tensor. The output is a distinct allocation from {@code sources} (no
 * input alias). Per-source list/tensor structure is not modeled here; the common single-source case
 * recovers the right shape and dtype, which is what downstream tests in the suite assert. See
 * wala/ML#430.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/GradientTape#gradient">tf.GradientTape.gradient</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Gradient extends TensorGenerator {

  private enum Parameters {
    TARGET,
    SOURCES,
    OUTPUT_GRADIENTS,
    UNCONNECTED_GRADIENTS;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public Gradient(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return shapesOfArg(builder, Parameters.SOURCES.getIndex(), Parameters.SOURCES.getName());
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> dtypes =
        dtypesOfArg(builder, Parameters.SOURCES.getIndex(), Parameters.SOURCES.getName());
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  private Set<List<Dimension<?>>> shapesOfArg(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts != null && !pts.isEmpty()) {
      Set<List<Dimension<?>>> shapes = this.getShapesOfValue(builder, pts);
      if (shapes != null && !shapes.isEmpty()) return shapes;
    }
    return this.getArgumentShapesViaCallers(builder, paramPos, paramName);
  }

  private Set<DType> dtypesOfArg(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts != null && !pts.isEmpty()) {
      Set<DType> dtypes = this.getDTypesOfValue(builder, pts);
      if (dtypes != null && !dtypes.isEmpty()) return dtypes;
    }
    return this.getArgumentDTypesViaCallers(builder, paramPos, paramName);
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
}
