package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.Locale;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A {@link TensorGenerator} for {@code tf.scatter_nd(indices, updates, shape)}: a fresh tensor
 * whose shape is the {@code shape} argument and whose dtype is that of {@code updates}. The shape
 * read is the allocators' (a literal, a {@code tf.shape} piece, a {@code .shape} attribute); the
 * dtype has no argument of its own and no API default, so an {@code updates} the analysis cannot
 * type leaves the dtype unknown rather than defaulting.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ScatterNd extends TensorTypeAllocator {

  private static final Logger LOGGER = getLogger(ScatterNd.class.getName());

  protected enum Parameters {
    INDICES,
    UPDATES,
    SHAPE,
    NAME;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public ScatterNd(PointsToSetVariable source) {
    super(source);
  }

  public ScatterNd(CGNode node) {
    super(node);
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
    OrdinalSet<InstanceKey> updates =
        this.getArgumentPointsToSet(
            builder, Parameters.UPDATES.getIndex(), Parameters.UPDATES.getName());
    Set<DType> dTypes = this.getDTypesOfValue(builder, updates);
    LOGGER.fine(
        () -> "tf.scatter_nd() dtype from updates for source " + describe(source) + ": " + dTypes);
    return (dTypes == null || dTypes.isEmpty()) ? EnumSet.of(DType.UNKNOWN) : dTypes;
  }
}
