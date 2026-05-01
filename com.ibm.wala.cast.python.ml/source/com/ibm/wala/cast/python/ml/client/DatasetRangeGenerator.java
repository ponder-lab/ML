package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A generator for tensors created by {@code tf.data.Dataset.range}.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetRangeGenerator extends DatasetGenerator {

  /** Parameter indices for {@code tf.data.Dataset.range}. */
  protected enum Parameters {
    /** The start of the range. */
    START,
    /** The end of the range. */
    STOP,
    /** The step of the range. */
    STEP,
    /** The output type of the range. */
    OUTPUT_TYPE,
    /** The name of the range operation. */
    NAME;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public DatasetRangeGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetRangeGenerator(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // tf.data.Dataset.range produces scalars.
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    ret.add(Collections.emptyList());
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // tf.data.Dataset.range produces int64.
    return Set.of(DType.INT64);
  }

  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    Set<Long> starts =
        getPossibleLongValues(
            this.getArgumentPointsToSet(
                builder, Parameters.START.getIndex(), Parameters.START.getName()));
    Set<Long> stops =
        getPossibleLongValues(
            this.getArgumentPointsToSet(
                builder, Parameters.STOP.getIndex(), Parameters.STOP.getName()));
    Set<Long> steps =
        getPossibleLongValues(
            this.getArgumentPointsToSet(
                builder, Parameters.STEP.getIndex(), Parameters.STEP.getName()));

    if (stops.isEmpty() && starts.isEmpty()) {
      return Collections.emptySet();
    }

    // Handle the case where range(stop) is called (start defaults to 0).
    if (stops.isEmpty()) {
      stops = starts;
      starts = Set.of(0L);
    }

    if (steps.isEmpty()) {
      steps = Set.of(1L);
    }

    Set<Long> ret = HashSetFactory.make();
    for (Long start : starts) {
      for (Long stop : stops) {
        for (Long step : steps) {
          if (step != 0) {
            long size = 0;
            if (step > 0) {
              if (stop > start) {
                size = (stop - start + step - 1) / step;
              }
            } else {
              if (stop < start) {
                size = (stop - start + step + 1) / step;
              }
            }
            ret.add(Math.max(0L, size));
          }
        }
      }
    }
    return ret;
  }
}
