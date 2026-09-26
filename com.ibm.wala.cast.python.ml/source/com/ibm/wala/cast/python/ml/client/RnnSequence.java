package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Generator for the {@code RNN} summary's internal {@code rnn_sequence} op (wala/ML#973): the
 * layer's output from its cell's step output. The dtype is the step output's. With {@code
 * return_sequences} true the layer stacks one step output per time step, so the time axis of the
 * layer's input (axis 1, batch-major) is reinserted after the batch axis; with it false the output
 * is the last step's, so the step shape stands. When the flag cannot be decided, or the time axis
 * cannot be read, the shape is unknown while the dtype still holds.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class RnnSequence extends PassThroughUnaryTensorGenerator {

  private static final int TIME_AXIS = 1;

  private static final String RETURN_SEQUENCES_FIELD_NAME = "return_sequences";

  private enum Parameters {
    STEP,
    INPUTS,
    LAYER;

    String getName() {
      return name().toLowerCase(java.util.Locale.ROOT);
    }
  }

  public RnnSequence(PointsToSetVariable source) {
    super(source);
  }

  public RnnSequence(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return Parameters.STEP.ordinal();
  }

  @Override
  protected String getInputParameterName() {
    return Parameters.STEP.getName();
  }

  /**
   * The layer's {@code return_sequences}, read off every layer instance the call may run on: {@code
   * TRUE} or {@code FALSE} when every instance agrees on a supplied constant or leaves Keras's
   * default of {@code false}, {@code null} otherwise.
   */
  private Boolean returnSequences(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> layers =
        this.getArgumentPointsToSet(
            builder, Parameters.LAYER.ordinal(), Parameters.LAYER.getName());
    if (layers == null || layers.isEmpty()) return null;
    Boolean decided = null;
    for (InstanceKey layer : layers) {
      if (!(layer instanceof AllocationSiteInNode)) return null;
      Set<Boolean> values =
          getPossibleBooleanValues(
              getInstanceFieldPointsToSet(
                  builder, (AllocationSiteInNode) layer, RETURN_SEQUENCES_FIELD_NAME));
      if (values == null) return null;
      // An empty field is an argument no constructor call supplied: Keras's default holds.
      boolean value = values.isEmpty() ? false : values.iterator().next();
      if (values.size() > 1 || (decided != null && decided != value)) return null;
      decided = value;
    }
    return decided;
  }

  /**
   * The rule from one step-output shape to the layer's output shapes, given the flag and the time
   * extents the layer's input can have.
   */
  private ShapeTransform sequenceRule(PropagationCallGraphBuilder builder) {
    Boolean sequences = this.returnSequences(builder);
    if (sequences == null) return input -> null;
    if (!sequences) return input -> java.util.Collections.singleton(input);
    Set<List<Dimension<?>>> inputShapes =
        this.shapesOfArg(builder, Parameters.INPUTS.ordinal(), Parameters.INPUTS.getName());
    if (inputShapes == null || inputShapes.isEmpty()) return input -> null;
    Set<Dimension<?>> timeExtents = HashSetFactory.make();
    for (List<Dimension<?>> shape : inputShapes) {
      if (shape == null || shape.size() <= TIME_AXIS) return input -> null;
      timeExtents.add(shape.get(TIME_AXIS));
    }
    return input -> {
      if (input.isEmpty()) return null; // A step output has at least its batch axis.
      Set<List<Dimension<?>>> outs = HashSetFactory.make();
      for (Dimension<?> time : timeExtents) {
        List<Dimension<?>> out = new ArrayList<>(input);
        out.add(TIME_AXIS, time);
        outs.add(out);
      }
      return outs;
    };
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> stepShapes = super.getDefaultShapes(builder);
    if (stepShapes == null) return null;
    ShapeTransform rule = this.sequenceRule(builder);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> step : stepShapes) {
      Set<List<Dimension<?>>> outs = rule.apply(step);
      if (outs == null) return null;
      ret.addAll(outs);
    }
    return ret.isEmpty() ? null : ret;
  }

  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    return ShapeResult.fromLegacy(this.getDefaultShapes(builder));
  }

  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    return this.getTypeFeed(builder, this.sequenceRule(builder));
  }
}
