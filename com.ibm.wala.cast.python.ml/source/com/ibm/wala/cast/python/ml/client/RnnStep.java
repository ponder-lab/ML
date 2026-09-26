package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * Generator for the {@code RNN} summary's internal {@code rnn_step} op (wala/ML#973): one time step
 * of a batch-major sequence, so the output is the input without its time axis (axis 1), in the
 * input's dtype. It is what the layer hands its cell on each step.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class RnnStep extends PassThroughUnaryTensorGenerator {

  /** The time axis of a batch-major sequence. */
  private static final int TIME_AXIS = 1;

  public RnnStep(PointsToSetVariable source) {
    super(source);
  }

  public RnnStep(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "inputs";
  }

  /**
   * Drops the time axis: a sequence of rank {@code r} steps as rank {@code r - 1}. An input of rank
   * below two has no time axis, so the rule cannot type it.
   */
  private static final ShapeTransform DROP_TIME_AXIS =
      input -> {
        if (input.size() <= TIME_AXIS) return null;
        List<Dimension<?>> out = new ArrayList<>(input);
        out.remove(TIME_AXIS);
        return Collections.singleton(out);
      };

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> inputShapes = super.getDefaultShapes(builder);
    if (inputShapes == null) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> input : inputShapes) {
      Set<List<Dimension<?>>> outs = DROP_TIME_AXIS.apply(input);
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
    return this.getTypeFeed(builder, DROP_TIME_AXIS);
  }
}
