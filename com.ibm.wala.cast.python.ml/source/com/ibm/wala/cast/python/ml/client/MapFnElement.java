package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Generator for the element {@code tf.map_fn(fn, elems)} passes {@code fn}: {@code elems} unstacked
 * along its leading dimension, so the element has the {@code elems} shape without its first
 * dimension, and the dtype is inherited from {@code elems} (wala/ML#942). The summary produces the
 * element by a call to {@code map_fn_element} with {@code elems} as its only argument.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/map_fn">tf.map_fn</a>
 */
public class MapFnElement extends PassThroughUnaryTensorGenerator {

  public MapFnElement(PointsToSetVariable source) {
    super(source);
  }

  public MapFnElement(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "elems";
  }

  /**
   * Derives the element's shape from the {@code elems} shape by removing its leading dimension.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @return The element shapes, or {@code null} (⊤) when the {@code elems} shape is unknown or some
   *     member is a scalar.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> inputShapes = super.getDefaultShapes(builder);
    if (inputShapes == null || inputShapes.isEmpty()) return inputShapes;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> input : inputShapes) {
      if (input.isEmpty()) return null; // Mapping over a scalar is a runtime error.
      ret.add(new ArrayList<>(input.subList(1, input.size())));
    }
    return ret;
  }

  /**
   * Collapse-safe record view (wala/ML#718): this generator transforms its input shapes in {@link
   * #getDefaultShapes}, which the pass-through identity record path would bypass, so the record
   * view routes through the legacy transform.
   *
   * @param builder The propagation call graph builder.
   * @return The transformed result, with any partial input collapsed by the legacy view.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    return ShapeResult.fromLegacy(this.getDefaultShapes(builder));
  }

  /**
   * This generator transforms its input's shape, so forwarding operand shapes would overclaim; the
   * feed carries dtype only (wala/ML#682).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The dtype-only feed over the caller-side input keys, or {@code null} when none is
   *     located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    return this.getTypeFeed(builder, TypeFeedKind.DTYPE_ONLY);
  }
}
