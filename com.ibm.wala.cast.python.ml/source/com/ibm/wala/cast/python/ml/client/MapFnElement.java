package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
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
   * Whether {@code elems} is a structure, a tuple or a list, rather than a tensor. Given a
   * structure, {@code map_fn} unstacks each of its components and passes {@code fn} the same
   * structure of rows, which is not a tensor.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @return {@code true} iff every allocation {@code elems} points to is a tuple or a list.
   */
  private boolean elemsIsStructure(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(
            builder, this.getInputParameterPosition(), this.getInputParameterName());
    if (pts == null || pts.isEmpty()) return false;
    for (InstanceKey ik : pts) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin == null) return false;
      TypeReference type = asin.concreteType().getReference();
      if (!type.equals(tuple) && !type.equals(list)) return false;
    }
    return true;
  }

  /**
   * Derives the element's shape from the {@code elems} shape by removing its leading dimension.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @return The element shapes, {@code null} (⊤) when the {@code elems} shape is unknown or some
   *     member is a scalar, or the empty set (⊥) when {@code elems} is a structure.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // TODO: Type the structure's components (https://github.com/wala/ML/issues/1092).
    if (this.elemsIsStructure(builder)) return Collections.emptySet();

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
   * The element's dtypes are the {@code elems} dtypes, unless {@code elems} is a structure, whose
   * rows are not a tensor.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @return The element dtypes, or the empty set (⊥) when {@code elems} is a structure.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    if (this.elemsIsStructure(builder)) return Collections.emptySet();
    return super.getDefaultDTypes(builder);
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
