package com.ibm.wala.cast.python.ml.test;

import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.client.ContainerElements;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.debug.UnimplementedError;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * The tensor types the analysis gives a container's elements, read as a client reads them: the
 * solved {@link TensorTypeAnalysis}'s state of each element key {@link ContainerElements} names.
 */
public final class ElementTypes {

  private ElementTypes() {}

  /**
   * The tensor types the analysis gives the element at each position.
   *
   * @param builder The builder the analysis ran on.
   * @param analysis The solved analysis.
   * @param elements The container's elements.
   * @return Per position of {@link ContainerElements#positions()}, the element's tensor types, an
   *     empty set when the analysis gives it none; empty when the length is not known.
   */
  public static List<Set<TensorType>> positionTypes(
      PropagationCallGraphBuilder builder,
      TensorTypeAnalysis analysis,
      ContainerElements elements) {
    List<Set<TensorType>> ret = new ArrayList<>(elements.positions().size());
    for (PointerKey key : elements.positions())
      ret.add(Collections.unmodifiableSet(of(builder, analysis, key)));
    return Collections.unmodifiableList(ret);
  }

  /**
   * The tensor types the analysis gives the container's elements, whatever their positions, over
   * every element key ({@link ContainerElements#allElements()}).
   *
   * @param builder The builder the analysis ran on.
   * @param analysis The solved analysis.
   * @param elements The container's elements.
   * @return The union of the elements' tensor types; empty when no element has tensor evidence.
   */
  public static Set<TensorType> elementTypes(
      PropagationCallGraphBuilder builder,
      TensorTypeAnalysis analysis,
      ContainerElements elements) {
    Set<TensorType> ret = new HashSet<>();
    for (PointerKey key : elements.allElements()) ret.addAll(of(builder, analysis, key));
    return Collections.unmodifiableSet(ret);
  }

  /**
   * The tensor types the analysis gives a key.
   *
   * @param builder The builder the analysis ran on.
   * @param analysis The solved analysis.
   * @param key The key.
   * @return Its types; empty for a key the analysis's flow graph does not hold, or whose state is
   *     empty or unknown.
   */
  private static Set<TensorType> of(
      PropagationCallGraphBuilder builder, TensorTypeAnalysis analysis, PointerKey key) {
    Set<TensorType> ret = new HashSet<>();
    if (builder.getPropagationSystem().isImplicit(key)) return ret;
    PointsToSetVariable variable;
    try {
      variable = builder.getPropagationSystem().findOrCreatePointsToSet(key);
    } catch (UnimplementedError e) {
      return ret;
    }
    if (!analysis.getProblem().getFlowGraph().containsNode(variable)) return ret;
    Set<TensorType> state = analysis.getOutState(variable);
    if (state != null) ret.addAll(state);
    return ret;
  }
}
