package com.ibm.wala.cast.python.ml.test;

import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.client.ContainerElements;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.util.collections.Pair;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;

/**
 * The tensor types the analysis gives a container's elements, read as a client reads them: the
 * solved {@link TensorTypeAnalysis}'s typed keys, through its {@link
 * TensorTypeAnalysis#iterator()}, looked up by each element key {@link ContainerElements} names. A
 * key the iterator does not yield has no tensor types.
 */
public final class ElementTypes {

  /** Per solved analysis, the types of each key it gives a type. */
  private static final Map<TensorTypeAnalysis, Map<PointerKey, Set<TensorType>>> TYPED =
      Collections.synchronizedMap(new WeakHashMap<>());

  private ElementTypes() {}

  /**
   * The tensor types the analysis gives the element at each position.
   *
   * @param analysis The solved analysis.
   * @param elements The container's elements.
   * @return Per position of {@link ContainerElements#positions()}, the element's tensor types, an
   *     empty set when the analysis gives it none; empty when the length is not known.
   */
  public static List<Set<TensorType>> positionTypes(
      TensorTypeAnalysis analysis, ContainerElements elements) {
    List<Set<TensorType>> ret = new ArrayList<>(elements.positions().size());
    for (PointerKey key : elements.positions()) ret.add(of(analysis, key));
    return Collections.unmodifiableList(ret);
  }

  /**
   * The tensor types the analysis gives the container's elements, whatever their positions, over
   * every element key ({@link ContainerElements#allElements()}).
   *
   * @param analysis The solved analysis.
   * @param elements The container's elements.
   * @return The union of the elements' tensor types; empty when no element has tensor evidence.
   */
  public static Set<TensorType> elementTypes(
      TensorTypeAnalysis analysis, ContainerElements elements) {
    Set<TensorType> ret = new HashSet<>();
    for (PointerKey key : elements.allElements()) ret.addAll(of(analysis, key));
    return Collections.unmodifiableSet(ret);
  }

  /**
   * The tensor types the analysis gives a key.
   *
   * @param analysis The solved analysis.
   * @param key The key.
   * @return Its types; empty for a key the analysis gives none.
   */
  private static Set<TensorType> of(TensorTypeAnalysis analysis, PointerKey key) {
    return TYPED.computeIfAbsent(analysis, ElementTypes::typed).getOrDefault(key, Set.of());
  }

  /**
   * The keys a solved analysis gives a tensor type.
   *
   * @param analysis The analysis.
   * @return Each key's types.
   */
  private static Map<PointerKey, Set<TensorType>> typed(TensorTypeAnalysis analysis) {
    Map<PointerKey, Set<TensorType>> ret = new HashMap<>();
    for (Iterator<Pair<PointerKey, TensorVariable>> it = analysis.iterator(); it.hasNext(); ) {
      Pair<PointerKey, TensorVariable> pair = it.next();
      ret.computeIfAbsent(pair.fst, k -> new HashSet<>()).addAll(pair.snd.getTypes());
    }
    ret.replaceAll((k, types) -> Collections.unmodifiableSet(types));
    return ret;
  }
}
