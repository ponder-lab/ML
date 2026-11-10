package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.OptionalInt;
import java.util.Set;

/**
 * The elements of a value, read through {@link ContainerElements#of} for each instance the value's
 * points-to set holds, merged. Their common length, when they agree, the union of their types, and
 * completeness only when each is complete. A value whose points-to set is empty has no length and
 * is not complete.
 */
public final class ValueElements {

  /** The elements of each instance the value may hold. */
  private final List<ContainerElements> parts;

  private ValueElements(List<ContainerElements> parts) {
    this.parts = parts;
  }

  /**
   * Reads the elements of a value.
   *
   * @param builder The builder whose pointer analysis models the value.
   * @param value The value.
   * @return Its elements.
   */
  public static ValueElements of(PropagationCallGraphBuilder builder, PointerKey value) {
    List<ContainerElements> parts = new ArrayList<>();
    for (InstanceKey instance : builder.getPointerAnalysis().getPointsToSet(value))
      parts.add(ContainerElements.of(builder, instance));
    return new ValueElements(parts);
  }

  /**
   * The length the instances agree on.
   *
   * @return The length, or empty when they disagree, when one has none, or when there is none.
   */
  public OptionalInt arity() {
    if (this.parts.isEmpty()) return OptionalInt.empty();
    OptionalInt ret = this.parts.get(0).arity();
    for (ContainerElements part : this.parts)
      if (!part.arity().equals(ret)) return OptionalInt.empty();
    return ret;
  }

  /**
   * Whether every instance's elements are all known.
   *
   * @return {@code true} iff there is an instance and each is complete.
   */
  public boolean isComplete() {
    if (this.parts.isEmpty()) return false;
    for (ContainerElements part : this.parts) if (!part.isComplete()) return false;
    return true;
  }

  /**
   * The keys of the one instance's positions.
   *
   * @return Its positions, or empty for none or several instances.
   */
  public List<PointerKey> positions() {
    return this.parts.size() == 1 ? this.parts.get(0).positions() : List.of();
  }

  /**
   * Every key that can hold an element of any instance.
   *
   * @return The keys.
   */
  public Set<PointerKey> allElements() {
    Set<PointerKey> ret = new LinkedHashSet<>();
    for (ContainerElements part : this.parts) ret.addAll(part.allElements());
    return Collections.unmodifiableSet(ret);
  }

  /**
   * Per position, the union of the instances' types there.
   *
   * @param analysis The solved analysis.
   * @return The types, empty when the length is not known.
   */
  public List<Set<TensorType>> positionTypes(TensorTypeAnalysis analysis) {
    if (this.parts.size() == 1) return ElementTypes.positionTypes(analysis, this.parts.get(0));
    OptionalInt arity = this.arity();
    if (arity.isEmpty()) return List.of();
    List<Set<TensorType>> ret = new ArrayList<>();
    for (int i = 0; i < arity.getAsInt(); i++) ret.add(new HashSet<>());
    for (ContainerElements part : this.parts) {
      List<Set<TensorType>> types = ElementTypes.positionTypes(analysis, part);
      for (int i = 0; i < ret.size() && i < types.size(); i++) ret.get(i).addAll(types.get(i));
    }
    List<Set<TensorType>> unmodifiable = new ArrayList<>();
    for (Set<TensorType> types : ret) unmodifiable.add(Collections.unmodifiableSet(types));
    return Collections.unmodifiableList(unmodifiable);
  }

  /**
   * The union of the instances' element types.
   *
   * @param analysis The solved analysis.
   * @return The types.
   */
  public Set<TensorType> elementTypes(TensorTypeAnalysis analysis) {
    Set<TensorType> ret = new HashSet<>();
    for (ContainerElements part : this.parts) ret.addAll(ElementTypes.elementTypes(analysis, part));
    return Collections.unmodifiableSet(ret);
  }

  /**
   * Whether an element of an instance may be {@code None}.
   *
   * @return {@code true} iff some instance's may, or there is no instance.
   */
  public boolean mayHoldNone() {
    if (this.parts.isEmpty()) return true;
    for (ContainerElements part : this.parts) if (part.mayHoldNone()) return true;
    return false;
  }
}
