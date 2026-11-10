package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import java.util.List;
import java.util.OptionalInt;
import java.util.Set;

/**
 * Queries a client makes of a container instance after the analysis: its length, the keys of its
 * elements, whether those keys are all of them, and whether an element may be {@code None}. A
 * container is a list, tuple, set, dict or {@code enumerate} object, as the analysis models them;
 * the queries do not check that the instance is one, and answer for any other instance as for a
 * container with no recorded elements, so a client asks them only of a container.
 *
 * <p>The element keys are pointer keys of the builder's pointer analysis, so a client reads an
 * element's points-to set, or its tensor types from the {@link
 * com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis} over those keys, itself. Each query
 * reads the pointer analysis only, and makes no tensor-type read.
 */
public final class Containers {

  private Containers() {}

  /**
   * Gets a container's length, when the analysis knows it: a list or tuple display that program
   * code allocates, whose elements all sit at constant indices {@code 0..n-1}, and, for a list,
   * that nothing in the program may mutate in a way the analysis does not record.
   *
   * @param builder The builder whose pointer analysis models the container.
   * @param container The container instance: a list, tuple, set, dict or {@code enumerate} object.
   * @return The number of elements, or empty when the length is not known.
   */
  public static OptionalInt getLength(
      PythonSSAPropagationCallGraphBuilder builder, InstanceKey container) {
    return ContainerElements.of(builder, container).getArity();
  }

  /**
   * Gets the key of a container's element at each position.
   *
   * @param builder The builder whose pointer analysis models the container.
   * @param container The container instance: a list, tuple, set, dict or {@code enumerate} object.
   * @return The keys of positions {@code 0..n-1}, where {@code n} is {@link #getLength}; empty when
   *     the length is not known.
   */
  public static List<PointerKey> getPositionKeys(
      PythonSSAPropagationCallGraphBuilder builder, InstanceKey container) {
    return ContainerElements.of(builder, container).getPositions();
  }

  /**
   * Gets every key that can hold an element of a container, including those of elements whose
   * position is not known. They are all of them only when {@link #isComplete} holds; otherwise a
   * subset.
   *
   * @param builder The builder whose pointer analysis models the container.
   * @param container The container instance: a list, tuple, set, dict or {@code enumerate} object.
   * @return The element keys.
   */
  public static Set<PointerKey> getElementKeys(
      PythonSSAPropagationCallGraphBuilder builder, InstanceKey container) {
    return ContainerElements.of(builder, container).allElements();
  }

  /**
   * Whether {@link #getElementKeys} holds every key that can hold an element of a container. It
   * does not when the analysis saw an element stored under a key it could not name, such as {@code
   * d[k] = t} with {@code k} unresolved, or the program may add one by a means it does not record,
   * such as {@code l.extend(m)} or an unresolved call the container is passed to.
   *
   * @param builder The builder whose pointer analysis models the container.
   * @param container The container instance: a list, tuple, set, dict or {@code enumerate} object.
   * @return {@code true} iff the element keys are all of them.
   */
  public static boolean isComplete(
      PythonSSAPropagationCallGraphBuilder builder, InstanceKey container) {
    return ContainerElements.of(builder, container).isComplete();
  }

  /**
   * Whether an element of a container may be {@code None}: some element key may hold the {@code
   * None} constant, or the keys are not all of them ({@link #isComplete}), so an element may be
   * anything.
   *
   * @param builder The builder whose pointer analysis models the container.
   * @param container The container instance: a list, tuple, set, dict or {@code enumerate} object.
   * @return {@code true} iff an element may be {@code None}.
   */
  public static boolean mayHoldNone(
      PythonSSAPropagationCallGraphBuilder builder, InstanceKey container) {
    return ContainerElements.of(builder, container).mayHoldNone();
  }
}
