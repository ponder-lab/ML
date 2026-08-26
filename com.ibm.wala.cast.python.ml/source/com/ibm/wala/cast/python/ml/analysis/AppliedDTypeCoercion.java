package com.ibm.wala.cast.python.ml.analysis;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import java.util.EnumSet;

/**
 * The record of one applied parameter dtype coercion (wala/ML#838): the dtypes the parameter's
 * callers FED it beside the dtypes its consumers IMPOSE on it eagerly (wala/ML#828, wala/ML#829).
 * The analysis reports the imposed dtypes as the parameter's own, which is correct for a client
 * writing input declarations; a client converting a function WITHOUT writing a declaration needs
 * the fed side back, since tracing materializes the argument at its fed dtype and the conversion is
 * safe exactly when nothing was changed.
 *
 * @param fed The dtypes the callers feed, read off the parameter's dataflow predecessors after the
 *     fixpoint; empty when no predecessor carried a resolved dtype.
 * @param imposed The dtypes eager execution imposes at the parameter's consumers: a singleton where
 *     the consumers agree, the union where they disagree (wala/ML#829).
 */
public record AppliedDTypeCoercion(EnumSet<DType> fed, EnumSet<DType> imposed) {

  /**
   * Whether the applied coercion CHANGED the parameter's dtype at some consumer: {@code false}
   * exactly when the imposition is a SINGLETON that contains every fed dtype. An imposition equal
   * to the fed dtype is recorded for conflict detection but changes nothing, and a conversion
   * without a declaration is safe there. A disagreement union (wala/ML#829) always reads changed,
   * whatever the fed side: the parameter is computed at more than one dtype, so at least one
   * consumer's imposition differs from any single materialization. Subset reasoning would be
   * unsound there, since the union hides WHICH consumer imposes which dtype. An empty fed side
   * under a singleton reads unchanged, since nothing observable was replaced.
   *
   * @return {@code true} iff some consumer computes with a dtype other than a fed one.
   */
  public boolean changed() {
    return this.imposed.size() != 1 || !this.imposed.containsAll(this.fed);
  }
}
