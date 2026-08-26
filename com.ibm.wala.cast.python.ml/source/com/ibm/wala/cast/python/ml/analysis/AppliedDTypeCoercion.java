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
   * The three answers the fed-beside-imposed comparison can give. Ignorance is first-class: reading
   * an empty fed side as "unchanged" would be an absence read as evidence of absence, exactly the
   * failure a safety-deciding client must not inherit from its instrument.
   */
  public enum Resolution {
    /** Some consumer computes with a dtype other than a fed one; bare conversion diverges. */
    CHANGED,

    /**
     * The callers resolved and the imposition is a singleton equal to what they feed; the coercion
     * changed nothing and bare conversion is safe.
     */
    UNCHANGED,

    /**
     * No caller-side dtype resolved, so whether the coercion changed anything is UNKNOWN: not a
     * hazard signal, but never safety either.
     */
    UNRESOLVED
  }

  /**
   * Classifies this coercion. {@link Resolution#UNCHANGED} exactly when the fed side is NON-EMPTY
   * and the imposition is a singleton containing every fed dtype. A disagreement union
   * (wala/ML#829) always reads {@link Resolution#CHANGED}, whatever the fed side: the parameter is
   * computed at more than one dtype, so at least one consumer's imposition differs from any single
   * materialization, and subset reasoning would be unsound since the union hides WHICH consumer
   * imposes which dtype. An empty fed side reads {@link Resolution#UNRESOLVED}.
   *
   * @return The three-valued classification.
   */
  public Resolution resolution() {
    if (this.fed.isEmpty()) return Resolution.UNRESOLVED;
    return this.imposed.size() == 1 && this.imposed.containsAll(this.fed)
        ? Resolution.UNCHANGED
        : Resolution.CHANGED;
  }

  /**
   * The two-valued convenience over {@link #resolution()}: {@code false} only for {@link
   * Resolution#UNCHANGED}. {@link Resolution#UNRESOLVED} folds into {@code true} deliberately —
   * wrong-but-safe for a safety-deciding client, which declines and loses an optimization rather
   * than shipping a break; a client wanting the distinction reads {@link #resolution()} (or {@link
   * #fed}'s emptiness) directly.
   *
   * @return {@code true} unless the callers resolved and the coercion changed nothing.
   */
  public boolean changed() {
    return this.resolution() != Resolution.UNCHANGED;
  }
}
