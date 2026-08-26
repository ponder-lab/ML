package com.ibm.wala.cast.python.ml.analysis;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import java.util.EnumSet;
import java.util.Objects;

/**
 * The record of one applied parameter dtype coercion (wala/ML#838): the dtypes the parameter's
 * callers FED it beside the dtypes its consumers IMPOSE on it eagerly (wala/ML#828, wala/ML#829).
 * The analysis reports the imposed dtypes as the parameter's own, which is correct for a client
 * writing input declarations; a client converting a function WITHOUT writing a declaration needs
 * the fed side back, since tracing materializes the argument at its fed dtype and the conversion is
 * safe exactly when nothing was changed.
 *
 * @param fed The resolved dtypes the coercion edges observed arriving before the rewrite; never
 *     carries {@link DType#UNKNOWN}, and empty when nothing resolved.
 * @param imposed The dtypes eager execution imposes at the parameter's consumers: a singleton where
 *     the consumers agree, the union where they disagree (wala/ML#829).
 * @param fedComplete Whether {@code fed} accounts for every inflow: {@code false} when some inflow
 *     carried no resolvable dtype, or when another coerced parameter sits in the backward slice,
 *     whose analysis-side state forwards its IMPOSED dtype where the runtime forwards the original
 *     value.
 */
public record AppliedDTypeCoercion(
    EnumSet<DType> fed, EnumSet<DType> imposed, boolean fedComplete) {

  /**
   * Defensive normalization: both sets are required and copied, so a client can neither mutate this
   * record's classification in place (the wala/ML#753 hazard class) nor construct it around a live
   * set.
   *
   * @param fed The fed dtypes.
   * @param imposed The imposed dtypes.
   * @param fedComplete Whether the fed side accounts for every inflow.
   */
  public AppliedDTypeCoercion {
    Objects.requireNonNull(fed);
    Objects.requireNonNull(imposed);
    fed = EnumSet.copyOf(fed);
    imposed = EnumSet.copyOf(imposed);
  }

  /**
   * The three answers the fed-beside-imposed comparison can give. Ignorance is first-class: reading
   * an empty or incomplete fed side as "unchanged" would be an absence read as evidence of absence,
   * exactly the failure a safety-deciding client must not inherit from its instrument.
   */
  public enum Resolution {
    /** Some consumer computes with a dtype other than a fed one; bare conversion diverges. */
    CHANGED,

    /**
     * Every caller resolved and the imposition is a singleton equal to what they feed; the coercion
     * changed nothing and bare conversion is safe.
     */
    UNCHANGED,

    /**
     * The fed side is empty or incomplete, so whether the coercion changed anything is UNKNOWN: not
     * a hazard signal, but never safety either.
     */
    UNRESOLVED
  }

  /**
   * Classifies this coercion. {@link Resolution#CHANGED} on positive evidence: a disagreement union
   * (wala/ML#829) whatever the fed side — the parameter is computed at more than one dtype, so at
   * least one consumer's imposition differs from any single materialization, and subset reasoning
   * would be unsound since the union hides WHICH consumer imposes which dtype — or a fed dtype
   * outside the imposition. {@link Resolution#UNCHANGED} only when the fed side is non-empty,
   * COMPLETE, and equal in effect to the singleton imposition. Everything else is {@link
   * Resolution#UNRESOLVED}.
   *
   * @return The three-valued classification.
   */
  public Resolution resolution() {
    if (this.imposed.size() != 1) return Resolution.CHANGED;
    if (!this.imposed.containsAll(this.fed)) return Resolution.CHANGED;
    if (this.fed.isEmpty() || !this.fedComplete) return Resolution.UNRESOLVED;
    return Resolution.UNCHANGED;
  }

  /**
   * The two-valued convenience over {@link #resolution()}: {@code false} only for {@link
   * Resolution#UNCHANGED}. {@link Resolution#UNRESOLVED} folds into {@code true} deliberately —
   * wrong-but-safe for a safety-deciding client, which declines and loses an optimization rather
   * than shipping a break; a client wanting the distinction reads {@link #resolution()} directly.
   *
   * @return {@code true} unless every caller resolved and the coercion changed nothing.
   */
  public boolean changed() {
    return this.resolution() != Resolution.UNCHANGED;
  }
}
