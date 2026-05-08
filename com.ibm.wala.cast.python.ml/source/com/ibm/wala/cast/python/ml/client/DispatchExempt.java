package com.ibm.wala.cast.python.ml.client;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a {@link TensorGenerator} subclass as exempt from the dispatch-coverage meta-test (see
 * {@code TestTensorGeneratorDispatchCoverage} in the test module). The annotation suppresses the
 * "must be constructed in {@link TensorGeneratorFactory#getGenerator} or {@link
 * TensorGenerator#createManualGenerator}" check for the annotated subclass. Use sparingly.
 *
 * <p>Two distinct uses:
 *
 * <ol>
 *   <li><strong>Permanent exemption — delegation-only subclasses</strong>: a generator that is
 *       constructed by another generator rather than by either dispatch table directly. The
 *       annotation here documents an architectural choice; no follow-up action is implied.
 *   <li><strong>Transient exemption — known orphan pending wire-or-delete</strong>: a generator
 *       that has no construction site anywhere in the codebase (orphan flagged by the meta-test
 *       itself). The annotation here is a TODO marker — the {@link #value} should reference the
 *       filed sub-issue tracking the wire-or-delete decision, and the annotation should be removed
 *       when that decision lands.
 * </ol>
 *
 * <p>Anonymous inline subclasses (which can't be annotated anyway) are skipped by the meta-test via
 * the {@link Class#getCanonicalName} check. Abstract bases are skipped via {@link
 * java.lang.reflect.Modifier#isAbstract}; they don't need this annotation.
 *
 * <p>This is a stop-gap until the dispatch-table unification proposed in wala/ML#469 lands. After
 * that, this annotation can be retired.
 *
 * @see TensorGeneratorFactory
 * @see TensorGenerator#createManualGenerator
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface DispatchExempt {
  /**
   * Free-form reason the subclass is exempt — surfaced in the meta-test diagnostic if its dispatch
   * status is ever questioned.
   *
   * @return The reason for exemption.
   */
  String value() default "";
}
