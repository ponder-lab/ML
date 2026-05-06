package com.ibm.wala.cast.python.ml.client;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a {@link TensorGenerator} subclass as exempt from the dispatch-coverage meta-test (see
 * {@code TestTensorGeneratorDispatchCoverage} in the test module). Use sparingly, and only when the
 * subclass is genuinely not reachable from either dispatch entry point — i.e., it's constructed by
 * other generators rather than from {@link TensorGeneratorFactory#getGenerator} or {@link
 * TensorGenerator#createManualGenerator}.
 *
 * <p>Examples of legitimate exemptions:
 *
 * <ul>
 *   <li>Delegation-only subclasses that are constructed by another generator rather than by either
 *       dispatch table directly.
 *   <li>Anonymous inline subclasses (which can't be annotated anyway — the meta-test naturally
 *       skips these via the {@link Class#getCanonicalName} check).
 * </ul>
 *
 * <p>Abstract bases are skipped automatically by the meta-test (via {@link
 * java.lang.reflect.Modifier#isAbstract}); they don't need this annotation.
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
