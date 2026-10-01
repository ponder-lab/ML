package com.ibm.wala.cast.python.ipa.callgraph;

import com.ibm.wala.cast.loader.DynamicCallSiteReference;
import com.ibm.wala.types.TypeReference;

/**
 * The call site applying a bare decorator ({@code @d}, no parentheses) to the function it decorates
 * (wala/ML#188). Python applies a bare decorator as {@code d(f)}; the site is marked so that {@link
 * BareDecoratorTargetSelector} can keep library decorator summaries, which are written as factories
 * ({@code d()(f)}), on their own convention.
 */
public class BareDecoratorCallSiteReference extends DynamicCallSiteReference {

  /**
   * Creates the site.
   *
   * @param ref The declared target type of the application.
   * @param pc The program counter of the application.
   */
  public BareDecoratorCallSiteReference(TypeReference ref, int pc) {
    super(ref, pc);
  }
}
