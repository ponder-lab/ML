package com.ibm.wala.cast.python.ipa.callgraph;

import com.ibm.wala.cast.loader.DynamicCallSiteReference;
import com.ibm.wala.types.TypeReference;

/** The call site applying a parenthesized decorator to a class; see {@link ClassDecoratorSite}. */
public class ClassDecoratorCallSiteReference extends DynamicCallSiteReference
    implements ClassDecoratorSite {

  /**
   * Creates the site.
   *
   * @param ref The declared target type of the application.
   * @param pc The program counter of the application.
   */
  public ClassDecoratorCallSiteReference(TypeReference ref, int pc) {
    super(ref, pc);
  }
}
