package com.ibm.wala.cast.python.ipa.callgraph;

import com.ibm.wala.types.TypeReference;

/** The call site applying a bare decorator to a class; see {@link ClassDecoratorSite}. */
public class BareClassDecoratorCallSiteReference extends BareDecoratorCallSiteReference
    implements ClassDecoratorSite {

  /**
   * Creates the site.
   *
   * @param ref The declared target type of the application.
   * @param pc The program counter of the application.
   */
  public BareClassDecoratorCallSiteReference(TypeReference ref, int pc) {
    super(ref, pc);
  }
}
