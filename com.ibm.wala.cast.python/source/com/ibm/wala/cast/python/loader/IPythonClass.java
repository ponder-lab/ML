package com.ibm.wala.cast.python.loader;

import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeReference;
import java.util.Collection;

/**
 * An interface for Python classes that provides access to their methods and inner types. This
 * allows both standard Python classes and summarized synthetic classes to be treated uniformly by
 * call graph selectors and trampoline generators.
 */
public interface IPythonClass extends IClass {

  /** Returns references to the methods defined in this class. */
  Collection<MethodReference> getMethodReferences();

  /** Returns references to the inner types defined in this class. */
  Collection<TypeReference> getInnerReferences();
}
