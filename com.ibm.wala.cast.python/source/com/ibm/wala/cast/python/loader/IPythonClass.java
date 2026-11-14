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

  /**
   * The names of this class's bases as declared, in order, each resolved to the class it names in
   * this unit or to the summary class shell a missing base matches, so a walk over them in method
   * resolution order reaches every base's methods, not only the single superclass the class model
   * records (<a href="https://github.com/wala/ML/issues/1006">wala/ML#1006</a>). A class whose
   * bases were not recorded answers with its superclass alone: the loader's classes record them,
   * and the engine's synthesized Python classes take this default.
   *
   * @return The base names, in declaration order; empty for a class with no base.
   */
  default java.util.List<com.ibm.wala.types.TypeName> getBaseTypeNames() {
    IClass superclass = getSuperclass();
    return superclass == null
        ? java.util.Collections.emptyList()
        : java.util.Collections.singletonList(superclass.getName());
  }

  /**
   * The names of the bases this class declares that resolve to no class of this unit and no summary
   * class shell, and so are missing from {@link #getBaseTypeNames()}: a library class no summary
   * models, such as {@code types.SimpleNamespace}, or a base whose name mis-resolves. Their
   * methods, an initializer among them, are not in the analysis. A class whose bases were not
   * recorded answers with none.
   *
   * @return The unresolved bases' names, as the class statement writes them.
   */
  default java.util.Set<String> getUnresolvedBaseNames() {
    return java.util.Collections.emptySet();
  }
}
