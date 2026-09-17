package com.ibm.wala.cast.python.parser;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;

/**
 * The modules each script's classes take their bases from, recorded by the parser at parse time and
 * read by the loader before translation, so that a base's module is translated before the module of
 * the class that extends it (wala/ML#944). Keyed by the type dictionary the parsers of one analysis
 * share, so concurrent analyses do not see each other's edges.
 *
 * <p>A dependency is a dotted module path as the import binding spells it (for example {@code
 * pkg.mod} for {@code from pkg.mod import Base; class X(Base)}, or every wildcard source of the
 * module for a bare base name no explicit binding covers). The loader resolves it against the
 * script names in scope by path suffix, the same match the importer uses.
 */
public final class BaseDependencies {

  private static final Map<Object, Map<String, Set<String>>> DEPENDENCIES =
      Collections.synchronizedMap(new WeakHashMap<>());

  private BaseDependencies() {}

  /**
   * Records that the given script has a class whose base comes from the given module.
   *
   * @param key The analysis's shared type dictionary.
   * @param script The script name, as the parser spells it.
   * @param module The dotted module path the base is bound to.
   */
  public static void note(Object key, String script, String module) {
    DEPENDENCIES
        .computeIfAbsent(key, k -> Collections.synchronizedMap(new LinkedHashMap<>()))
        .computeIfAbsent(script, k -> Collections.synchronizedSet(new LinkedHashSet<>()))
        .add(module);
  }

  /**
   * Returns the recorded dependencies of every script of the given analysis.
   *
   * @param key The analysis's shared type dictionary.
   * @return Script name to the dotted module paths its classes take bases from; empty if none.
   */
  public static Map<String, Set<String>> get(Object key) {
    Map<String, Set<String>> m = DEPENDENCIES.get(key);
    return m == null ? Collections.emptyMap() : m;
  }
}
