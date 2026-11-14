package com.ibm.wala.cast.python.loader;

import java.util.Map;

/**
 * A class entity that carries the keywords its class statement passes, such as {@code metaclass=}
 * (<a href="https://github.com/wala/ML/issues/1260">wala/ML#1260</a>).
 */
public interface ClassKeywordsEntity {

  /**
   * The keywords the class statement passes, in source order, each mapped to the dotted name of its
   * value as the statement writes it. A {@code **} spread has no keyword name and is not recorded.
   *
   * @return The keywords and their values' names.
   */
  Map<String, String> classKeywords();
}
