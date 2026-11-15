package com.ibm.wala.cast.python.loader;

/**
 * A class-body field declared with an annotation, {@code name: T} or {@code name: T = value}: one
 * of the fields a dataclass's generated initializer takes, in declaration order (<a
 * href="https://github.com/wala/ML/issues/205">wala/ML#205</a>). A plain assignment in the class
 * body declares a class attribute, not such a field.
 */
public interface AnnotatedFieldEntity {}
