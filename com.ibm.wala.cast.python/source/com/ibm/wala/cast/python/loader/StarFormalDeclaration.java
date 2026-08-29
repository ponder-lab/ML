package com.ibm.wala.cast.python.loader;

/**
 * A function declaration, an entity or the method built from it, that reports how many of its
 * trailing formals cannot receive a positional default: the {@code *args} and {@code **kwargs}
 * formals, and any keyword-only parameters.
 *
 * <p>Python's positional defaults apply to the LAST parameters of the plain positional list, but
 * the parser appends those trailing formals to the same argument array. Anything that locates the
 * defaulted range by counting back from the end of that array therefore overshoots by exactly this
 * count, binding each default to the parameter after the one it belongs to and leaving the first
 * defaulted parameter with nothing. That is a wrong constant flowing rather than no constant, so a
 * consumer reads a resolved value contradicting the source (wala/ML#843).
 *
 * <p>Implemented by the parser's function entities and by the methods the loader builds from them.
 * A declaration that does not implement it is treated as declaring none, which is the correct
 * reading for a function whose formals are all plain positionals.
 */
public interface StarFormalDeclaration {

  /**
   * The number of trailing formals that cannot receive a positional default.
   *
   * @return The count, zero when every formal is a plain positional parameter.
   */
  int getNumberOfTrailingNonDefaultableParameters();
}
