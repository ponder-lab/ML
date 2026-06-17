package com.ibm.wala.cast.python.util;

import java.lang.reflect.InvocationTargetException;

public abstract class PythonInterpreter {

  static {
    try {
      @SuppressWarnings("unchecked")
      Class<PythonInterpreter> i4 =
          (Class<PythonInterpreter>)
              Class.forName("com.ibm.wala.cast.python.jep.CPythonInterpreter");
      setInterpreter(i4.getDeclaredConstructor().newInstance());
    } catch (ClassNotFoundException
        | InstantiationException
        | IllegalAccessException
        | UnsatisfiedLinkError
        | IllegalArgumentException
        | InvocationTargetException
        | NoSuchMethodException
        | SecurityException e2) {
      try {
        Class<?> i3 = Class.forName("com.ibm.wala.cast.python.util.Python3Interpreter");
        setInterpreter((PythonInterpreter) i3.getDeclaredConstructor().newInstance());
      } catch (ClassNotFoundException
          | InstantiationException
          | IllegalAccessException
          | IllegalArgumentException
          | InvocationTargetException
          | NoSuchMethodException
          | SecurityException e) {
        try {
          Class<?> i2 = Class.forName("com.ibm.wala.cast.python.util.Python2Interpreter");
          setInterpreter((PythonInterpreter) i2.getDeclaredConstructor().newInstance());
        } catch (ClassNotFoundException
            | InstantiationException
            | IllegalAccessException
            | IllegalArgumentException
            | InvocationTargetException
            | NoSuchMethodException
            | SecurityException e1) {
          assert false : e.getMessage() + ", then " + e1.getMessage();
        }
      }
    }
  }

  public abstract Integer evalAsInteger(String expr);

  private static PythonInterpreter interp;

  public static void setInterpreter(PythonInterpreter interpreter) {
    interp = interpreter;
  }

  public static Integer interpretAsInt(String expr) {
    return interp.evalAsInteger(expr);
  }

  /**
   * Counts expressions that could not be evaluated specifically because the interpreter was
   * unavailable (Jython init failed), as distinct from expressions the interpreter is up for but
   * genuinely cannot reduce to a constant. Lets a client emit an end-of-analysis summary of the
   * precision lost to interpreter unavailability rather than relying on a single early warning that
   * is easily lost in build noise. See <a
   * href="https://github.com/wala/ML/issues/444">wala/ML#444</a>.
   */
  private static final java.util.concurrent.atomic.AtomicInteger interpreterUnavailableMisses =
      new java.util.concurrent.atomic.AtomicInteger();

  /** Records that an expression could not be evaluated because the interpreter is unavailable. */
  public static void recordInterpreterUnavailableMiss() {
    interpreterUnavailableMisses.incrementAndGet();
  }

  /**
   * Atomically returns the current interpreter-unavailable miss count and resets it to zero.
   * Reading and resetting in one operation means a concurrent {@link
   * #recordInterpreterUnavailableMiss()} can't be lost between a separate read and reset, which
   * would otherwise undercount the next run. See <a
   * href="https://github.com/wala/ML/issues/444">wala/ML#444</a>.
   *
   * @return The interpreter-unavailable miss count immediately before the reset.
   */
  public static int getAndResetInterpreterUnavailableMisses() {
    return interpreterUnavailableMisses.getAndSet(0);
  }
}
