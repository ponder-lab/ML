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
   *
   * <p>The count is <strong>thread-scoped</strong>. An analysis run both records misses (during
   * shape evaluation, via {@code interpretAsInt}) and reports them (at the end of {@code
   * performAnalysis}) on the same thread, so a {@link ThreadLocal} gives each concurrent run its
   * own count. A single process-global counter would let one run's {@link
   * #getAndResetInterpreterUnavailableMisses()} clear misses belonging to another in-progress run,
   * or attribute one run's misses to another's summary.
   */
  private static final ThreadLocal<java.util.concurrent.atomic.AtomicInteger>
      interpreterUnavailableMisses =
          ThreadLocal.withInitial(java.util.concurrent.atomic.AtomicInteger::new);

  /**
   * Records, for the current thread, that an expression could not be evaluated because the
   * interpreter is unavailable.
   */
  public static void recordInterpreterUnavailableMiss() {
    interpreterUnavailableMisses.get().incrementAndGet();
  }

  /**
   * Returns the current thread's interpreter-unavailable miss count and resets it to zero. The
   * count is thread-scoped (see {@link #interpreterUnavailableMisses}), so this reflects only the
   * misses recorded on the calling thread — i.e. the current analysis run — and cannot disturb a
   * concurrent run on another thread. See <a
   * href="https://github.com/wala/ML/issues/444">wala/ML#444</a>.
   *
   * @return The calling thread's interpreter-unavailable miss count immediately before the reset.
   */
  public static int getAndResetInterpreterUnavailableMisses() {
    return interpreterUnavailableMisses.get().getAndSet(0);
  }
}
