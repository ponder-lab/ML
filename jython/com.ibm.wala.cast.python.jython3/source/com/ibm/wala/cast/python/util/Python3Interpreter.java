package com.ibm.wala.cast.python.util;

import java.util.logging.Level;
import java.util.logging.Logger;
import org.python.core.PyException;
import org.python.core.PyObject;
import org.python.core.PySystemState;
import org.python.util.PythonInterpreter;

public class Python3Interpreter extends com.ibm.wala.cast.python.util.PythonInterpreter {

  private static final Logger LOGGER = Logger.getLogger(Python3Interpreter.class.getName());

  private static PythonInterpreter interp;

  /**
   * Memoizes a failed Jython init so subsequent {@link #getInterp()} calls return {@code null}
   * cheaply instead of re-running {@code new PythonInterpreter()} (which can be expensive when it
   * fails — site-import walks the Jython resource path on every attempt). When a single failure has
   * occurred, callers receive {@code null} and can degrade their behavior (e.g., {@link
   * com.ibm.wala.cast.python.loader.Python3Loader} skips constant folding).
   */
  private static volatile boolean initFailed = false;

  public static PythonInterpreter getInterp() {
    if (initFailed) return null;
    if (interp == null) {
      try {
        PySystemState.initialize();
        interp = new PythonInterpreter();
      } catch (Throwable t) {
        // Jython init can fail when bootstrap resources (e.g., the embedded
        // _frozen_importlib bytecode used by org.python.core.imp) aren't reachable from the
        // current classloader/working directory. This is environment-dependent (e.g., happens
        // under Tycho-OSGi but not under plain Maven-surefire). Treat as a recoverable failure:
        // log once, memoize, and let callers degrade gracefully.
        initFailed = true;
        LOGGER.log(
            Level.WARNING,
            t,
            () ->
                "Jython interpreter init failed; constant folding will be disabled for this run.");
        return null;
      }
    }
    return interp;
  }

  public Integer evalAsInteger(String expr) {
    try {
      PyObject val = getInterp().eval(expr);
      if (val.isInteger()) {
        return val.asInt();
      } else
        throw new IllegalArgumentException(
            "Python expression: " + expr + " cannot be evaluated to an integer.");
    } catch (PyException e) {
      LOGGER.log(Level.SEVERE, "Unable to interpret Python expression: " + expr, e);
      throw new IllegalArgumentException("Can't interpret Python expression: " + expr + ".", e);
    }
  }
}
