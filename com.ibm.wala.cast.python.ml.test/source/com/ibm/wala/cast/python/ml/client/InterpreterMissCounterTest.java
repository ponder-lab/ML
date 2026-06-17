package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;

import com.ibm.wala.cast.python.util.PythonInterpreter;
import org.junit.After;
import org.junit.Test;

/**
 * Unit tests for the interpreter-unavailable miss counter and its end-of-analysis summary (<a
 * href="https://github.com/wala/ML/issues/444">wala/ML#444</a>). The counter is thread-scoped, so
 * each test method (which runs on a single thread) is isolated from misses recorded by concurrent
 * analyses on other threads; {@link #clear()} drains the thread's count in case the executor reuses
 * the thread for a later test.
 */
public class InterpreterMissCounterTest {

  @After
  public void clear() {
    PythonInterpreter.getAndResetInterpreterUnavailableMisses();
  }

  @Test
  public void testRecordThenGetAndReset() {
    PythonInterpreter.getAndResetInterpreterUnavailableMisses();
    PythonInterpreter.recordInterpreterUnavailableMiss();
    PythonInterpreter.recordInterpreterUnavailableMiss();
    PythonInterpreter.recordInterpreterUnavailableMiss();
    assertEquals(3, PythonInterpreter.getAndResetInterpreterUnavailableMisses());
    assertEquals(0, PythonInterpreter.getAndResetInterpreterUnavailableMisses());
  }

  @Test
  public void testReportWithMissesResetsCount() {
    PythonInterpreter.getAndResetInterpreterUnavailableMisses();
    PythonInterpreter.recordInterpreterUnavailableMiss();
    PythonInterpreter.recordInterpreterUnavailableMiss();
    PythonTensorAnalysisEngine.reportAndResetInterpreterUnavailableMisses();
    assertEquals(0, PythonInterpreter.getAndResetInterpreterUnavailableMisses());
  }

  @Test
  public void testReportWithNoMissesIsANoOp() {
    PythonInterpreter.getAndResetInterpreterUnavailableMisses();
    PythonTensorAnalysisEngine.reportAndResetInterpreterUnavailableMisses();
    assertEquals(0, PythonInterpreter.getAndResetInterpreterUnavailableMisses());
  }
}
