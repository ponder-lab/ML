package com.ibm.wala.cast.python.test;

import static com.ibm.wala.cast.python.types.PythonTypes.isContainer;
import static com.ibm.wala.cast.python.types.PythonTypes.isSequence;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.types.PythonTypes;
import org.junit.Test;

/**
 * Tests {@link PythonTypes#isContainer} and {@link PythonTypes#isSequence}, the analysis's one
 * authority on which built-in types hold elements and which of those index them by position.
 */
public class TestPythonTypesContainers {

  /** A list and a tuple are containers and sequences. */
  @Test
  public void testSequences() {
    assertTrue(isContainer(PythonTypes.list));
    assertTrue(isSequence(PythonTypes.list));
    assertTrue(isContainer(PythonTypes.tuple));
    assertTrue(isSequence(PythonTypes.tuple));
  }

  /** A set, a dict and an enumerate object are containers, but not sequences. */
  @Test
  public void testNonSequenceContainers() {
    assertTrue(isContainer(PythonTypes.set));
    assertFalse(isSequence(PythonTypes.set));
    assertTrue(isContainer(PythonTypes.dict));
    assertFalse(isSequence(PythonTypes.dict));
    assertTrue(isContainer(PythonTypes.enumerate));
    assertFalse(isSequence(PythonTypes.enumerate));
  }

  /**
   * A string, which Python iterates, and the builtins that produce containers are not containers
   * themselves, nor is a generic object.
   */
  @Test
  public void testNonContainers() {
    assertFalse(isContainer(PythonTypes.string));
    assertFalse(isSequence(PythonTypes.string));
    assertFalse(isContainer(PythonTypes.ENUMERATE_BUILTIN));
    assertFalse(isContainer(PythonTypes.object));
    assertFalse(isContainer(PythonTypes.iterator));
  }
}
