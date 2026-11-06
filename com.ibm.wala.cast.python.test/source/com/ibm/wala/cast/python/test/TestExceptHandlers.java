package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertEquals;

import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import org.junit.Test;

/**
 * Tests that an {@code except} clause is reached from what may raise in its {@code try} body,
 * whatever the clause names (<a href="https://github.com/wala/ML/issues/1005">wala/ML#1005</a>).
 */
public class TestExceptHandlers extends TestJythonCallGraphShape {

  /**
   * A call in each handler of {@code except_handlers1.py} is reached: a handler naming a type, one
   * binding it with {@code as}, one naming a tuple of types, one naming {@code Exception}, and a
   * bare one; the handler of an enclosing {@code try} when the inner one names another type; the
   * code after a {@code try} whose body always raises; a handler of a {@code raise} of a value
   * rather than of a call; and a handler naming a type inside a function.
   */
  @Test
  public void testHandlersReached()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph CG = process("except_handlers1.py");
    List<String> callees =
        List.of(
            "typed",
            "typed_as",
            "typed_tuple",
            "broad",
            "bare",
            "nested_outer",
            "after",
            "raised_directly",
            "in_function");
    Set<String> unreached = new TreeSet<>();
    for (String callee : callees)
      if (getNodes(CG, "script except_handlers1.py/" + callee).isEmpty()) unreached.add(callee);
    assertEquals(Set.of(), unreached);
  }
}
