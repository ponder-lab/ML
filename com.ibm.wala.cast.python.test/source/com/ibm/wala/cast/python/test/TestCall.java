package com.ibm.wala.cast.python.test;

import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import org.junit.Test;

public class TestCall extends TestJythonCallGraphShape {
  @Test
  public void testCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph CG = process("test_call.py");
    System.err.println(CG);
  }
}
