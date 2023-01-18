package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertNotNull;

import java.io.IOException;

import org.junit.Test;

import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;

public class TestRaffi extends TestPythonCallGraphShape {

	@Test
	public void testCode() throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
		CallGraph CG = process("raffi.py");
		assertNotNull(CG);
	}
}
