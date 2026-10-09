package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.ir.ssa.AstLexicalAccess.Access;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.SSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import org.junit.Test;

/**
 * A module under a package directory reads the builtins bound in its module scope (wala/ML#1026): a
 * function's calls to {@code print} and {@code isinstance} each reach the builtin. The script's
 * class is named after the script's path, so a module in a directory carries a slash in its name as
 * a function nested under a script does; a test of the module body by the absence of a slash took
 * such a module for a function and refined its builtin reads to nothing, which {@link TestPrint},
 * over a script at the root, could not see.
 */
public class TestPrintInPackage extends TestJythonCallGraphShape {

  /** The builtins the function calls, by the variable name each is read under. */
  private static final Set<String> BUILTINS = Set.of("print", "isinstance");

  /**
   * Each builtin call in the function under the package directory reaches its builtin's node.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBuiltinsReachedFromPackageModule()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // Loaded under its project root as a Python path entry, so the module's name is its path
    // within the project, `pkg/report.py`, as a project's modules are named.
    PythonAnalysisEngine<?> engine =
        makeEngine(
            List.of(new File(TestPrintInPackage.class.getResource("/print_proj").getPath())),
            "print_proj/pkg/report.py");
    SSAPropagationCallGraphBuilder builder =
        (SSAPropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());

    // The script's class is named after the script's path, so the function's node is found by its
    // name's suffix; the module's own name must carry the directory for this test to exercise it.
    List<CGNode> nodes = new ArrayList<>();
    List<String> scripts = new ArrayList<>();
    for (CGNode node : CG) {
      String name = node.getMethod().getDeclaringClass().getName().toString();
      if (name.startsWith("Lscript ")) scripts.add(name);
      if (name.endsWith("report.py/f")) nodes.add(node);
    }
    assertEquals("One node for f among " + scripts, 1, nodes.size());
    CGNode fNode = nodes.get(0);
    assertTrue(
        "The module is under a directory: " + scripts,
        fNode.getMethod().getDeclaringClass().getName().toString().endsWith("/report.py/f"));

    Set<String> reached = new HashSet<>();

    for (Iterator<SSAInstruction> iit = fNode.getIR().iterateNormalInstructions();
        iit.hasNext(); ) {
      SSAInstruction instruction = iit.next();

      if (instruction instanceof PythonInvokeInstruction invokeInstruction
          && fNode.getDU().getDef(invokeInstruction.getReceiver()) instanceof AstLexicalRead read
          && read.getAccessCount() == 1) {
        Access access = read.getAccess(0);
        String name = access.variableName();

        if (BUILTINS.contains(name)) {
          TypeReference builtin =
              TypeReference.findOrCreate(PythonTypes.pythonLoader, "Lwala/builtin/" + name);

          for (CGNode target : CG.getPossibleTargets(fNode, invokeInstruction.getCallSite()))
            if (target.getMethod().getDeclaringClass().getReference().equals(builtin))
              reached.add(name);
        }
      }
    }

    assertEquals("Each builtin call reaches its builtin.", BUILTINS, reached);
  }
}
