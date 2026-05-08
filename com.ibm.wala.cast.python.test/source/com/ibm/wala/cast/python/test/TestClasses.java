package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.ipa.callgraph.CAstCallGraphUtil;
import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.SSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Collection;
import java.util.Iterator;
import org.junit.Test;

public class TestClasses extends TestJythonCallGraphShape {

  protected static final Object[][] assertionsClasses1 =
      new Object[][] {
        new Object[] {ROOT, new String[] {"script classes1.py"}},
        new Object[] {
          "script classes1.py",
          new String[] {
            "script classes1.py/Outer",
            "$script classes1.py/Outer/foo:trampoline2",
            "script classes1.py/Outer/Inner",
            "$script classes1.py/Outer/Inner/foo:trampoline2"
          }
        },
        new Object[] {
          "$script classes1.py/Outer/foo:trampoline2", new String[] {"script classes1.py/Outer/foo"}
        },
        new Object[] {
          "$script classes1.py/Outer/Inner/foo:trampoline2",
          new String[] {"script classes1.py/Outer/Inner/foo"}
        }
      };

  @Test
  public void testClasses1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("classes1.py");
    SSAPropagationCallGraphBuilder builder =
        (SSAPropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());

    CAstCallGraphUtil.AVOID_DUMP.set(false);
    CAstCallGraphUtil.dumpCG(builder.getCFAContextInterpreter(), builder.getPointerAnalysis(), CG);
    System.err.println("Call graph:\n" + CG);

    verifyGraphAssertions(CG, assertionsClasses1);
  }

  protected static final Object[][] assertionsClasses2 =
      new Object[][] {
        new Object[] {ROOT, new String[] {"script classes2.py"}},
        new Object[] {
          "script classes2.py",
          new String[] {
            "script classes2.py/fc",
            "script classes2.py/Ctor",
            "$script classes2.py/Ctor/get:trampoline2"
          }
        },
        new Object[] {"script classes2.py/Ctor", new String[] {"script classes2.py/Ctor/__init__"}},
        new Object[] {
          "$script classes2.py/Ctor/get:trampoline2", new String[] {"script classes2.py/Ctor/get"}
        },
        new Object[] {
          "script classes2.py/Ctor/get",
          new String[] {"script classes2.py/fa", "script classes2.py/fb", "script classes2.py/fc"}
        }
      };

  @Test
  public void testClasses2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph CG = process("classes2.py");
    verifyGraphAssertions(CG, assertionsClasses2);
  }

  protected static final Object[][] assertionsClasses3 =
      new Object[][] {
        new Object[] {ROOT, new String[] {"script classes3.py"}},
        new Object[] {
          "script classes3.py",
          new String[] {
            "script classes3.py/Ctor",
            "$script classes3.py/Ctor/get:trampoline2",
            "script classes3.py/SubCtor",
            "script classes3.py/OtherSubCtor"
          }
        },
        new Object[] {
          "script classes3.py",
          new String[] {"script classes3.py/Ctor", "$script classes3.py/Ctor/get:trampoline2"}
        },
        new Object[] {"script classes3.py/Ctor", new String[] {"script classes3.py/Ctor/__init__"}},
        new Object[] {
          "script classes3.py/SubCtor", new String[] {"script classes3.py/SubCtor/__init__"}
        },
        new Object[] {
          "script classes3.py/OtherSubCtor",
          new String[] {"script classes3.py/OtherSubCtor/__init__"}
        },
        new Object[] {
          "script classes3.py/SubCtor/__init__",
          new String[] {"$script classes3.py/Ctor/__init__:trampoline4"}
        },
        new Object[] {
          "$script classes3.py/Ctor/__init__:trampoline4",
          new String[] {"script classes3.py/Ctor/__init__"}
        },
        new Object[] {
          "script classes3.py/OtherSubCtor/__init__",
          new String[] {"$script classes3.py/Ctor/__init__:trampoline4"}
        }
      };

  @Test
  public void testClasses3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<?> engine = makeEngine("classes3.py");
    SSAPropagationCallGraphBuilder builder =
        (SSAPropagationCallGraphBuilder) engine.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    System.err.println(CG);
    verifyGraphAssertions(CG, assertionsClasses3);
  }

  protected static final Object[][] assertionsClasses4 =
      new Object[][] {
        new Object[] {ROOT, new String[] {"script classes4_client.py", "script classes4.py"}},
        new Object[] {
          "script classes4_client.py",
          new String[] {
            "script classes4_client.py/f",
          }
        },
        new Object[] {
          "script classes4_client.py/f",
          new String[] {
            "script classes4.py/C",
          }
        }
      };

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/146">wala/ML#146</a> ("Can't
   * find external classes"). The analyzer should resolve {@code C}'s constructor from {@code
   * classes4_client.py} even though {@code C} is defined in a separate module {@code classes4.py}.
   *
   * <p>This test originally documented the bug as a known failure; the bug has since been fixed by
   * a separate change on master, so this is now a positive regression test. The split-module
   * fixture ({@code classes4_client.py} importing from {@code classes4.py}) was renamed from {@code
   * client.py} on this PR's revival to avoid colliding with master's {@code client.py}, which is a
   * different test fixture for <a href="https://github.com/wala/ML/issues/211">wala/ML#211</a>.
   */
  @Test
  public void testExternalClass()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph callGraph = this.process("classes4_client.py", "classes4.py");

    verifyGraphAssertions(callGraph, assertionsClasses4);

    Collection<CGNode> nodes = this.getNodes(callGraph, "script classes4_client.py/f");
    assertEquals(1, nodes.size());
    CGNode f = nodes.iterator().next();

    Iterator<CGNode> succNodes = callGraph.getSuccNodes(f);
    assertTrue(succNodes.hasNext());

    CGNode node = succNodes.next();
    assertFalse("Expecting only one callee.", succNodes.hasNext());

    IMethod method = node.getMethod();
    IClass declaringClass = method.getDeclaringClass();
    TypeName name = declaringClass.getName();

    assertEquals("Lscript classes4.py/C", name.toString());
  }
}
