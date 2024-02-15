package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;

import com.ibm.wala.cast.ipa.callgraph.CAstCallGraphUtil;
import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.SSAContextInterpreter;
import com.ibm.wala.ipa.callgraph.propagation.SSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Collection;
import java.util.Iterator;
import java.util.logging.Logger;
import org.junit.Test;

public class TestClasses extends TestPythonCallGraphShape {

  private static final Logger LOGGER = Logger.getLogger(TestClasses.class.getName());

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
    CallGraph CG = process("classes1.py");
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
    CAstCallGraphUtil.AVOID_DUMP = false;
    CAstCallGraphUtil.dumpCG(
        (SSAContextInterpreter) builder.getContextInterpreter(), builder.getPointerAnalysis(), CG);
    verifyGraphAssertions(CG, assertionsClasses3);
  }

  protected static final Object[][] externalClassAssertions =
      new Object[][] {
        new Object[] {ROOT, new String[] {"script client.py"}},
        new Object[] {
          "script client.py",
          new String[] {
            "script client.py/f",
          }
        },
        // TODO: Re-add once https://github.com/wala/ML/issues/146 is fixed.
        /*
        new Object[] {
          "script client.py/f",
          new String[] {
            "script client.py/C",
          }
        }
        */
      };

  /** Can we find a class (and initialize it) from another file? */
  @Test
  public void testExternalClass()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    CallGraph callGraph = this.process("client.py", "classes4.py");

    verifyGraphAssertions(callGraph, externalClassAssertions);

    Collection<CGNode> nodes = this.getNodes(callGraph, "script client.py/f");
    assertEquals(1, nodes.size());
    CGNode f = nodes.iterator().next();

    Iterator<CGNode> succNodes = callGraph.getSuccNodes(f);
    // TODO: Change to assertTrue() once https://github.com/wala/ML/issues/146 is fixed.
    assertFalse(succNodes.hasNext());

    CGNode node =
        f; // Change to succNodes.next() once https://github.com/wala/ML/issues/146 is fixed.
    assertFalse("Expecting only one callee.", succNodes.hasNext());

    IMethod method = node.getMethod();
    IClass declaringClass = method.getDeclaringClass();
    TypeName name = declaringClass.getName();

    // TODO: Change to assertEquals() once https://github.com/wala/ML/issues/146 is fixed.
    assertNotEquals("Lscript classes4.py/C", name.toString());
  }
}
