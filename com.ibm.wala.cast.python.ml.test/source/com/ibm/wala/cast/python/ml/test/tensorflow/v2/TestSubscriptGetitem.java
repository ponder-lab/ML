package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.loader.DynamicCallSiteReference;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.collections.HashSetFactory;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import org.junit.Test;

/**
 * Tests that a subscript of an object whose class defines {@code __getitem__} calls the method
 * (wala/ML#1306). Before, the subscript read a field of the object and the method had no caller.
 */
public class TestSubscriptGetitem extends AbstractTensorTest {

  private static final String FILE = "tf2_test_subscript_getitem.py";

  /** The subscript's value is what the method returns. */
  @Test
  public void testSubscript() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_subscript", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** The controls: a call of another method, and a call of the method by its name. */
  @Test
  public void testCalls() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_method", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FILE, "by_explicit_call", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** The control: a subscript of a list reads its element, as before. */
  @Test
  public void testList() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_list", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** The method's body is reached. */
  @Test
  public void testBody() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "in_body", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** A tuple the method returns is unpacked member by member. */
  @Test
  public void testUnpacking() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_unpacking", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FILE, "by_unpacking_other", 1, 1, Map.of(2, Set.of(TENSOR_2_INT64)));
  }

  /** A variable made from the subscript has its shape and dtype. */
  @Test
  public void testVariable() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "variable_of_subscript", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** A method a base class defines is called. */
  @Test
  public void testInherited() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_inherited", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** Two instances of one class each give their own value. */
  @Test
  public void testPerInstance() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "per_instance", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FILE, "per_instance_other", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /** A constant key reaches the method's parameter, where it selects a member. */
  @Test
  public void testConstantKey() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_index", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FILE, "by_index_other", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test(FILE, "by_string_key", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /** A key that is a tensor reaches the method's parameter. */
  @Test
  public void testTensorKey() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "key_in_body", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /** A subscript inside a function of an object passed to it calls the method. */
  @Test
  public void testInFunction() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "in_function", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** A method that subscripts another such object calls that object's method. */
  @Test
  public void testNested() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_wrapper", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** The control: a constant index passed to another method selects a member there too. */
  @Test
  public void testConstantKeyControl()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_index_method", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FILE, "by_index_method_other", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /** A subscript of {@code self} inside another method of the class calls the method. */
  @Test
  public void testInMethod() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "in_method", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** A subscript inside a loop calls the method. */
  @Test
  public void testInLoop() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "in_loop", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** A subscript inside a comprehension calls the method. */
  @Test
  public void testInComprehension() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "in_comprehension", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** A subclass of a library class that defines the method has it called. */
  @Test
  public void testLibrarySubclass() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "of_sequence", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /**
   * A receiver that is such an object at one call and a list at another gives the method's value
   * and the list's element.
   */
  @Test
  public void testEitherReceiver() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "of_either", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32, TENSOR_3_INT32)));
  }

  /**
   * Each subscript that calls the method has a node for the call, in the context of the node that
   * holds the subscript, and a subscript of a list, a tuple or a dictionary has none. The node
   * calls one method, or two at the subscript whose receiver is an instance of either of two
   * classes. The edge to the node is at a call site of the read's own index, which the caller's
   * instructions do not list as a call site.
   */
  @Test
  public void testCallGraph() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph callGraph = builder.makeCallGraph(builder.getOptions());
    String script = "Lscript " + FILE;
    Set<IMethod> calls = HashSetFactory.make();
    Set<String> callers = new TreeSet<>();
    int callingTwo = 0;
    for (CGNode node : callGraph) {
      String method = node.getMethod().getName().toString();
      if (!method.startsWith("__getitem__@")) continue;
      calls.add(node.getMethod());
      int index = Integer.parseInt(method.substring(method.indexOf('@') + 1, method.indexOf(' ')));
      int predecessors = 0;
      for (Iterator<CGNode> it = callGraph.getPredNodes(node); it.hasNext(); predecessors++) {
        CGNode caller = it.next();
        assertEquals(caller.getContext(), node.getContext());
        assertTrue(
            node.toString(),
            callGraph
                .getPossibleTargets(
                    caller, new DynamicCallSiteReference(PythonTypes.CodeBody, index))
                .contains(node));
        assertFalse(node.toString(), callGraph.getPossibleSites(caller, node).hasNext());
        String name = caller.getMethod().getDeclaringClass().getName().toString();
        assertEquals(name, script, name.substring(0, script.length()));
        callers.add(name.substring(script.length()));
      }
      assertEquals(node.toString(), 1, predecessors);
      if (callGraph.getSuccNodeCount(node) == 2) callingTwo++;
      else assertEquals(node.toString(), 1, callGraph.getSuccNodeCount(node));
    }
    assertEquals(1, callingTwo);
    assertEquals(
        Set.of(
            "",
            "/Reader/first",
            "/Subscripts/call",
            "/Wrapper/__getitem__",
            "/comprehension5",
            "/read",
            "/read_either"),
        callers);
    // The script's 17 subscripts of such objects, and one in each other caller.
    assertEquals(23, calls.size());
  }

  /** A subscript of such an object by a tuple of keys. */
  @Test
  public void testTupleKey() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_tuple_key", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /**
   * A subscript of such an object by a string that names one of the object's attributes reads the
   * attribute beside what the method returns.
   *
   * <p>TODO: Expect {@link #TENSOR_2_4_FLOAT32} alone, which the method returns, once wala/ML#1311
   * is fixed.
   */
  @Test
  public void testAttributeName() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_attribute_name", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32, TENSOR_3_INT32)));
  }

  /**
   * A subscript of an instance of a class that extends {@code dict} and hands the key on to it: the
   * base class's method has no body, and the entry is read from the instance.
   */
  @Test
  public void testDictSubclass() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_dict_subclass", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** A subscript of a string constant, which holds no method and is no tensor. */
  @Test
  public void testConstantReceiver() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_constant_receiver", 0, 0);
  }

  /**
   * A subscript of an instance of a class that extends {@code dict} and declares another method and
   * no {@code __getitem__}: the entry is read from the instance, and {@link #testCallGraph} pins
   * that nothing is called.
   */
  @Test
  public void testPlainDictSubclass() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_plain_dict_subclass", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /**
   * A layer called inside a {@code __getitem__} that a model's {@code call} subscripts takes the
   * model's {@code training}, since the method runs in the call context of the node that holds the
   * subscript.
   */
  @Test
  public void testTrainingContext() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph callGraph = builder.makeCallGraph(builder.getOptions());
    Set<Object> values = HashSetFactory.make();
    for (CGNode node : callGraph) {
      String type = node.getMethod().getDeclaringClass().getName().toString();
      if (!type.equals("Lscript " + FILE + "/Noted/call")) continue;
      for (int parameter : node.getIR().getParameterValueNumbers()) {
        String[] names = node.getIR().getLocalNames(0, parameter);
        if (names == null || !Arrays.asList(names).contains("training")) continue;
        PointerKey key =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, parameter);
        for (InstanceKey value : builder.getPointerAnalysis().getPointsToSet(key))
          values.add(
              value instanceof ConstantKey<?> constant && constant.getValue() != null
                  ? constant.getValue()
                  : value.toString());
      }
    }
    assertEquals(Set.of(Boolean.TRUE), values);
  }

  /**
   * A subscript whose receiver is an instance of one class at one pass of a loop and of another at
   * the next: both classes' methods are called, and both values are read.
   */
  @Test
  public void testTwoClasses() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "of_two_classes",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_4_FLOAT32, TensorType.of(INT_64, 2, 4))));
  }

  /**
   * A loop over such an object, whose class defines no {@code __iter__}, does not call the method,
   * so its variable is no tensor.
   *
   * <p>TODO: Expect {@link #TENSOR_2_4_FLOAT32}, which the method returns, once wala/ML#1310 is
   * fixed.
   */
  @Test
  public void testIteration() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_iteration", 0, 0);
  }

  /**
   * A slice of such an object does not call the method: it is read as the object itself, which has
   * no shape and no dtype.
   *
   * <p>TODO: Expect {@link #TENSOR_2_4_FLOAT32}, which the method returns, once wala/ML#1310 is
   * fixed.
   */
  @Test
  public void testSlice() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_slice", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }
}
