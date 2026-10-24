package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.CallTargets;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.client.TensorFlowApis;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import java.io.File;
import java.util.Collections;
import java.util.Set;
import java.util.TreeSet;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Tests {@link TensorFlowApis#of}, the client-facing authority on which TensorFlow API a call
 * invokes. Each function of the fixture makes one call of interest, and a test reads the APIs the
 * calls in that function invoke.
 */
public class TestTensorFlowApis extends TestPythonMLCallGraphShape {

  private static PythonSSAPropagationCallGraphBuilder builder;

  private static CallGraph callGraph;

  @BeforeClass
  public static void analyze() throws Exception {
    PythonTensorAnalysisEngine engine =
        new TestTensorFlowApis()
            .makeEngine(Collections.<File>emptyList(), "tf2_test_tensorflow_apis.py");
    builder = engine.defaultCallGraphBuilder();
    callGraph = builder.makeCallGraph(builder.getOptions());
  }

  /**
   * The APIs the calls of a fixture function invoke, across every node of the function.
   *
   * @param function The function's name.
   * @return The APIs' names.
   */
  private static Set<String> apisIn(String function) {
    Set<String> ret = new TreeSet<>();
    boolean found = false;
    for (CGNode node : callGraph) {
      String name = node.getMethod().getDeclaringClass().getName().toString();
      // Only program code: an instance-method trampoline is named like the method it forwards to.
      if (!CallTargets.isProgramCode(node)
          || !name.endsWith("/" + function)
          || node.getIR() == null) continue;
      found = true;
      for (SSAInstruction instruction : node.getIR().getInstructions())
        if (instruction instanceof SSAAbstractInvokeInstruction call)
          ret.addAll(TensorFlowApis.of(builder, node, call));
    }
    assertFalse("`" + function + "` must be in the call graph.", !found);
    return ret;
  }

  @Test
  public void testDirectReshape() {
    assertEquals(Set.of("tensorflow.reshape"), apisIn("direct_reshape"));
  }

  @Test
  public void testDirectShape() {
    assertEquals(Set.of("tensorflow.shape"), apisIn("direct_shape"));
  }

  @Test
  public void testDirectZeros() {
    assertEquals(Set.of("tensorflow.zeros"), apisIn("direct_zeros"));
  }

  @Test
  public void testDirectRandomNormal() {
    assertEquals(Set.of("tensorflow.random.normal"), apisIn("direct_random_normal"));
  }

  /** An API imported under another name, {@code from tensorflow import reshape as rs}. */
  @Test
  public void testAliasedImport() {
    assertEquals(Set.of("tensorflow.reshape"), apisIn("aliased_reshape"));
  }

  /** A method of a tensor, {@code t.set_shape(...)}. */
  @Test
  public void testMethodCall() {
    assertEquals(Set.of("tensorflow.Tensor.set_shape"), apisIn("method_set_shape"));
  }

  @Test
  public void testKerasBackend() {
    assertEquals(Set.of("tensorflow.keras.backend.floatx"), apisIn("keras_backend"));
  }

  /** A callee that is one of two APIs, {@code (tf.zeros if c else tf.ones)(...)}: both. */
  @Test
  public void testConditionalCallee() {
    assertEquals(Set.of("tensorflow.ones", "tensorflow.zeros"), apisIn("either"));
  }

  /** A call of a program function is no API call, whatever the function calls. */
  @Test
  public void testUserFunction() {
    assertEquals(Set.of(), apisIn("user_function"));
  }

  /** A call with no call-graph target invokes no modeled API. */
  @Test
  public void testUnresolvedCall() {
    assertEquals(Set.of(), apisIn("unresolved"));
  }

  /** The module imported under another name, {@code import tensorflow as T; T.shape(x)}. */
  @Test
  public void testModuleAlias() {
    assertEquals(Set.of("tensorflow.shape"), apisIn("module_alias"));
  }

  /** An API bound to a local first, {@code s = tf.shape; s(x)}. */
  @Test
  public void testReboundLocal() {
    assertEquals(Set.of("tensorflow.shape"), apisIn("rebound_local"));
  }

  /**
   * An API bound to a module global and called from a function, {@code S = tf.shape} then {@code
   * S(x)}: the calling function reads no attribute chain, so only the call's target names it.
   */
  @Test
  public void testReboundGlobal() {
    assertEquals(Set.of("tensorflow.shape"), apisIn("rebound_global"));
  }

  /**
   * An API modeled only by a summary it shares with others, {@code tf.image.resize}: named by the
   * callee's attribute chain.
   */
  @Test
  public void testSharedSummaryByAttributeChain() {
    assertEquals(Set.of("tensorflow.image.resize"), apisIn("image_resize"));
  }

  /** A Keras layer applied, {@code layer(x)} with a {@code Dense} layer. */
  @Test
  public void testKerasLayerCall() {
    assertEquals(Set.of("tensorflow.keras.layers.Dense.__call__"), apisIn("dense_call"));
  }

  /** A Keras layer the table does not list, named by its summary's class path. */
  @Test
  public void testKerasLayerCallByClassPath() {
    assertEquals(Set.of("tensorflow.keras.layers.Dropout.__call__"), apisIn("dropout_call"));
  }

  /**
   * A program function stored on the module object, {@code tf.myfn = helper; tf.myfn(x)}: the call
   * runs program code, so its attribute chain names nothing.
   */
  @Test
  public void testProgramFunctionStoredOnModule() {
    assertEquals(Set.of(), apisIn("stored_on_module"));
  }

  /**
   * An aliased API, {@code tf.math.sigmoid}: its canonical name, {@code tensorflow.nn.sigmoid}, and
   * the spelling at the call site.
   */
  @Test
  public void testAliasSpelling() {
    assertEquals(
        Set.of("tensorflow.math.sigmoid", "tensorflow.nn.sigmoid"), apisIn("aliased_sigmoid"));
  }

  /** A Keras application's model applied: its class is shared by every application, no API. */
  @Test
  public void testApplicationCall() {
    assertEquals(Set.of(), apisIn("apply_application"));
  }

  /**
   * {@code tf.keras.layers.serialize}, whose summary class several modules share: named by its
   * attribute chain, not by the shared class's path.
   */
  @Test
  public void testSharedSerializationClass() {
    assertEquals(Set.of("tensorflow.keras.layers.serialize"), apisIn("serialize_layer"));
  }

  /**
   * {@code add_weight} called through a bound method, {@code self.add_weight(...)}, and directly,
   * {@code tf.keras.Model.add_weight(m, ...)}: one name.
   */
  @Test
  public void testOneNameWhateverTheCallForm() {
    assertEquals(Set.of("tensorflow.keras.Model.add_weight"), apisIn("Weighted/build"));
    assertEquals(Set.of("tensorflow.keras.Model.add_weight"), apisIn("direct_add_weight"));
  }

  /**
   * The calls a recurrent layer's summary makes of its internal step and sequence operations name
   * no API: the operations are in {@link
   * com.ibm.wala.cast.python.ml.types.TensorFlowTypes#NOT_API_CLASSES}.
   */
  @Test
  public void testInternalOperations() {
    Set<String> names = new TreeSet<>();
    boolean found = false;
    for (CGNode node : callGraph) {
      String name = node.getMethod().getDeclaringClass().getName().toString();
      if (!name.startsWith("Ltensorflow/keras/layers/RNN/") || node.getIR() == null) continue;
      found = true;
      for (SSAInstruction instruction : node.getIR().getInstructions())
        if (instruction instanceof SSAAbstractInvokeInstruction call)
          names.addAll(TensorFlowApis.of(builder, node, call));
    }
    assertFalse("The recurrent layer's summary must be in the call graph.", !found);
    assertFalse(names.toString(), names.contains("tensorflow.keras.layers.SimpleRNNCell"));
    assertFalse(names.toString(), names.contains("tensorflow.keras.layers.RNN"));
  }

  /** A slice, {@code l[1:3]}, which the front end lowers to a call of the {@code slice} builtin. */
  @Test
  public void testSliceSyntax() {
    assertEquals(Set.of("builtins.slice"), apisIn("slice_list"));
  }

  @Test
  public void testSliceBuiltin() {
    assertEquals(Set.of("builtins.slice"), apisIn("slice_builtin"));
  }

  @Test
  public void testLenBuiltin() {
    assertEquals(Set.of("builtins.len"), apisIn("length"));
  }

  /** {@code np.asarray}, which no summary models: named by its attribute chain. */
  @Test
  public void testNumpyAsarray() {
    assertEquals(Set.of("numpy.asarray"), apisIn("np_asarray"));
  }

  @Test
  public void testNumpyArray() {
    assertEquals(Set.of("numpy.array"), apisIn("np_array"));
  }

  /**
   * {@code import scipy.special} then {@code scipy.special.expit(a)}: no name. The front end binds
   * the import to {@code special} rather than {@code scipy}, so the chain's root, the global {@code
   * scipy}, has an empty points-to set, and an empty root is not named by its spelling.
   *
   * <p>TODO: name {@code scipy.special.expit} once the front end binds {@code import a.b} to {@code
   * a}: https://github.com/wala/ML/issues/1140.
   */
  @Test
  public void testScipyExpit() {
    assertEquals(Set.of(), apisIn("scipy_expit"));
  }

  /**
   * {@code from scipy import special} then {@code special.expit(a)}: the chain roots at the {@code
   * special} submodule, whose functions are not modeled.
   */
  @Test
  public void testSpecialExpit() {
    assertEquals(Set.of("scipy.special.expit"), apisIn("special_expit"));
  }

  /** {@code import scipy} then {@code scipy.special.expit(a)}: the chain roots at a module. */
  @Test
  public void testScipyRootExpit() {
    assertEquals(Set.of("scipy.special.expit"), apisIn("scipy_root_expit"));
  }

  /** {@code import scipy} then {@code scipy.sparse.diags(a)}: the chain roots at a module. */
  @Test
  public void testScipyRootDiags() {
    assertEquals(Set.of("scipy.sparse.diags"), apisIn("scipy_root_diags"));
  }

  /**
   * {@code import scipy.sparse as sp} then {@code sp.diags(a)}: the chain roots at the {@code
   * sparse} submodule, and the modeled function's class adds no second name.
   */
  @Test
  public void testSparseDiags() {
    assertEquals(Set.of("scipy.sparse.diags"), apisIn("sparse_diags"));
  }

  @Test
  public void testBool() {
    assertEquals(Set.of("builtins.bool"), apisIn("to_bool"));
  }
}
