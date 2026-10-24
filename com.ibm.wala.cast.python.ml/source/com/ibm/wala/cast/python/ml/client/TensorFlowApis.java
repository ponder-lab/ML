package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.util.Util.withoutPythonScalarKeys;

import com.ibm.wala.cast.python.ml.types.NumpyTypes;
import com.ibm.wala.cast.python.ml.types.ScipyTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.TypeReference;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * The client-facing authority on which TensorFlow or NumPy API a call invokes. A client asks it
 * rather than matching the analysis's summary class names, following its trampolines, or walking a
 * callee's attribute chain itself, which would tie the client to how the analysis lays out its
 * models and its IR.
 *
 * <p>An API is named by its fully qualified Python name, such as {@code tensorflow.reshape}, {@code
 * tensorflow.random.normal}, {@code tensorflow.Tensor.set_shape}, {@code
 * tensorflow.keras.layers.Dense.__call__} or {@code numpy.zeros}. A call's names are its targets'
 * canonical names plus the spelling at the call site:
 *
 * <ol>
 *   <li>The canonical name of each call-graph target, trampolines followed: a target the analysis's
 *       table of modeled functions lists ({@link TensorFlowTypes#TYPE_REFERENCE_TO_SIGNATURE},
 *       {@link NumpyTypes#TYPE_REFERENCE_TO_SIGNATURE}) is named by its signature, and a Keras
 *       summary in a package whose class paths are its APIs' paths ({@code
 *       tensorflow/keras/layers/Dense/__call__}) by its class path. One API has one canonical name
 *       whatever the call form: {@code tf.math.sigmoid} and {@code tf.nn.sigmoid} are both {@code
 *       tensorflow.nn.sigmoid}. A class in {@link TensorFlowTypes#NOT_API_CLASSES} names nothing.
 *   <li>The callee's attribute chain in the calling IR, when it roots at a module object the
 *       analysis models (TensorFlow, NumPy, SciPy, or SciPy's {@code sparse} or {@code special}
 *       submodule), as {@code T.math.sigmoid} with {@code import tensorflow as T} does: {@code
 *       tensorflow.math.sigmoid}. A client matching one spelling of an aliased API finds it this
 *       way. This is also the only way an API the table does not list is named, whether the
 *       analysis models it through a summary shared with others, such as {@code tf.image.resize},
 *       or does not model it. It is never used for a call that may run a program function, since a
 *       program can store its own function on a module object.
 * </ol>
 *
 * A Python builtin a call reaches is named in the {@code builtins} module, as {@code
 * builtins.slice}, {@code builtins.len} or {@code builtins.zip}; the front end lowers a slice,
 * {@code l[1:3]}, to a call of the {@code slice} builtin. A call that names none of these, such as
 * a call of a program function, names no API.
 *
 * @see CallTargets
 */
public final class TensorFlowApis {

  /** The prefix the analysis's signatures abbreviate {@code tensorflow} to. */
  private static final String TF_PREFIX = "tf.";

  /** The prefix the analysis's signatures abbreviate {@code numpy} to. */
  private static final String NP_PREFIX = "np.";

  /** The suffix that marks a signature as a call of the named function. */
  private static final String CALL_SUFFIX = "()";

  /**
   * The prefix of the Python builtins' summary classes, such as {@code slice}'s and {@code len}'s.
   */
  private static final String BUILTIN_CLASS_PREFIX = "Lwala/builtin/";

  /** The prefix of the Keras summaries' class paths. */
  private static final String KERAS_CLASS_PREFIX = "Ltensorflow/keras/";

  /**
   * The Keras packages whose summaries' class paths are their APIs' paths, below {@link
   * #KERAS_CLASS_PREFIX}. A package that holds shared or synthetic classes, such as the
   * serialization functions several modules share, is not one.
   */
  private static final Set<String> KERAS_API_PACKAGES =
      Set.of(
          "layers/",
          "models/",
          "losses/",
          "optimizers/",
          "constraints/",
          "datasets/",
          "utils/",
          "preprocessing/image/",
          "applications/");

  /** The class-path segment of a Keras class object, whose {@code do} constructs, not applies. */
  private static final String CLASS_OBJECT_SEGMENT = "/class/";

  /** The module objects an attribute chain may root at, with the module's Python name. */
  private static final Map<TypeReference, String> MODULES =
      Map.of(
          TensorFlowTypes.TENSORFLOW_TYPE,
          "tensorflow",
          TensorFlowTypes.NUMPY_TYPE,
          "numpy",
          ScipyTypes.SCIPY_TYPE,
          "scipy",
          TypeReference.findOrCreate(PythonTypes.pythonLoader, "Lscipy/sparse"),
          "scipy.sparse",
          TypeReference.findOrCreate(PythonTypes.pythonLoader, "Lscipy/special"),
          "scipy.special");

  private TensorFlowApis() {}

  /**
   * Returns the APIs a call may invoke.
   *
   * @param builder The builder whose call graph and pointer analysis hold the call's targets.
   * @param node The node whose IR holds the call.
   * @param call The call.
   * @return The fully qualified names of the APIs the call may invoke, sorted; empty when it
   *     invokes no API this can name.
   */
  public static Set<String> of(
      PropagationCallGraphBuilder builder, CGNode node, SSAAbstractInvokeInstruction call) {
    CallGraph callGraph = builder.getCallGraph();
    IClassHierarchy cha = builder.getClassHierarchy();
    Set<String> ret = new TreeSet<>();
    Set<CGNode> seen = new HashSet<>();
    Deque<CGNode> worklist =
        new ArrayDeque<>(callGraph.getPossibleTargets(node, call.getCallSite()));
    boolean program = false;
    while (!worklist.isEmpty()) {
      CGNode target = worklist.pop();
      if (!seen.add(target)) continue;
      // A trampoline is followed before it is named, so that one API has one name whether it is
      // called through a bound method or directly.
      if (CallTargets.isTrampoline(cha, target.getMethod())) {
        for (Iterator<CGNode> forwarded = callGraph.getSuccNodes(target); forwarded.hasNext(); )
          worklist.push(forwarded.next());
        continue;
      }
      if (CallTargets.isProgramCode(target.getMethod())) {
        program = true;
        continue;
      }
      String name = apiName(target.getMethod().getDeclaringClass().getReference());
      if (name != null) ret.add(name);
    }
    if (!program && call.getNumberOfUses() > 0) {
      String chained = attributeChainName(builder.getPointerAnalysis(), node, call.getUse(0));
      if (chained != null) ret.add(chained);
    }
    return Collections.unmodifiableSet(ret);
  }

  /**
   * The fully qualified name of the API a modeled function's declaring class stands for. A class
   * the analysis's table lists names the API its signature does: {@code tf.reshape()} names {@code
   * tensorflow.reshape}, and {@code np.concatenate()} and {@code numpy.concatenate()} both name
   * {@code numpy.concatenate}. A signature that is not of a call, such as a dataset field's, names
   * none. A Keras summary the table does not list names the API its class path does, when it is in
   * one of {@link #KERAS_API_PACKAGES}.
   *
   * @param function The declaring class.
   * @return The API's name, or {@code null} when the class names no API.
   */
  static String apiName(TypeReference function) {
    if (TensorFlowTypes.NOT_API_CLASSES.contains(function)) return null;
    String signature =
        TensorFlowTypes.TYPE_REFERENCE_TO_SIGNATURE.getOrDefault(
            function, NumpyTypes.TYPE_REFERENCE_TO_SIGNATURE.get(function));
    if (signature != null) {
      if (!signature.endsWith(CALL_SUFFIX)) return null;
      String name = signature.substring(0, signature.length() - CALL_SUFFIX.length());
      if (name.isEmpty() || name.contains(" ") || name.contains("/") || name.contains("("))
        return null;
      if (name.startsWith(TF_PREFIX)) return "tensorflow." + name.substring(TF_PREFIX.length());
      if (name.startsWith(NP_PREFIX)) return "numpy." + name.substring(NP_PREFIX.length());
      return name;
    }
    String path = function.getName().toString();
    if (path.startsWith(BUILTIN_CLASS_PREFIX)) {
      String builtin = path.substring(BUILTIN_CLASS_PREFIX.length());
      return builtin.isEmpty() || builtin.contains("/") ? null : "builtins." + builtin;
    }
    if (!path.startsWith(KERAS_CLASS_PREFIX) || path.contains(CLASS_OBJECT_SEGMENT)) return null;
    String rest = path.substring(KERAS_CLASS_PREFIX.length());
    for (String keras : KERAS_API_PACKAGES)
      if (rest.startsWith(keras)) return path.substring(1).replace('/', '.');
    return null;
  }

  /**
   * The name of a callee read as an attribute chain off a module object in the calling IR, as
   * {@code tf.image.resize} is: the module's name, then each attribute's.
   *
   * @param pointerAnalysis The pointer analysis.
   * @param node The node whose IR reads the callee.
   * @param callee The callee's value number.
   * @return The chain's name, or {@code null} when the callee is no attribute chain rooted at a
   *     module object.
   */
  private static String attributeChainName(
      PointerAnalysis<InstanceKey> pointerAnalysis, CGNode node, int callee) {
    DefUse du = node.getDU();
    SymbolTable symbolTable = node.getIR() == null ? null : node.getIR().getSymbolTable();
    if (du == null || symbolTable == null) return null;
    StringBuilder suffix = new StringBuilder();
    int current = callee;
    for (int depth = 0; depth < 16; depth++) {
      String module = moduleName(pointerAnalysis, node, current);
      if (module != null) return suffix.length() == 0 ? null : module + suffix;
      SSAInstruction def = du.getDef(current);
      if (!(def instanceof PythonPropertyRead read)
          || !symbolTable.isStringConstant(read.getMemberRef())) return null;
      suffix.insert(0, "." + symbolTable.getStringValue(read.getMemberRef()));
      current = read.getObjectRef();
    }
    return null;
  }

  /**
   * The name of the module object a value is, when every member of its points-to set is the same
   * module.
   *
   * @param pointerAnalysis The pointer analysis.
   * @param node The value's node.
   * @param vn The value.
   * @return The module's name, or {@code null}.
   */
  private static String moduleName(
      PointerAnalysis<InstanceKey> pointerAnalysis, CGNode node, int vn) {
    String ret = null;
    for (InstanceKey ik :
        withoutPythonScalarKeys(
            pointerAnalysis.getPointsToSet(
                pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, vn)))) {
      String module = MODULES.get(ik.concreteType().getReference());
      if (module == null || (ret != null && !ret.equals(module))) return null;
      ret = module;
    }
    return ret;
  }
}
