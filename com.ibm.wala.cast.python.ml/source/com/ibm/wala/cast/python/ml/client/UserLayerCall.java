package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.types.PythonTypes.CALLABLE_METHOD_NAME;
import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.util.TensorShapeUtil;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.shrike.shrikeBT.IBinaryOpInstruction;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.ISSABasicBlock;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A composed transform for a layer that is a USER class: a {@code tf.keras.Model} or layer subclass
 * held in a {@code Sequential}'s list, whose {@code call} body is never invoked (the model's
 * summary allocates its result without calling anything), so the body has no call-graph node and
 * neither the dispatch table nor the dataflow can type it (<a
 * href="https://github.com/wala/ML/issues/832">wala/ML#832</a>).
 *
 * <p>This is deliberately NOT a general interpreter. It is the fold's own discipline applied one
 * level down: the body's IR is obtained without a call-graph node, and the walk follows the def-use
 * chain from the input parameter to the return, where every hop must be something the walk can name
 * — a layer-attribute call dispatched through the existing table (or recursively through this
 * class), a member of the witnessed pass-through function set, or an elementwise operation over two
 * resolved operands. Any other hop declines the whole composition, exactly as an unmodeled layer
 * declines the list fold: a partially understood body would compose a shape for a program that was
 * not written.
 *
 * <p>The bare-function form serves a lambda held as a layer attribute (the residual-shortcut idiom,
 * {@code self.shortcut = lambda x, _: x}): its body folds by the same walk with no receiver, and
 * the identity comes out of the walk rather than being assumed.
 *
 * <p><b>Invariant: this generator is valid only inside a composition.</b> It is constructed nowhere
 * but the fold's layer dispatch and its own nested-body recursion, never by a dispatch table, and
 * it surrenders both axes when no {@link ComposedArguments} are in force. That standing is what
 * licenses the one deliberate deviation from the documented dtype lattice: a declined fold answers
 * {@code null} on the dtype axis (an encoding the lattice does not define) so the enclosing fold
 * can fall through to the model treatment that owns the fallback, instead of reading a {@code
 * UNKNOWN} floor as a successful resolution. Registering this class in either dispatch table would
 * therefore introduce a lattice violation at every unseeded use; do not.
 */
@DispatchExempt(
    "Constructed only by delegation: the list fold's layer dispatch falls back to this class when"
        + " a layer's type has no table arm, and the body walk recurses through it for nested user"
        + " bodies; no dispatch-table type names it (wala/ML#832).")
public class UserLayerCall extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(UserLayerCall.class.getName());

  /** The recursion budget across nested user bodies and the elements they apply. */
  static final int BODY_FOLD_DEPTH_CAP = 6;

  /**
   * Function-form members the walk passes a value through unchanged on both axes. Entries are added
   * only with a composing witness (the wala/ML#832 allowlist discipline): each name is a claim that
   * the function preserves its argument's shape and dtype, and a wrong one composes a chain the
   * program does not run. {@code relu} is witnessed by the residual-block fixture.
   */
  private static final Set<String> PASS_THROUGH_FUNCTION_MEMBERS = Set.of("relu");

  /** The user body to fold. */
  private final IMethod body;

  /** Whether the body is a method (receiver at parameter 1, input at 2) or a bare function. */
  private final boolean methodForm;

  /** The remaining recursion budget for nested user bodies. */
  private final int depth;

  private UserLayerCall(CGNode node, IMethod body, boolean methodForm, int depth) {
    super(node);
    this.body = body;
    this.methodForm = methodForm;
    this.depth = depth;
  }

  /**
   * The result of folding a body or a hop: both axes together, since the hops and refusals are
   * identical between them.
   *
   * @param shapes The shape members; never {@code null} in a non-null state.
   * @param dTypes The dtype members; {@code UNKNOWN} stands in when a hop loses the dtype.
   */
  private record BodyState(Set<List<Dimension<?>>> shapes, Set<DType> dTypes) {}

  /**
   * Resolves the folding generator for a layer instance of a user class, or {@code null} when the
   * instance has no body this walk can obtain: the layer's class must carry a {@code call} member
   * body, or be a bare function object folded directly.
   *
   * @param builder The propagation call graph builder.
   * @param node The model-call node anchoring the composition.
   * @param layer The layer instance.
   * @param depth The remaining recursion budget.
   * @return The generator, or {@code null}.
   */
  static UserLayerCall forLayerInstance(
      PropagationCallGraphBuilder builder, CGNode node, InstanceKey layer, int depth) {
    if (depth <= 0) return null;
    AllocationSiteInNode allocation = getAllocationSiteInNode(layer);
    if (allocation == null) return null;
    // A user-class instance is allocated as a generic object inside the CLASS's synthetic
    // constructor, so the class identity is the allocating node's declaring class, not the
    // allocation's concrete type.
    TypeReference layerType = allocation.concreteType().getReference();
    if (layerType.equals(com.ibm.wala.cast.python.types.PythonTypes.object))
      layerType = allocation.getNode().getMethod().getDeclaringClass().getReference();
    IClassHierarchy cha = builder.getClassHierarchy();

    // A class instance: the body is the class's `call` member, itself a code-body class.
    IClass callClass =
        cha.lookupClass(
            TypeReference.findOrCreate(
                layerType.getClassLoader(),
                TypeName.string2TypeName(layerType.getName().toString() + "/call")));
    if (callClass != null) {
      IMethod callBody = callClass.getMethod(AstMethodReference.fnSelector);
      if (callBody instanceof AstMethod) return new UserLayerCall(node, callBody, true, depth);
    }

    // A bare function object (a lambda held as a layer attribute): fold its own body.
    IClass functionClass = cha.lookupClass(layerType);
    if (functionClass != null) {
      IMethod functionBody = functionClass.getMethod(AstMethodReference.fnSelector);
      if (functionBody instanceof AstMethod)
        return new UserLayerCall(node, functionBody, false, depth);
    }

    return null;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    BodyState folded = this.fold(builder);
    return folded == null || folded.shapes().isEmpty() ? null : folded.shapes();
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // A declined fold surrenders the axis entirely (null, not the UNKNOWN floor): this generator
    // runs only inside a composition, and handing the fold a "resolved" UNKNOWN would stop it
    // from deferring to the model treatment that owns the fallback.
    BodyState folded = this.fold(builder);
    return folded == null || folded.dTypes() == null || folded.dTypes().isEmpty()
        ? null
        : folded.dTypes();
  }

  /**
   * Folds the body over the composed input: every return's value must resolve through the hop walk,
   * and the result is their union.
   *
   * @param builder The propagation call graph builder.
   * @return The folded state, or {@code null} when any hop declines.
   */
  private BodyState fold(PropagationCallGraphBuilder builder) {
    ComposedArguments composed = this.getComposedArguments();
    if (composed == null || composed.inputShapes() == null) return null;

    IR ir = builder.getAnalysisCache().getIR(this.body);
    if (ir == null) return null;
    DefUse du = builder.getAnalysisCache().getDefUse(ir);
    SymbolTable symbolTable = ir.getSymbolTable();

    // Code-body convention: parameter 0 is the function object, 1 the receiver for a method.
    int inputIndex = this.methodForm ? 2 : 1;
    if (inputIndex >= ir.getNumberOfParameters()) return null;
    int selfVn = this.methodForm ? ir.getParameter(1) : -1;
    int inputVn = ir.getParameter(inputIndex);

    BodyState input =
        new BodyState(
            composed.inputShapes(),
            composed.inputDTypes() == null ? EnumSet.of(DType.UNKNOWN) : composed.inputDTypes());

    Set<List<Dimension<?>>> shapes = HashSetFactory.make();
    Set<DType> dTypes = EnumSet.noneOf(DType.class);
    boolean returned = false;

    for (SSAInstruction instruction : ir.getInstructions()) {
      if (!(instruction instanceof SSAReturnInstruction ret)) continue;
      int resultVn = ret.getResult();
      if (resultVn <= 0) return null; // A bare return has no value to type.

      BodyState resolved =
          this.resolveBodyValue(
              builder, du, symbolTable, selfVn, inputVn, input, resultVn, this.depth);
      if (resolved == null) {
        LOGGER.fine(
            () -> "Body fold of " + this.body.getReference() + " declined at return " + resultVn);
        return null;
      }
      shapes.addAll(resolved.shapes());
      dTypes.addAll(resolved.dTypes());
      returned = true;
    }

    return returned ? new BodyState(shapes, dTypes) : null;
  }

  /**
   * Resolves one body value through the hop walk.
   *
   * @param builder The propagation call graph builder.
   * @param du The body's def-use.
   * @param symbolTable The body's symbol table.
   * @param selfVn The receiver parameter's value number, or {@code -1} for a bare function.
   * @param inputVn The input parameter's value number.
   * @param input The composed input state.
   * @param vn The value to resolve.
   * @param remaining The remaining recursion budget.
   * @return The value's state, or {@code null} when a hop declines.
   */
  private BodyState resolveBodyValue(
      PropagationCallGraphBuilder builder,
      DefUse du,
      SymbolTable symbolTable,
      int selfVn,
      int inputVn,
      BodyState input,
      int vn,
      int remaining) {
    if (vn <= 0 || remaining <= 0) return null;
    if (vn == inputVn) return input;

    SSAInstruction def = du.getDef(vn);

    if (def instanceof PythonInvokeInstruction call) {
      SSAInstruction calleeDef = du.getDef(call.getUse(0));
      if (!(calleeDef instanceof PythonPropertyRead read)) return null;
      if (!symbolTable.isStringConstant(read.getMemberRef())) return null;
      String member = symbolTable.getStringValue(read.getMemberRef());

      if (call.getNumberOfPositionalParameters() < 2) return null;
      BodyState argument =
          this.resolveBodyValue(
              builder, du, symbolTable, selfVn, inputVn, input, call.getUse(1), remaining);
      if (argument == null) return null;

      // A layer-attribute call: the transform lives on the attribute's instances, read off the
      // composed receiver. Every member of the attribute's points-to union must compose, and the
      // result is their union — a residual shortcut holds a projection on one construction branch
      // and an identity function on the other, and both are chains the program can run.
      if (selfVn > 0 && read.getObjectRef() == selfVn)
        return this.applyAttributeMembers(builder, member, argument, remaining);

      // A function-form op passes through only when the member is on the witnessed set; the
      // composed seams cannot serve function-form generators (their input position is the seam's
      // receiver position), so anything off the set declines rather than dispatching wrong.
      if (PASS_THROUGH_FUNCTION_MEMBERS.contains(member)) return argument;

      return null;
    }

    // An elementwise operation over two resolved operands: the residual `out += shortcut(x)`.
    if (def instanceof SSABinaryOpInstruction binop
        && binop.getOperator() == IBinaryOpInstruction.Operator.ADD) {
      BodyState left =
          this.resolveBodyValue(
              builder, du, symbolTable, selfVn, inputVn, input, binop.getUse(0), remaining);
      if (left == null) return null;
      BodyState right =
          this.resolveBodyValue(
              builder, du, symbolTable, selfVn, inputVn, input, binop.getUse(1), remaining);
      if (right == null) return null;

      // Operand members that cannot broadcast cannot co-occur at runtime (the path-insensitive
      // shortcut attribute unions both construction branches, and only the coherent one pairs
      // with the residual), so incompatible PAIRS are filtered rather than declining the hop; the
      // hop declines only when no pair is possible at all.
      Set<List<Dimension<?>>> broadcast = HashSetFactory.make();
      for (List<Dimension<?>> l : left.shapes())
        for (List<Dimension<?>> r : right.shapes())
          if (TensorShapeUtil.areBroadcastable(l, r))
            broadcast.add(TensorShapeUtil.getBroadcastedShapes(l, r));
      if (broadcast.isEmpty()) return null;
      Set<DType> dTypes = EnumSet.copyOf(left.dTypes());
      dTypes.addAll(right.dTypes());
      return new BodyState(broadcast, dTypes);
    }

    return null;
  }

  /**
   * Applies a layer attribute's transform: each instance the attribute may hold must compose, and
   * the result is the union over them.
   *
   * @param builder The propagation call graph builder.
   * @param member The attribute name.
   * @param argument The state flowing into the attribute call.
   * @param remaining The remaining recursion budget.
   * @return The union state, or {@code null} when any member declines.
   */
  private BodyState applyAttributeMembers(
      PropagationCallGraphBuilder builder, String member, BodyState argument, int remaining) {
    ComposedArguments composed = this.getComposedArguments();

    Set<List<Dimension<?>>> shapes = HashSetFactory.make();
    Set<DType> dTypes = EnumSet.noneOf(DType.class);
    boolean any = false;

    for (InstanceKey receiver : composed.receiver()) {
      AllocationSiteInNode receiverASIN = getAllocationSiteInNode(receiver);
      if (receiverASIN == null) return null;

      IField field =
          builder
              .getClassHierarchy()
              .resolveField(FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(member), Root));
      if (field == null) return null;

      OrdinalSet<InstanceKey> attributePTS =
          builder
              .getPointerAnalysis()
              .getPointsToSet(builder.getPointerKeyForInstanceField(receiverASIN, field));
      if (attributePTS == null || attributePTS.isEmpty()) return null;

      for (InstanceKey element : attributePTS) {
        // A member allocated only on a branch the context's decided guards prove dead is the
        // OTHER construction path's value, not a live possibility of this field: the residual
        // shortcut holds a projection on one branch and an identity on the other, and each
        // instance context keeps exactly one alive (the wala/ML#763 criterion, reused).
        if (!isLiveAllocation(builder, element)) {
          LOGGER.fine(
              () -> "Attribute " + member + " member " + describe(element) + " is context-dead.");
          continue;
        }
        BodyState applied = this.applyElement(builder, element, argument, remaining - 1);
        if (applied == null) {
          LOGGER.fine(
              () -> "Attribute " + member + " member " + describe(element) + " declined the fold.");
          return null;
        }
        shapes.addAll(applied.shapes());
        dTypes.addAll(applied.dTypes());
        any = true;
      }
    }

    return any ? new BodyState(shapes, dTypes) : null;
  }

  /**
   * Whether an allocation is live in its context: its own instruction's block must be reachable
   * under the node's decided guards, and a synthetic constructor allocation must have at least one
   * live constructor call reaching it (both halves are the wala/ML#763 reachability criterion).
   *
   * @param builder The propagation call graph builder.
   * @param element The instance to test.
   * @return {@code true} iff the allocation can happen in this context.
   */
  private static boolean isLiveAllocation(
      PropagationCallGraphBuilder builder, InstanceKey element) {
    AllocationSiteInNode allocation = getAllocationSiteInNode(element);
    if (allocation == null) return true; // Not an allocation-site key; nothing to test.
    CGNode node = allocation.getNode();

    if (node.getIR() != null) {
      SSAInstruction alloc = node.getIR().getNew(allocation.getSite());
      if (alloc != null && alloc.iIndex() >= 0) {
        ISSABasicBlock block =
            node.getIR().getControlFlowGraph().getBlockForInstruction(alloc.iIndex());
        if (block != null
            && !computeReachableBlocksUnderBindings(builder, node, java.util.Map.of())
                .contains(block)) return false;
      }
    }

    return builder.getCallGraph().getPredNodeCount(node) == 0
        || !getCallerInvokes(builder, node).isEmpty();
  }

  /**
   * Applies one element's transform through the same dispatch the list fold uses, with this class
   * as the fallback for a nested user body.
   *
   * @param builder The propagation call graph builder.
   * @param element The layer or function instance.
   * @param argument The state flowing in.
   * @param remaining The remaining recursion budget.
   * @return The transformed state, or {@code null} when the element declines.
   */
  private BodyState applyElement(
      PropagationCallGraphBuilder builder, InstanceKey element, BodyState argument, int remaining) {
    if (remaining <= 0) return null;

    TensorGenerator generator = null;
    AllocationSiteInNode allocation = getAllocationSiteInNode(element);
    if (allocation != null) {
      TypeReference elementType = allocation.concreteType().getReference();
      generator =
          createManualGenerator(
              this.getNode(),
              TypeReference.findOrCreate(
                  elementType.getClassLoader(),
                  TypeName.string2TypeName(
                      elementType.getName().toString() + "/" + CALLABLE_METHOD_NAME)),
              builder);
    }
    if (generator == null)
      generator = forLayerInstance(builder, this.getNode(), element, remaining);
    if (generator == null) return null;

    generator.composeWith(
        new ComposedArguments(
            OrdinalSet.toOrdinalSet(
                List.of(element), builder.getPointerAnalysis().getInstanceKeyMapping()),
            argument.shapes(),
            argument.dTypes()));

    Set<List<Dimension<?>>> shapes = generator.getDefaultShapes(builder);
    if (shapes == null || shapes.isEmpty()) return null;
    Set<DType> dTypes = generator.getDefaultDTypes(builder);
    return new BodyState(
        shapes,
        dTypes == null || dTypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : EnumSet.copyOf(dTypes));
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
