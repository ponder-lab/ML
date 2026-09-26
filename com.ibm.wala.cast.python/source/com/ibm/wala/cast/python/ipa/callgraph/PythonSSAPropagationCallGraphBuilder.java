/*
 * Copyright (c) 2018 IBM Corporation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 */
package com.ibm.wala.cast.python.ipa.callgraph;

import static com.ibm.wala.cast.python.types.PythonTypes.CALLABLE_METHOD_NAME;
import static com.ibm.wala.cast.python.types.PythonTypes.CALLABLE_METHOD_NAME_FOR_KERAS_MODELS;
import static com.ibm.wala.cast.python.types.PythonTypes.DO_METHOD_NAME;
import static com.ibm.wala.cast.python.types.PythonTypes.INIT_METHOD_NAME;
import static com.ibm.wala.cast.python.util.Util.IMPORT_WILDCARD_CHARACTER;
import static com.ibm.wala.cast.python.util.Util.MODULE_INITIALIZATION_FILENAME;
import static com.ibm.wala.cast.python.util.Util.PYTHON_FILE_EXTENSION;

import com.google.common.collect.Maps;
import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.ipa.callgraph.AstSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.ipa.callgraph.GlobalObjectKey;
import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalWrite;
import com.ibm.wala.cast.ir.ssa.AstPropertyRead;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ipa.summaries.PythonConstructorFunction;
import com.ibm.wala.cast.python.ipa.summaries.PythonInstanceMethodTrampoline;
import com.ibm.wala.cast.python.ir.PythonLanguage;
import com.ibm.wala.cast.python.loader.StarFormalDeclaration;
import com.ibm.wala.cast.python.ssa.ForElementGetInstruction;
import com.ibm.wala.cast.python.ssa.PythonBinaryOpInstruction;
import com.ibm.wala.cast.python.ssa.PythonInstructionVisitor;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.core.util.CancelRuntimeException;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.fixpoint.AbstractOperator;
import com.ibm.wala.fixpoint.UnaryOperator;
import com.ibm.wala.ipa.callgraph.AnalysisOptions;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.ContextItem;
import com.ibm.wala.ipa.callgraph.ContextKey;
import com.ibm.wala.ipa.callgraph.IAnalysisCacheView;
import com.ibm.wala.ipa.callgraph.propagation.AbstractFieldPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.FilteredPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.FilteredPointerKey.TypeFilter;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKeyFactory;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.StaticFieldKey;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.shrike.shrikeBT.IBinaryOpInstruction;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAArrayLoadInstruction;
import com.ibm.wala.ssa.SSAArrayStoreInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAGetInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAInvokeInstruction;
import com.ibm.wala.ssa.SSAPutInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.Descriptor;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.IntIterator;
import com.ibm.wala.util.intset.IntSetUtil;
import com.ibm.wala.util.intset.MutableIntSet;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.logging.Logger;

public class PythonSSAPropagationCallGraphBuilder extends AstSSAPropagationCallGraphBuilder {

  /**
   * The synthetic property key under which {@code xs.append(v)} stores {@code v} on {@code xs}; see
   * {@code PythonConstraintVisitor.processListAppend}. Value-iteration surfaces the property's
   * values regardless of its name, so the name only needs to avoid colliding with real program
   * properties.
   */
  public static final String LIST_APPEND_CONTENTS_FIELD = "__list_append_contents__";

  /**
   * The synthetic field holding the elements of a list or tuple produced by repetition ({@code xs *
   * n}) or concatenation ({@code xs + ys}) (wala/ML#960). Like {@value #LIST_APPEND_CONTENTS_FIELD}
   * it is read by every non-constant subscript, iteration and {@code zip}; unlike it, no shape
   * reader interprets it, so a synthesized list, whose length is unknowable, never yields an
   * extent.
   */
  public static final String LIST_OPERATION_CONTENTS_FIELD = "__list_operation_contents__";

  private static final Logger logger =
      Logger.getLogger(PythonSSAPropagationCallGraphBuilder.class.getName());

  public PythonSSAPropagationCallGraphBuilder(
      IClassHierarchy cha,
      AnalysisOptions options,
      IAnalysisCacheView cache,
      PointerKeyFactory pointerKeyFactory) {
    super(PythonLanguage.Python.getFakeRootMethod(cha, cache), options, cache, pointerKeyFactory);
  }

  protected boolean isConstantRef(SymbolTable symbolTable, int valueNumber) {
    return valueNumber != -1 && symbolTable.isConstant(valueNumber);
  }

  @Override
  protected boolean useObjectCatalog() {
    return true;
  }

  @Override
  public GlobalObjectKey getGlobalObject(Atom language) {
    assert language.equals(PythonLanguage.Python.getName());
    return new GlobalObjectKey(cha.lookupClass(PythonTypes.Root));
  }

  @Override
  protected AbstractFieldPointerKey fieldKeyForUnknownWrites(AbstractFieldPointerKey fieldKey) {
    return null;
  }

  @Override
  protected boolean sameMethod(CGNode opNode, String definingMethod) {
    return definingMethod.equals(
        opNode.getMethod().getReference().getDeclaringClass().getName().toString());
  }

  private static final Collection<TypeReference> types =
      Arrays.asList(PythonTypes.string, TypeReference.Int);

  /**
   * A mapping of script names to wildcard imports. We use a {@link Deque} here because we want to
   * always examine the last (front of the queue) encountered wildcard import library for known
   * names assuming that import instructions are traversed from first to last.
   */
  private Map<String, Deque<MethodReference>> scriptToWildcardImports = Maps.newHashMap();

  /**
   * The element types whose slice results get an allocation of their own (wala/ML#916). A subscript
   * with slice syntax lowers to a call of the {@code slice} builtin, whose result used to be its
   * receiver's points-to set, so every reader of the result through the points-to set saw the
   * receiver's pre-slice window. For a receiver key whose concrete type is in this set, the result
   * instead gets a fresh allocation of that type at the call site; every other key passes through
   * as before. Empty by default, so a client that has no element types to name sees no change; the
   * tensor analysis names its tensor and array types. Staged on purpose: only where the element
   * type is known does a slice yield a value of the same kind, so dispatch through the result
   * survives; a general container's subscript yields an element of unknown type and keeps the
   * pass-through.
   */
  private Set<TypeReference> freshSliceResultTypes = Collections.emptySet();

  /**
   * Names the element types whose slice results get an allocation of their own (wala/ML#916); see
   * {@link #freshSliceResultTypes}.
   *
   * @param types The concrete receiver types whose slices allocate.
   */
  public void setFreshSliceResultTypes(Set<TypeReference> types) {
    this.freshSliceResultTypes = types == null ? Collections.emptySet() : Set.copyOf(types);
  }

  public static class PythonConstraintVisitor extends AstConstraintVisitor
      implements PythonInstructionVisitor {

    /**
     * The None constant has no object catalog for the pointer analysis (wala/ML#964): no field name
     * is recorded on it, so a wildcard read over a set that includes {@code None} enumerates
     * nothing from it. The reflected write null-guards this key; the heap model's catalog key,
     * which consumers read, stays the factory's.
     *
     * @param I The instance key.
     * @return The catalog key, or {@code null} for the None constant.
     */
    @Override
    public PointerKey getPointerKeyForObjectCatalog(InstanceKey I) {
      if (isNoneConstant(I)) return null;
      return super.getPointerKeyForObjectCatalog(I);
    }

    /**
     * A reflected field read on the None constant reads nothing (wala/ML#964).
     *
     * @param I The receiver.
     * @param F The field name key.
     * @return The pointer keys, none for the None constant.
     */
    @Override
    public Iterator<PointerKey> getPointerKeysForReflectedFieldRead(InstanceKey I, InstanceKey F) {
      if (isNoneConstant(I)) return Collections.emptyIterator();
      return super.getPointerKeysForReflectedFieldRead(I, F);
    }

    /**
     * A reflected field write on the None constant writes nothing (wala/ML#964).
     *
     * @param I The receiver.
     * @param F The field name key.
     * @return The pointer keys, none for the None constant.
     */
    @Override
    public Iterator<PointerKey> getPointerKeysForReflectedFieldWrite(InstanceKey I, InstanceKey F) {
      if (isNoneConstant(I)) return Collections.emptyIterator();
      return super.getPointerKeysForReflectedFieldWrite(I, F);
    }

    private static final String GLOBAL_IDENTIFIER = "global";

    private static final Atom IMPORT_FUNCTION_NAME = Atom.findOrCreateAsciiAtom("import");

    @Override
    protected PythonSSAPropagationCallGraphBuilder getBuilder() {
      return (PythonSSAPropagationCallGraphBuilder) builder;
    }

    public PythonConstraintVisitor(AstSSAPropagationCallGraphBuilder builder, CGNode node) {
      super(builder, node);
    }

    /**
     * @param objType the type of the container of which iteration is being done
     * @return whether iteration is over values rather than keys
     *     <p>For some collection types in Python, mainly sets and lists, iteration over a
     *     collection returns the values contained in that collection. For other types, such as
     *     dictionaries, iteration is over the keys that that collection contains.
     *     <p>We also use this mechanism for generators, which put every yielded value in the
     *     synthetic field named {@value PythonTypes#GENERATOR_CONTENT_FIELD_NAME}, which is read by
     *     this mechanism. Generator expressions use the iterator type and generator functions use
     *     CodeBody (wala/ML#696).
     */
    private boolean isValueForKeyType(IClass objType) {
      IClassHierarchy cha = getClassHierarchy();
      return cha.isSubclassOf(objType, cha.lookupClass(PythonTypes.list))
          || cha.isSubclassOf(objType, cha.lookupClass(PythonTypes.set))
          || cha.isSubclassOf(objType, cha.lookupClass(PythonTypes.iterator))
          || cha.isSubclassOf(objType, cha.lookupClass(PythonTypes.CodeBody));
    }

    /**
     * Re-runs the given lexical-access visit whenever a new closure function object reaches this
     * node's function value (value number 1). {@code AstConstraintVisitor.visitLexical} resolves
     * the defining frames of a lexical access from a one-time snapshot of that value's points-to
     * set; when distinct closures of the same function share this node (e.g. under call-string
     * context truncation), a closure arriving after the snapshot never gets its frame wired,
     * silently starving the access and every dispatch downstream of it (<a
     * href="https://github.com/wala/ML/issues/690">wala/ML#690</a>). The side effect re-resolves on
     * every growth of the function value's points-to set; the underlying constraint additions are
     * idempotent, so re-running is safe.
     *
     * @param instruction The lexical access to re-resolve on closure growth.
     * @param revisit Re-invokes the superclass visit for {@code instruction}.
     */
    private void refreshLexicalOnClosureGrowth(SSAInstruction instruction, Runnable revisit) {
      // When the function value's contents are invariant, the visit's snapshot is already
      // complete (the single closure object is known statically) and its points-to set is
      // implicitly represented, so registering a side effect is both unnecessary and disallowed:
      // `newSideEffect` would crash `findOrCreatePointsToSet` (the wala/ML#668 trap; observed as
      // `UnimplementedError` on script-toplevel nodes in WALA's JS test suite for the upstream
      // form of this fix, wala/WALA#1991).
      if (contentsAreInvariant(ir.getSymbolTable(), du, 1)) return;

      PointerKey function = getPointerKeyForLocal(1);
      system.newSideEffect(
          new LexicalRefreshOperator(node, instruction.iIndex(), revisit), function);
    }

    /**
     * The side-effect operator registered by {@link #refreshLexicalOnClosureGrowth}; identity is
     * the (node, instruction index) pair so the fixpoint system de-duplicates re-registrations of
     * the same lexical access.
     */
    private static final class LexicalRefreshOperator extends UnaryOperator<PointsToSetVariable> {
      private final CGNode node;
      private final int instructionIndex;
      private final Runnable revisit;

      private LexicalRefreshOperator(CGNode node, int instructionIndex, Runnable revisit) {
        this.node = node;
        this.instructionIndex = instructionIndex;
        this.revisit = revisit;
      }

      @Override
      public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
        revisit.run();
        return NOT_CHANGED;
      }

      @Override
      public int hashCode() {
        return node.hashCode() * 31 + instructionIndex;
      }

      @Override
      public boolean equals(Object o) {
        return o instanceof LexicalRefreshOperator other
            && instructionIndex == other.instructionIndex
            && node.equals(other.node);
      }

      @Override
      public String toString() {
        return "lexical refresh of " + instructionIndex + " in " + node;
      }
    }

    @Override
    public void visitAstLexicalRead(AstLexicalRead instruction) {
      super.visitAstLexicalRead(instruction);
      refreshLexicalOnClosureGrowth(instruction, () -> super.visitAstLexicalRead(instruction));
    }

    @Override
    public void visitAstLexicalWrite(AstLexicalWrite instruction) {
      super.visitAstLexicalWrite(instruction);
      refreshLexicalOnClosureGrowth(instruction, () -> super.visitAstLexicalWrite(instruction));
    }

    @Override
    public void visitForElementGet(ForElementGetInstruction forElementGet) {
      SymbolTable symtab = ir.getSymbolTable();
      int objVn = forElementGet.getUse(0);
      final PointerKey objKey = getPointerKeyForLocal(objVn);
      int eltVn = forElementGet.getUse(1);
      final PointerKey eltKey = getPointerKeyForLocal(eltVn);
      int resultVn = forElementGet.getDef();
      final PointerKey resultKey = getPointerKeyForLocal(resultVn);

      if (contentsAreInvariant(symtab, du, objVn)) {
        for (InstanceKey ik : getInvariantContents(objVn)) {
          if (!isValueForKeyType(ik.concreteType())) {
            system.newConstraint(resultKey, assignOperator, eltKey);
          } else {
            newFieldRead(node, objVn, eltVn, resultVn);
          }
        }
      } else {
        system.newConstraint(
            resultKey,
            new AbstractOperator<PointsToSetVariable>() {
              @Override
              public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable[] rhs) {
                boolean changed = false;
                for (PointsToSetVariable rv : rhs) {
                  if (rv.getValue() != null) {
                    IntIterator is = rv.getValue().intIterator();
                    while (is.hasNext()) {
                      InstanceKey ik = system.getInstanceKey(is.next());
                      if (!isValueForKeyType(ik.concreteType())) {
                        changed |= system.newConstraint(resultKey, assignOperator, eltKey);
                      } else {
                        newFieldRead(node, objVn, eltVn, resultVn);
                      }
                    }
                  }
                }
                if (changed) {
                  return CHANGED;
                } else {
                  return NOT_CHANGED;
                }
              }

              @Override
              public int hashCode() {
                return objKey.hashCode() * eltKey.hashCode();
              }

              @Override
              public boolean equals(Object o) {
                return this == o;
              }

              @Override
              public String toString() {
                return "next element of " + objKey;
              }
            },
            objKey,
            eltKey);
      }
    }

    @Override
    public void visitGet(SSAGetInstruction instruction) {
      SymbolTable symtab = ir.getSymbolTable();
      String name = instruction.getDeclaredField().getName().toString();

      int objVn = instruction.getRef();
      final PointerKey objKey = getPointerKeyForLocal(objVn);

      int lvalVn = instruction.getDef();
      final PointerKey lvalKey = getPointerKeyForLocal(lvalVn);

      if (contentsAreInvariant(symtab, du, objVn)) {
        system.recordImplicitPointsToSet(objKey);
        for (InstanceKey ik : getInvariantContents(objVn)) {
          if (types.contains(ik.concreteType().getReference())) {
            @SuppressWarnings("unused")
            Pair<String, TypeReference> key = Pair.make(name, ik.concreteType().getReference());
            // system.newConstraint(lvalKey, new ConcreteTypeKey(getBuilder().ensure(key)));
          }
        }
      } else {
        system.newSideEffect(
            new AbstractOperator<PointsToSetVariable>() {
              @Override
              public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable[] rhs) {
                if (rhs[0].getValue() != null)
                  rhs[0]
                      .getValue()
                      .foreach(
                          (i) -> {
                            InstanceKey ik = system.getInstanceKey(i);
                            if (types.contains(ik.concreteType().getReference())) {
                              @SuppressWarnings("unused")
                              Pair<String, TypeReference> key =
                                  Pair.make(name, ik.concreteType().getReference());
                              // system.newConstraint(lvalKey, new
                              // ConcreteTypeKey(getBuilder().ensure(key)));
                            }
                          });
                return NOT_CHANGED;
              }

              @Override
              public int hashCode() {
                return node.hashCode() * instruction.hashCode();
              }

              @Override
              public boolean equals(Object o) {
                return getClass().equals(o.getClass()) && hashCode() == o.hashCode();
              }

              @Override
              public String toString() {
                return "get function " + name + " at " + instruction;
              }
            },
            new PointerKey[] {lvalKey});
      }

      // TODO Auto-generated method stub
      super.visitGet(instruction);
    }

    @Override
    public void visitPythonInvoke(PythonInvokeInstruction inst) {
      visitInvokeInternal(inst, new DefaultInvariantComputer());
      processListAppend(inst);
      processTextRead(inst);
    }

    /**
     * Surfaces append-accumulated list contents at subscript reads (<a
     * href="https://github.com/wala/ML/issues/661">wala/ML#661</a>): a property read whose member
     * is not a constant string also reads the synthetic {@value #LIST_APPEND_CONTENTS_FIELD}
     * property, the read-side dual of {@link #processListAppend}'s write. Without this, values
     * accumulated through {@code append} surface only under value iteration, and an indexed
     * dispatch such as {@code self.sub_layers[i](x)} over an append-built list has an empty callee
     * set. Named-attribute reads (constant string members) are excluded so method and attribute
     * lookups do not observe element values; the synthetic property is only ever written on append
     * receivers, so on all other objects the extra read contributes nothing.
     *
     * @param instruction The property read to examine.
     */
    private void processListContentsRead(AstPropertyRead instruction) {
      SymbolTable symtab = ir.getSymbolTable();
      int memberRef = instruction.getMemberRef();
      if (symtab.isConstant(memberRef) && symtab.getConstantValue(memberRef) instanceof String)
        return;

      InstanceKey contentsKey =
          getBuilder().getInstanceKeyForConstant(PythonTypes.string, LIST_APPEND_CONTENTS_FIELD);
      InstanceKey operationKey =
          getBuilder().getInstanceKeyForConstant(PythonTypes.string, LIST_OPERATION_CONTENTS_FIELD);

      newFieldOperationFieldConstant(
          node,
          true,
          fieldReadAction(getPointerKeyForLocal(instruction.getDef())),
          instruction.getObjectRef(),
          new InstanceKey[] {contentsKey, operationKey});
    }

    /**
     * Models {@code xs.append(v)} as a property write of {@code v} onto {@code xs} under the
     * synthetic {@value #LIST_APPEND_CONTENTS_FIELD} key, so values accumulated through {@code
     * append} surface when the collection is iterated ({@code visitForElementGet} reads the
     * cataloged properties of value-iterated collections). Nothing else models {@code list.append},
     * so without this the appended values are unreachable from the collection and every value
     * flowing through an append-accumulate-iterate chain unravels (wala/ML#570, wala/ML#618).
     *
     * <p>Detection is syntactic on the du-chain: an invoke whose callee value is a property read of
     * the constant name {@code append}. The property write coexists with the invoke's normal
     * dispatch, so a user-defined {@code append} method still dispatches; such a receiver merely
     * gains a stray property under the synthetic key, which is only observable if that same object
     * is also value-iterated.
     *
     * @param inst the invoke instruction to examine
     */
    private void processListAppend(PythonInvokeInstruction inst) {
      if (inst.getNumberOfPositionalParameters() != 2) return;

      SSAInstruction calleeDef = du.getDef(inst.getUse(0));
      if (!(calleeDef instanceof AstPropertyRead)) return;

      AstPropertyRead read = (AstPropertyRead) calleeDef;
      SymbolTable symtab = ir.getSymbolTable();
      if (!symtab.isConstant(read.getMemberRef())
          || !"append".equals(symtab.getConstantValue(read.getMemberRef()))) return;

      InstanceKey contentsKey =
          getBuilder().getInstanceKeyForConstant(PythonTypes.string, LIST_APPEND_CONTENTS_FIELD);
      int valueVn = inst.getUse(1);

      if (contentsAreInvariant(symtab, du, valueVn)) {
        // The invoke's own argument processing records an invariant value's pointer key as
        // implicitly represented, so a raw constraint on that key crashes
        // `findOrCreatePointsToSet` (wala/ML#668). Write the invariant instance keys directly.
        system.recordImplicitPointsToSet(getPointerKeyForLocal(valueVn));
        newFieldWrite(
            node,
            read.getObjectRef(),
            new InstanceKey[] {contentsKey},
            getInvariantContents(symtab, du, node, valueVn));
      } else
        newFieldWrite(
            node,
            read.getObjectRef(),
            new InstanceKey[] {contentsKey},
            getPointerKeyForLocal(valueVn));
    }

    /**
     * A text read yields strings: {@code f.read()} and {@code f.readline()} on the object {@code
     * open} returns yield a string, and {@code f.readlines()} on it, or {@code s.splitlines()} and
     * {@code s.split(...)} on a string, yield a list of strings. The receiver is checked when its
     * points-to set arrives, so a method of the same name on another object is untouched. Without
     * this a text dataset built from a file's lines carried no element type at all.
     *
     * @param inst The call.
     */
    private void processTextRead(PythonInvokeInstruction inst) {
      SSAInstruction calleeDef = du.getDef(inst.getUse(0));
      if (!(calleeDef instanceof AstPropertyRead)) return;
      AstPropertyRead read = (AstPropertyRead) calleeDef;
      SymbolTable symtab = ir.getSymbolTable();
      if (!symtab.isConstant(read.getMemberRef())) return;
      Object member = symtab.getConstantValue(read.getMemberRef());
      TextReadOperator.Kind kind;
      if ("read".equals(member) || "readline".equals(member))
        kind = TextReadOperator.Kind.FILE_TEXT;
      else if ("readlines".equals(member)) kind = TextReadOperator.Kind.FILE_LINES;
      else if ("splitlines".equals(member) || "split".equals(member))
        kind = TextReadOperator.Kind.STRING_PIECES;
      else return;
      TextReadOperator operator =
          getBuilder()
          .new TextReadOperator(node, inst.iIndex(), getPointerKeyForLocal(inst.getDef()), kind);
      int receiverVn = read.getObjectRef();
      PointerKey receiverKey = getPointerKeyForLocal(receiverVn);
      // A literal receiver (`"a,b".split(",")`) has an implicitly represented key, which a side
      // effect must not touch (wala/ML#668): read its contents here and apply the operator once.
      if (contentsAreInvariant(symtab, du, receiverVn) || system.isImplicit(receiverKey)) {
        operator.apply(getInvariantContents(symtab, du, node, receiverVn));
        return;
      }
      system.newSideEffect(operator, receiverKey);
    }

    /**
     * Binop allocation synthesis is intentionally a no-op. An earlier version of this method
     * registered a per-instruction {@link NewSiteReference} (keyed by the SSA instruction index) so
     * the pointer analysis could track binop results (wala/ML#398). That fixed {@code
     * testBinopThroughDataset} but caused a regression elsewhere: seeding the binop def's PTS with
     * a non-tensor-producing {@code Lobject} instance key suppressed tensor identification for
     * downstream values, dropping counts on {@code testNeuralNetwork}, {@code testAutoencoder*},
     * and similar. Preserving tensor identification is a merge-safety invariant against {@code
     * master}, so the allocation is disabled pending a narrower approach. The {@link
     * PythonBinaryOpInstruction} scaffolding and factory override remain in place so a future
     * gated-allocation strategy can hook here. See wala/ML#398.
     */
    @Override
    public void visitPythonBinaryOp(PythonBinaryOpInstruction binop) {
      // List repetition and concatenation (wala/ML#960): `xs * n` and `xs + ys` produce a fresh
      // list (or tuple) whose elements are the operands' elements. Only a list or tuple key
      // flowing into an operand produces anything, so a tensor binop keeps its empty result set
      // and tensor identification is untouched (the wala/ML#398 regression class). The fresh key
      // is allocated at this instruction's index, so `xs + ys` over two lists yields ONE fresh
      // list, and the elements go to the synthetic {@value #LIST_OPERATION_CONTENTS_FIELD} field
      // that every non-constant subscript, iteration and `zip` read beside the append contents:
      // indices are unknowable here, a numeric field would let a length-counting reader derive a
      // wrong extent, and the append field would let the shape readers that interpret appended
      // contents derive one.
      IBinaryOpInstruction.IOperator operator = binop.getOperator();
      if (operator != IBinaryOpInstruction.Operator.ADD
          && operator != IBinaryOpInstruction.Operator.MUL) return;
      PointerKey resultKey = getPointerKeyForLocal(binop.getDef());
      SymbolTable symtab = ir.getSymbolTable();
      int[] operands = {binop.getUse(0), binop.getUse(1)};
      PointerKey[] keys = new PointerKey[2];
      InstanceKey[][] invariant = new InstanceKey[2][];
      for (int i = 0; i < 2; i++) {
        int use = operands[i];
        if (use <= 0 || symtab.isConstant(use)) continue; // the multiplier, or a literal
        keys[i] = getPointerKeyForLocal(use);
        // An operand whose contents are invariant (a literal list) or whose key is otherwise
        // represented implicitly (a summary return) is read directly, as the append model does:
        // a constraint over such a key would crash `findOrCreatePointsToSet` (the wala/ML#668
        // trap), and even a side effect would MATERIALIZE the key, turning an implicit parameter
        // or return explicit and changing how the tensor analysis reads it (a parameter with no
        // list evidence at all gained a tensor state that way).
        if (contentsAreInvariant(symtab, du, use) || system.isImplicit(keys[i]))
          invariant[i] = getInvariantContents(symtab, du, node, use);
      }
      for (int i = 0; i < 2; i++) {
        if (keys[i] == null) continue;
        int other = 1 - i;
        ListOperationOperator listOperation =
            getBuilder()
            .new ListOperationOperator(
                node, binop.iIndex(), resultKey, operator, i, keys[other], invariant[other]);
        if (invariant[i] != null) {
          for (InstanceKey key : invariant[i]) listOperation.contribute(key);
        } else {
          // A side effect, not an assignment constraint: the flow graph the tensor dataflow walks
          // includes every unary constraint's edge, and an operand-to-result edge here would carry
          // a tensor operand's state into the result of `x * y` (the wala/ML#405 substrate-leak
          // class). The side effect adds only the fresh key.
          system.newSideEffect(listOperation, keys[i]);
        }
        // A key declined because the other operand has not grown into the rule yet is retried
        // when that operand's set changes, so the operator also watches it (a represented one;
        // an invariant or implicit other operand has static contents and nothing arrives late).
        if (keys[other] != null && invariant[other] == null)
          system.newSideEffect(listOperation, keys[other]);
      }
    }

    @Override
    public void visitArrayLoad(SSAArrayLoadInstruction inst) {
      newFieldRead(node, inst.getArrayRef(), inst.getIndex(), inst.getDef());
    }

    @Override
    public void visitArrayStore(SSAArrayStoreInstruction inst) {
      newFieldWrite(node, inst.getArrayRef(), inst.getIndex(), inst.getValue());
    }

    @Override
    public void visitPropertyRead(AstPropertyRead instruction) {
      super.visitPropertyRead(instruction);
      processListContentsRead(instruction);

      if (this.ir.getSymbolTable().isConstant(instruction.getMemberRef())) {
        Object constantValue =
            this.ir.getSymbolTable().getConstantValue(instruction.getMemberRef());

        if (Objects.equals(constantValue, IMPORT_WILDCARD_CHARACTER)) {
          // We have a wildcard.
          logger.fine(
              "Detected wildcard for " + instruction.getMemberRef() + " in " + instruction + ".");

          processWildcardImports(instruction);
        }

        // check if we are reading from an module initialization script.
        SSAInstruction objRefDef = du.getDef(instruction.getObjectRef());
        logger.finest(
            () ->
                "Found def: "
                    + objRefDef
                    + " for object reference: "
                    + instruction.getObjectRef()
                    + " in instruction: "
                    + instruction
                    + ".");

        if (objRefDef instanceof AstGlobalRead) {
          AstGlobalRead agr = (AstGlobalRead) objRefDef;
          String fieldName = getStrippedDeclaredFieldName(agr);
          logger.finer("Found field name: " + fieldName);

          // if the "receiver" is a module initialization script.
          if (fieldName.endsWith("/" + MODULE_INITIALIZATION_FILENAME))
            try {
              processWildcardImports(instruction, fieldName, constantValue.toString());
            } catch (CancelException e) {
              throw new CancelRuntimeException(e);
            }
        }
      }
    }

    /**
     * Processes the given {@link AstPropertyRead} for any potential wildcard imports being utilized
     * by the instruction.
     *
     * @param instruction The {@link AstPropertyRead} whose definition may depend on a wildcard
     *     import.
     */
    private void processWildcardImports(AstPropertyRead instruction) {
      int objRef = instruction.getObjectRef();
      logger.fine("Seeing if " + objRef + " refers to an import.");

      SSAInstruction def = this.du.getDef(objRef);
      logger.finer("Found definition: " + def + ".");

      TypeName scriptTypeName = this.ir.getMethod().getReference().getDeclaringClass().getName();
      logger.finer("Found script: " + scriptTypeName + ".");

      String scriptName = getScriptName(scriptTypeName);
      logger.fine("Script name is: " + scriptName);
      assert scriptName.endsWith("." + PYTHON_FILE_EXTENSION);

      if (def instanceof SSAInvokeInstruction) {
        // Library case.
        SSAInvokeInstruction invokeInstruction = (SSAInvokeInstruction) def;
        MethodReference declaredTarget = invokeInstruction.getDeclaredTarget();
        Atom declaredTargetName = declaredTarget.getName();

        if (declaredTargetName.equals(IMPORT_FUNCTION_NAME)) {
          // It's an import "statement" importing a library.
          logger.fine("Found library import statement in: " + scriptTypeName + ".");

          logger.info(
              "Adding: "
                  + declaredTarget.getDeclaringClass().getName().toString().substring(1)
                  + " to wildcard imports for: "
                  + scriptName
                  + ".");

          // Add the library to the script's queue of wildcard imports.
          getBuilder()
              .getScriptToWildcardImports()
              .compute(
                  scriptName,
                  (_, v) -> {
                    if (v == null) {
                      Deque<MethodReference> deque = new ArrayDeque<>();
                      deque.push(declaredTarget);
                      return deque;
                    } else {
                      v.push(declaredTarget);
                      return v;
                    }
                  });
        }
      } else if (def instanceof SSAGetInstruction) {
        // We are importing from a script.
        SSAGetInstruction getInstruction = (SSAGetInstruction) def;
        String strippedFieldName = getStrippedDeclaredFieldName(getInstruction);

        MethodReference methodReference = getMethodReferenceRepresentingScript(strippedFieldName);

        logger.info(
            "Adding: "
                + methodReference.getDeclaringClass().getName().toString().substring(1)
                + " to wildcard imports for: "
                + scriptName
                + ".");

        // Add the script to the queue of this script's wildcard imports.
        getBuilder()
            .getScriptToWildcardImports()
            .compute(
                scriptName,
                (_, v) -> {
                  if (v == null) {
                    Deque<MethodReference> deque = new ArrayDeque<>();
                    deque.push(methodReference);
                    return deque;
                  } else {
                    v.push(methodReference);
                    return v;
                  }
                });
      } else if (def instanceof AstPropertyRead) processWildcardImports((AstPropertyRead) def);
      else
        throw new IllegalArgumentException(
            "Not expecting the definition: "
                + def
                + " of the object reference of: "
                + instruction
                + " to be: "
                + def.getClass());
    }

    /**
     * Given a script's name, returns the {@link MethodReference} representing the script.
     *
     * @param scriptName The name of the script.
     * @return The corresponding {@link MethodReference} representing the script.
     */
    private static MethodReference getMethodReferenceRepresentingScript(String scriptName) {
      TypeReference typeReference =
          TypeReference.findOrCreate(PythonTypes.pythonLoader, "L" + scriptName);

      return MethodReference.findOrCreate(
          typeReference,
          Atom.findOrCreateAsciiAtom(DO_METHOD_NAME),
          Descriptor.findOrCreate(null, PythonTypes.rootTypeName));
    }

    @Override
    public void visitAstGlobalRead(AstGlobalRead globalRead) {
      super.visitAstGlobalRead(globalRead);

      TypeName enclosingMethodTypeName =
          this.ir.getMethod().getReference().getDeclaringClass().getName();

      String scriptName = getScriptName(enclosingMethodTypeName);

      if (scriptName.endsWith("." + PYTHON_FILE_EXTENSION)) {
        // We have a valid script name.
        logger.fine("Script name is: " + scriptName);
        String fieldName = getStrippedDeclaredFieldName(globalRead);
        try {
          processWildcardImports(globalRead, scriptName, fieldName);
        } catch (CancelException e) {
          throw new CancelRuntimeException(e);
        }
      }
    }

    /**
     * Returns the name of the script for the given {@link TypeName} representing a the name of a
     * method.
     *
     * @param methodName The name of the method.
     * @return The name of the corresponding script.
     * @implNote In Ariadne, scripts are also "methods" with the name "do."
     */
    private static String getScriptName(TypeName methodName) {
      String composed =
          methodName.getPackage() == null
              ? methodName.getClassName().toString()
              : methodName.getPackage().toString() + "/" + methodName.getClassName().toString();

      // A method nested in a class (e.g. `script layers/feed_forward.py/Conv1d/call`) composes to
      // a name whose script segment is interior, not terminal. Truncate at the script's file
      // extension so reads inside class methods key the same script as module-level reads;
      // without this, the wildcard lookup is skipped for them (wala/ML#665).
      String marker = "." + PYTHON_FILE_EXTENSION + "/";
      int extension = composed.indexOf(marker);
      if (extension >= 0) {
        return composed.substring(0, extension + marker.length() - 1);
      }

      if (composed.endsWith("." + PYTHON_FILE_EXTENSION)) {
        return composed;
      }

      return (methodName.getPackage() == null ? methodName.getClassName() : methodName.getPackage())
          .toString();
    }

    /**
     * Processes the given {@link SSAInstruction} for any potential wildcard imports being utilized
     * by the instruction.
     *
     * @param instruction The {@link SSAInstruction} whose definition may depend on a wildcard
     *     import.
     * @param scriptName The name of the script to check for wildcard imports.
     * @param fieldName The name of the field that may be imported using a wildcard.
     */
    private void processWildcardImports(
        SSAInstruction instruction, String scriptName, String fieldName) throws CancelException {
      // Get the method reference for the given script.
      MethodReference reference = getMethodReferenceRepresentingScript(scriptName);

      // Get the nodes for the script.
      Set<CGNode> scriptNodes = this.getBuilder().getCallGraph().getNodes(reference);

      // For each node representing the script.
      for (CGNode node : scriptNodes) {
        // if we haven't visited the node yet.
        if (!this.getBuilder().haveAlreadyVisited(node)) {
          // visit the node first. Otherwise, we won't know if there are any wildcard imports in
          // it.
          this.getBuilder().addConstraintsFromNode(node, null);

          assert this.getBuilder().haveAlreadyVisited(node);
        }
      }

      // Are there any wildcard imports for this script?
      if (getBuilder().getScriptToWildcardImports().containsKey(scriptName)) {
        logger.info("Found wildcard imports in " + scriptName + " for " + instruction + ".");

        Deque<MethodReference> deque = getBuilder().getScriptToWildcardImports().get(scriptName);

        for (MethodReference importMethodReference : deque) {
          logger.fine(
              "Library with wildcard import is: "
                  + importMethodReference.getDeclaringClass().getName().toString().substring(1)
                  + ".");

          logger.fine("Examining global: " + fieldName + " for wildcard import.");

          CallGraph callGraph = this.getBuilder().getCallGraph();
          Set<CGNode> nodes = callGraph.getNodes(importMethodReference);

          if (nodes.isEmpty())
            logger.warning(
                "Can't find CG node for import method: "
                    + importMethodReference.getSignature()
                    + ".");

          PointerKey defPK = this.getPointerKeyForLocal(instruction.getDef());
          assert defPK != null;

          for (CGNode n : nodes) {
            for (Iterator<NewSiteReference> nit = n.iterateNewSites(); nit.hasNext(); ) {
              NewSiteReference newSiteReference = nit.next();

              String name = newSiteReference.getDeclaredType().getName().getClassName().toString();
              logger.finest("Examining: " + name + ".");

              if (name.equals(fieldName)) {
                logger.info("Found wildcard import for: " + name + ".");

                InstanceKey instanceKey =
                    this.getBuilder().getInstanceKeyForAllocation(n, newSiteReference);

                if (this.system.newConstraint(defPK, instanceKey)) {
                  logger.fine("Added constraint that: " + defPK + " gets: " + instanceKey + ".");
                  return;
                }
              }
            }

            // Also check the put instructions, as these may be generated by the initialization
            // file.
            n.getIR()
                .visitNormalInstructions(
                    new PythonInstructionVisitor() {

                      @Override
                      public void visitPut(SSAPutInstruction putInstruction) {
                        FieldReference putField = putInstruction.getDeclaredField();

                        if (fieldName.equals(putField.getName().toString())) {
                          // Found it.
                          int putVal = putInstruction.getVal();

                          // Make the def point to the put instruction value.
                          PointerKey putValPK = getBuilder().getPointerKeyForLocal(n, putVal);

                          if (system.newConstraint(defPK, assignOperator, putValPK))
                            logger.fine(
                                "Added constraint that: " + defPK + " gets: " + putValPK + ".");
                        }
                      }
                    });

            // Also check the module's named bindings (wala/ML#665). Python's wildcard exports
            // every public module-level name, including modules the source module itself
            // imported (`import tensorflow as tf` makes `tf` an exported binding), and such a
            // binding is neither an allocation named after the field (the `def`/`class` case
            // above) nor a `put` onto a module object: it is a named local of the script body
            // holding the import's result. Match the script's local names and assign the
            // binding's value to the reader.
            if (n.getMethod() instanceof AstMethod) {
              for (Iterator<SSAInstruction> it = n.getIR().iterateAllInstructions();
                  it.hasNext(); ) {
                SSAInstruction inst = it.next();
                if (!inst.hasDef() || inst.iIndex() < 0) continue;

                String[] localNames = n.getIR().getLocalNames(inst.iIndex(), inst.getDef());
                if (localNames == null) continue;

                for (String localName : localNames) {
                  if (fieldName.equals(localName)) {
                    PointerKey boundValuePK = getBuilder().getPointerKeyForLocal(n, inst.getDef());

                    if (system.newConstraint(defPK, assignOperator, boundValuePK))
                      logger.fine(
                          "Added wildcard binding constraint that: "
                              + defPK
                              + " gets: "
                              + boundValuePK
                              + " (wala/ML#665).");
                  }
                }
              }
            }
          }
        }
      }
    }

    private static String getStrippedDeclaredFieldName(SSAGetInstruction instruction) {
      String declaredFieldName = instruction.getDeclaredField().getName().toString();
      assert declaredFieldName.startsWith(GLOBAL_IDENTIFIER + " ");

      // Remove the global identifier.
      return declaredFieldName.substring(
          (GLOBAL_IDENTIFIER + " ").length(), declaredFieldName.length());
    }
  }

  @Override
  protected void processCallingConstraints(
      CGNode caller,
      SSAAbstractInvokeInstruction instruction,
      CGNode target,
      InstanceKey[][] constParams,
      PointerKey uniqueCatchKey) {

    if (!(instruction instanceof PythonInvokeInstruction)) {
      super.processCallingConstraints(caller, instruction, target, constParams, uniqueCatchKey);
    } else {
      MutableIntSet args = IntSetUtil.make();

      // positional parameters
      PythonInvokeInstruction call = (PythonInvokeInstruction) instruction;
      for (int i = 0;
          i < call.getNumberOfPositionalParameters()
              && i < target.getMethod().getNumberOfParameters();
          i++) {
        PointerKey lval = getPointerKeyForLocal(target, i + 1);
        args.add(i);

        if (constParams != null && constParams[i] != null) {
          InstanceKey[] ik = constParams[i];
          for (InstanceKey element : ik) {
            system.newConstraint(lval, element);
          }
        } else {
          PointerKey rval = getPointerKeyForLocal(caller, call.getUse(i));

          // If we are looking at the implicit parameter of a callable.
          if (call.getCallSite().isDispatch()
              && isCallable(target.getMethod().getReference())
              && i == 0
              && refersToAnObject(rval)) {
            // Ensure that lval's variable refers to the callable method instead of callable object,
            // precisely linking the object to its trampoline using the `__call__` (or similar)
            // field. When the target is a trampoline keyed on a receiver instance (wala/ML#679),
            // link only that receiver's field; the site's other receivers dispatch to their own
            // trampoline nodes. Non-trampoline targets may inherit a receiver-keyed context whose
            // instance is unrelated to this site's receivers, so they are not restricted.
            ContextItem receiverItem =
                target.getMethod().getDeclaringClass() instanceof PythonInstanceMethodTrampoline
                    ? target.getContext().get(ContextKey.RECEIVER)
                    : null;
            InstanceKey contextReceiver =
                receiverItem instanceof InstanceKey ? (InstanceKey) receiverItem : null;

            IClassHierarchy cha = getClassHierarchy();

            Atom[] possibleFields = {
              Atom.findOrCreateUnicodeAtom(CALLABLE_METHOD_NAME),
              Atom.findOrCreateUnicodeAtom(CALLABLE_METHOD_NAME_FOR_KERAS_MODELS),
              Atom.findOrCreateUnicodeAtom(DO_METHOD_NAME)
            };

            getSystem()
                .newSideEffect(
                    new AbstractOperator<PointsToSetVariable>() {
                      @Override
                      public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable[] rhs) {
                        if (rhs[0].getValue() != null) {
                          rhs[0]
                              .getValue()
                              .foreach(
                                  i -> {
                                    InstanceKey ik = getSystem().getInstanceKey(i);
                                    if (contextReceiver != null && !contextReceiver.equals(ik))
                                      return;

                                    for (Atom fieldName : possibleFields) {
                                      FieldReference fieldRef =
                                          FieldReference.findOrCreate(
                                              PythonTypes.Root, fieldName, PythonTypes.Root);

                                      IField f = cha.resolveField(fieldRef);

                                      if (f != null) {
                                        PointerKey fieldPK =
                                            getPointerKeyFactory()
                                                .getPointerKeyForInstanceField(ik, f);

                                        getSystem().newConstraint(lval, assignOperator, fieldPK);
                                      }
                                    }
                                  });
                        }
                        return NOT_CHANGED;
                      }

                      @Override
                      public int hashCode() {
                        return lval.hashCode() ^ rval.hashCode();
                      }

                      @Override
                      public boolean equals(Object o) {
                        return this == o;
                      }

                      @Override
                      public String toString() {
                        return "precise trampoline link for " + rval;
                      }
                    },
                    new PointerKey[] {rval});
          } else {
            // A receiver-keyed target (wala/ML#679) filters the dispatched parameter to the
            // context's receiver instance; every other target binds the full argument set.
            PointerKey formal = i == 0 ? getReceiverFilteredPointerKey(target, lval) : lval;
            getSystem()
                .newConstraint(
                    formal,
                    formal instanceof FilteredPointerKey ? filterOperator : assignOperator,
                    rval);
          }
        }
      }

      // keyword arguments
      int paramNumber = call.getNumberOfPositionalParameters();
      keywords:
      for (String argName : call.getKeywords()) {
        int src = call.getUse(argName);
        for (int i = 0; i < target.getIR().getSymbolTable().getMaxValueNumber(); i++) {
          String[] paramNames = target.getIR().getLocalNames(0, i + 1);
          if (paramNames != null) {
            for (String destName : paramNames) {
              if (argName.equals(destName)) {
                PointerKey lval = getPointerKeyForLocal(target, i + 1);
                args.add(i);
                int p = paramNumber;
                if (constParams != null && constParams[p] != null) {
                  InstanceKey[] ik = constParams[p];
                  for (InstanceKey element : ik) {
                    system.newConstraint(lval, element);
                  }
                } else {
                  PointerKey rval = getPointerKeyForLocal(caller, src);
                  getSystem().newConstraint(lval, assignOperator, rval);
                }
                paramNumber++;
                continue keywords;
              }
            }
          }
        }
        // no such argument in callee
        paramNumber++;
      }

      // A synthesized constructor's summary declares the wrapped `__init__`'s parameter count,
      // although its own formals stop one short (formal i mirrors `__init__`'s parameter i + 1),
      // so its defaulted range shifts down by one (wala/ML#762).
      boolean ctorTarget = target.getMethod() instanceof PythonConstructorFunction;
      int numParams = target.getMethod().getNumberOfParameters() - (ctorTarget ? 1 : 0);
      // The `*args`, `**kwargs`, and keyword-only formals are parameters but cannot receive a
      // positional default, and the parser appends them after the plain positionals. Counting the
      // defaulted range back from the end of the formal list therefore has to discount them, or
      // every default binds one parameter to the right and the first defaulted parameter binds
      // nothing (wala/ML#843). The writer in `PythonCAstToIRTranslator` discounts them the same
      // way, so the two sides agree on which index carries which default.
      int trailing =
          target.getMethod() instanceof StarFormalDeclaration
              ? ((StarFormalDeclaration) target.getMethod())
                  .getNumberOfTrailingNonDefaultableParameters()
              : 0;
      int last = numParams - trailing;
      int dflts = last - target.getMethod().getNumberOfDefaultParameters();
      for (int i = dflts; i < last; i++) {
        if (!args.contains(i)) {
          // A synthesized constructor's trailing formal i mirrors `__init__`'s parameter i + 1,
          // and the default globals are written under `__init__`'s entity name, so the lookup
          // follows that mapping (wala/ML#762). Every other target reads its own entity's global.
          String name = defaultsGlobalName(target, i, "_defaults_");
          IField f = resolveGlobal(name);
          logger.fine(
              "DEFAULTS-BIND target " + target + " param " + i + " global " + name + " field " + f);
          PointerKey lval = getPointerKeyForLocal(target, i + 1);
          getSystem().newConstraint(lval, assignOperator, new StaticFieldKey(f));
          // A `@click.option` default is written under a global of its own and binds under a
          // constant key of its own class (wala/ML#971); the global of a parameter with no click
          // option holds nothing and contributes nothing. Globals are dynamic fields of `Root`, so
          // the name always resolves.
          String clickName = defaultsGlobalName(target, i, "_click_defaults_");
          getSystem()
              .newConstraint(
                  lval, new ClickDefaultOperator(), new StaticFieldKey(resolveGlobal(clickName)));
        }
      }

      // return values
      PointerKey rret = getPointerKeyForReturnValue(target);
      PointerKey lret = getPointerKeyForLocal(caller, call.getReturnValue(0));
      getSystem().newConstraint(lret, assignOperator, rret);

      PointerKey reret = getPointerKeyForExceptionalReturnValue(target);
      PointerKey leret = getPointerKeyForLocal(caller, call.getException());
      getSystem().newConstraint(leret, assignOperator, reret);

      if (target.getMethod().getDeclaringClass().getReference().equals(PythonTypes.SLICE_BUILTIN))
        processSliceResult(caller, call, constParams);
    }
  }

  /**
   * Supplies the result of a {@code slice} builtin call (wala/ML#916). The builtin's body returns
   * nothing; the result is the first argument's points-to set, as the body used to return, except
   * that a key whose concrete type is one of the {@link #freshSliceResultTypes} becomes a fresh
   * allocation of that type at this call, so a tensor's slice is a tensor of its own rather than an
   * alias of its receiver. The result is a unary constraint from the receiver to the result, an
   * edge of the assignment graph like the one the body used to make through its parameter and
   * return, because the tensor dataflow analysis uses that graph as its flow graph: a side effect
   * would supply the same keys but sever the edge, and every value flowing through a slice of a
   * pass-through receiver (an array's dtype state, a named tuple's element types) would stop at the
   * call. A constant first argument (the {@code slice(None, n, None)} form a subscript's bounds
   * lower to) flows as the constant, as before.
   *
   * @param caller The node containing the call.
   * @param call The {@code slice} call.
   * @param constParams The call's constant arguments, indexed by positional argument, or {@code
   *     null}.
   */
  private void processSliceResult(
      CGNode caller, PythonInvokeInstruction call, InstanceKey[][] constParams) {
    if (call.getNumberOfPositionalParameters() < 2 || !call.hasDef()) return;
    PointerKey def = getPointerKeyForLocal(caller, call.getDef());
    if (constParams != null && constParams.length > 1 && constParams[1] != null) {
      for (InstanceKey element : constParams[1]) system.newConstraint(def, element);
      return;
    }
    PointerKey receiver = getPointerKeyForLocal(caller, call.getUse(1));
    getSystem().newConstraint(def, new SliceResultOperator(caller, call.iIndex()), receiver);
  }

  /**
   * Mints the result of a text read once its receiver is known (see {@code processTextRead}): a
   * string constant for a file's {@code read} or {@code readline}, and a fresh list holding a
   * string constant under index {@code 0} for a file's {@code readlines} or a string's {@code
   * splitlines} and {@code split}. The list is populated as a literal list is (a catalog entry and
   * a numbered field), so every element reader sees the string.
   */
  public final class TextReadOperator extends UnaryOperator<PointsToSetVariable> {
    public enum Kind {
      FILE_TEXT,
      FILE_LINES,
      STRING_PIECES
    }

    private static final String TEXT = "<text>";
    private final CGNode node;
    private final int pc;
    private final PointerKey resultKey;
    private final Kind kind;
    private boolean contributed;

    private TextReadOperator(CGNode node, int pc, PointerKey resultKey, Kind kind) {
      this.node = node;
      this.pc = pc;
      this.resultKey = resultKey;
      this.kind = kind;
    }

    @Override
    public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
      if (contributed || rhs.getValue() == null) return NOT_CHANGED;
      java.util.List<InstanceKey> receivers = new java.util.ArrayList<>();
      rhs.getValue().foreach(i -> receivers.add(getSystem().getInstanceKey(i)));
      apply(receivers.toArray(new InstanceKey[0]));
      return NOT_CHANGED;
    }

    /**
     * Applies the read once some receiver matches its kind.
     *
     * @param receivers The receiver's instance keys.
     */
    void apply(InstanceKey[] receivers) {
      if (contributed || receivers == null) return;
      boolean applies = false;
      for (InstanceKey receiver : receivers) if (receiverMatches(receiver)) applies = true;
      if (!applies) return;
      contributed = true;
      logger.fine(() -> "text read at " + pc + " in " + node + ": " + kind + " applies.");
      InstanceKey text = getInstanceKeyForConstant(PythonTypes.string, TEXT);
      if (kind == Kind.FILE_TEXT) {
        getSystem().newConstraint(resultKey, text);
        return;
      }
      InstanceKey list =
          getInstanceKeyForAllocation(node, NewSiteReference.make(pc, PythonTypes.list));
      if (list == null) return;
      AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
      IField zero = resolveRootField(getClassHierarchy(), "0");
      if (zero == null) return;
      getSystem().newConstraint(resultKey, list);
      getSystem()
          .newConstraint(
              factory.getPointerKeyForObjectCatalog(list),
              getInstanceKeyForConstant(PythonLanguage.Python.getConstantType(0), 0));
      getSystem().newConstraint(factory.getPointerKeyForInstanceField(list, zero), text);
      logger.fine(() -> "text read at " + pc + " in " + node + ": " + kind + " -> " + list);
    }

    private boolean receiverMatches(InstanceKey receiver) {
      if (kind == Kind.STRING_PIECES)
        return receiver instanceof ConstantKey
            && ((ConstantKey<?>) receiver).getValue() instanceof String;
      IClassHierarchy cha = getClassHierarchy();
      IClass fileClass = cha.lookupClass(PythonTypes.file);
      return fileClass != null && cha.isSubclassOf(receiver.concreteType(), fileClass);
    }

    @Override
    public int hashCode() {
      return (node.hashCode() * 31 + pc) * 31 + kind.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof TextReadOperator
          && ((TextReadOperator) o).node.equals(node)
          && ((TextReadOperator) o).pc == pc
          && ((TextReadOperator) o).kind == kind;
    }

    @Override
    public String toString() {
      return "text read " + kind + " at " + pc + " in " + node;
    }
  }

  /**
   * The result of a list repetition or concatenation (wala/ML#960), attached to ONE operand: for
   * each list or tuple key flowing into that operand, when the operation's rule holds against the
   * other operand's contents, a fresh key of the same type allocated at the binop's instruction
   * index joins the result, and the operand key's elements, every field its object catalog names
   * plus its own appended and operation contents, flow into the fresh key's synthetic {@value
   * #LIST_OPERATION_CONTENTS_FIELD} field, which non-constant subscripts, iteration and {@code zip}
   * read and no shape reader interprets. Keys of other types contribute nothing, and a key already
   * contributed is not registered again.
   *
   * <p>The rule: {@code ADD} fires only when the other operand also carries a list or tuple (list
   * plus list); a list beside a tensor or an ndarray is that object's own addition, whose result is
   * no list. {@code MUL} fires only when the other operand carries neither a list or tuple nor a
   * tensor-like value (a tensor or an ndarray, by their model packages): repetition takes an
   * integer, and a list times a tensor is the tensor's multiplication.
   *
   * <p>A key declined because the other operand has not met the rule yet is kept pending and
   * retried when that operand's set changes. The converse is the monotone limit: a repetition
   * contributed while the other operand held only integers is not retracted if that operand later
   * grows a tensor member.
   */
  public final class ListOperationOperator extends UnaryOperator<PointsToSetVariable> {
    private final CGNode node;
    private final int pc;
    private final PointerKey resultKey;
    private final IBinaryOpInstruction.IOperator operator;
    private final int operandIndex;

    /** The other operand's key, or {@code null} for a constant (the multiplier). */
    private final PointerKey otherKey;

    /** The other operand's contents when invariant or implicit, else {@code null}. */
    private final InstanceKey[] otherInvariant;

    private final Set<InstanceKey> contributed = HashSetFactory.make();

    /** Keys of this operand declined so far because the other operand had not met the rule. */
    private final Set<InstanceKey> pending = HashSetFactory.make();

    private ListOperationOperator(
        CGNode node,
        int pc,
        PointerKey resultKey,
        IBinaryOpInstruction.IOperator operator,
        int operandIndex,
        PointerKey otherKey,
        InstanceKey[] otherInvariant) {
      this.node = node;
      this.pc = pc;
      this.resultKey = resultKey;
      this.operator = operator;
      this.operandIndex = operandIndex;
      this.otherKey = otherKey;
      this.otherInvariant = otherInvariant;
    }

    @Override
    public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
      if (rhs.getValue() == null) return NOT_CHANGED;
      if (otherKey != null && otherKey.equals(rhs.getPointerKey())) {
        // The other operand grew: retry this operand's keys declined before.
        for (InstanceKey key : new java.util.ArrayList<>(pending)) contribute(key);
        return NOT_CHANGED;
      }
      rhs.getValue().foreach(i -> contribute(getSystem().getInstanceKey(i)));
      return NOT_CHANGED;
    }

    /**
     * Adds the fresh key an operand key contributes to the result, if any and if the operation's
     * rule holds against the other operand, to the result's points-to set by an instance
     * constraint, which is not an edge of the flow graph. A key seen before contributes nothing
     * again.
     *
     * @param key An operand's instance key.
     */
    private void contribute(InstanceKey key) {
      if (!isListOrTuple(key) || contributed.contains(key)) return;
      if (!ruleHolds()) {
        pending.add(key); // retried when the other operand's set changes
        return;
      }
      pending.remove(key);
      contributed.add(key);
      InstanceKey fresh =
          getInstanceKeyForAllocation(
              node, NewSiteReference.make(pc, key.concreteType().getReference()));
      if (fresh == null) return;
      getSystem().newConstraint(resultKey, fresh);
      copyElements(key, fresh);
    }

    /** Whether the operation's rule holds against the other operand's current contents. */
    private boolean ruleHolds() {
      boolean otherList = false;
      boolean otherTensorLike = false;
      Iterable<InstanceKey> contents =
          otherInvariant == null ? null : java.util.Arrays.asList(otherInvariant);
      if (contents == null && otherKey != null) {
        PointsToSetVariable v = getSystem().findOrCreatePointsToSet(otherKey);
        java.util.List<InstanceKey> keys = new java.util.ArrayList<>();
        if (v.getValue() != null)
          v.getValue().foreach(i -> keys.add(getSystem().getInstanceKey(i)));
        contents = keys;
      }
      if (contents != null)
        for (InstanceKey ik : contents) {
          if (isListOrTuple(ik)) otherList = true;
          else if (isTensorLike(ik)) otherTensorLike = true;
        }
      if (operator == IBinaryOpInstruction.Operator.ADD) return otherList;
      return !otherList && !otherTensorLike;
    }

    private boolean isListOrTuple(InstanceKey key) {
      IClassHierarchy cha = getClassHierarchy();
      IClass type = key.concreteType();
      return cha.isSubclassOf(type, cha.lookupClass(PythonTypes.list))
          || cha.isSubclassOf(type, cha.lookupClass(PythonTypes.tuple));
    }

    /** A tensor or an ndarray, by the model packages that declare them. */
    private boolean isTensorLike(InstanceKey key) {
      String name = key.concreteType().getName().toString();
      return name.startsWith("Ltensorflow/") || name.startsWith("Lnumpy/");
    }

    /**
     * Flows every catalogued field of {@code from}, and its appended and operation contents, into
     * {@code to}'s operation-contents field. The catalog is read by a side effect with value
     * equality on (from, to), so re-evaluation registers nothing twice.
     */
    private void copyElements(InstanceKey from, InstanceKey to) {
      AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
      IClassHierarchy cha = getClassHierarchy();
      IField contents = resolveRootField(cha, LIST_OPERATION_CONTENTS_FIELD);
      IField appended = resolveRootField(cha, LIST_APPEND_CONTENTS_FIELD);
      if (contents == null || appended == null) return;
      PointerKey contentsKey = factory.getPointerKeyForInstanceField(to, contents);
      logger.fine(() -> "list operation at " + pc + " in " + node + ": " + to + " from " + from);
      // The operand's own appended contents and operation contents flow through unchanged, so a
      // chain of operations, or an operation over an appended list, keeps every element.
      getSystem()
          .newConstraint(
              contentsKey, assignOperator, factory.getPointerKeyForInstanceField(from, contents));
      getSystem()
          .newConstraint(
              contentsKey, assignOperator, factory.getPointerKeyForInstanceField(from, appended));
      getSystem()
          .newSideEffect(
              new ElementCopyOperator(from, contentsKey, pc),
              factory.getPointerKeyForObjectCatalog(from));
    }

    @Override
    public int hashCode() {
      return ((node.hashCode() * 31 + pc) * 31 + resultKey.hashCode()) * 31 + operandIndex;
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof ListOperationOperator
          && ((ListOperationOperator) o).node.equals(node)
          && ((ListOperationOperator) o).pc == pc
          && ((ListOperationOperator) o).resultKey.equals(resultKey)
          && ((ListOperationOperator) o).operandIndex == operandIndex;
    }

    @Override
    public String toString() {
      return "list operation result at " + pc + " in " + node + " (operand " + operandIndex + ")";
    }
  }

  private static IField resolveRootField(IClassHierarchy cha, String name) {
    return cha.resolveField(
        FieldReference.findOrCreate(
            PythonTypes.Root, Atom.findOrCreateUnicodeAtom(name), PythonTypes.Root));
  }

  /**
   * Flows every catalogued field of a list key into a contents field as the catalog's names arrive
   * (wala/ML#960). Equal for the same (from, contents) pair, so the propagation system keeps one.
   */
  private final class ElementCopyOperator extends UnaryOperator<PointsToSetVariable> {
    private final InstanceKey from;
    private final PointerKey contentsKey;
    private final int pc;

    private ElementCopyOperator(InstanceKey from, PointerKey contentsKey, int pc) {
      this.from = from;
      this.contentsKey = contentsKey;
      this.pc = pc;
    }

    @Override
    public byte evaluate(PointsToSetVariable l, PointsToSetVariable catalog) {
      if (catalog.getValue() == null) return NOT_CHANGED;
      AstPointerKeyFactory factory = (AstPointerKeyFactory) getPointerKeyFactory();
      IClassHierarchy cha = getClassHierarchy();
      catalog
          .getValue()
          .foreach(
              c -> {
                InstanceKey nameKey = getSystem().getInstanceKey(c);
                if (!(nameKey instanceof ConstantKey)) return;
                Object value = ((ConstantKey<?>) nameKey).getValue();
                // A literal's element fields are named by integer constants, a dictionary's by
                // strings; both name a field.
                if (!(value instanceof String) && !(value instanceof Number)) return;
                String name = value.toString();
                IField f = resolveRootField(cha, name);
                if (f == null) return;
                logger.fine(
                    () ->
                        "list operation at "
                            + pc
                            + ": field "
                            + name
                            + " of "
                            + from
                            + " into "
                            + contentsKey);
                getSystem()
                    .newConstraint(
                        contentsKey,
                        assignOperator,
                        factory.getPointerKeyForInstanceField(from, f));
              });
      return NOT_CHANGED;
    }

    @Override
    public int hashCode() {
      return from.hashCode() * 31 + contentsKey.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof ElementCopyOperator
          && ((ElementCopyOperator) o).from.equals(from)
          && ((ElementCopyOperator) o).contentsKey.equals(contentsKey);
    }

    @Override
    public String toString() {
      return "list-operation elements of " + from + " into " + contentsKey;
    }
  }

  /**
   * The assignment from a {@code slice} call's receiver to its result (wala/ML#916): every key
   * passes through except one of a {@link #freshSliceResultTypes fresh type}, which becomes the
   * fresh allocation of that type at the call. Equal for the same call, so the constraint is
   * idempotent like an assignment.
   *
   * <p>Every key of a fresh type in the receiver's set maps to the ONE allocation of that type at
   * this call, so two distinct tensors reaching the slice through a merge are one object after it.
   * That is a deliberate precision choice, not an accident: the result's shape and dtype are
   * resolved from the call itself (the slice operation over the whole receiver), never from the
   * individual key, so per-key allocations would multiply objects without sharpening a reading.
   *
   * <p>When the instance-key factory declines the allocation (a {@code null} key), the receiver's
   * key passes through instead: the result then aliases its receiver at that key exactly as before
   * this change, a decline rather than a failure inside the solver.
   */
  private final class SliceResultOperator extends UnaryOperator<PointsToSetVariable> {
    private final CGNode caller;
    private final int pc;

    private SliceResultOperator(CGNode caller, int pc) {
      this.caller = caller;
      this.pc = pc;
    }

    @Override
    public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
      if (rhs.getValue() == null) return NOT_CHANGED;
      Set<TypeReference> fresh = freshSliceResultTypes;
      MutableIntSet out = IntSetUtil.make();
      rhs.getValue()
          .foreach(
              i -> {
                InstanceKey key = getSystem().getInstanceKey(i);
                TypeReference type = key.concreteType().getReference();
                InstanceKey allocation =
                    fresh.contains(type)
                        ? getInstanceKeyForAllocation(caller, NewSiteReference.make(pc, type))
                        : null;
                // A declined allocation (null) falls back to the receiver's own key: the type came
                // off an existing key, so the class resolves and this is not expected to happen,
                // but a null inside the solver would take the whole analysis down (the wala/ML#925
                // class) where aliasing the receiver merely loses this change at one key.
                out.add(
                    allocation == null
                        ? i
                        : getSystem().findOrCreateIndexForInstanceKey(allocation));
              });
      return lhs.addAll(out) ? CHANGED : NOT_CHANGED;
    }

    @Override
    public int hashCode() {
      return caller.hashCode() * 31 + pc;
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof SliceResultOperator
          && ((SliceResultOperator) o).caller.equals(caller)
          && ((SliceResultOperator) o).pc == pc;
    }

    @Override
    public String toString() {
      return "slice result at " + pc + " in " + caller;
    }
  }

  /**
   * Returns the {@link PointerKey} to bind for the given target's dispatched (position-0)
   * parameter, honoring a receiver filter the target's context carries (<a
   * href="https://github.com/wala/ML/issues/679">wala/ML#679</a>). The filter applies only to
   * trampoline targets: their first parameter <em>is</em> the dispatched object the context is
   * keyed on. A real method body inheriting a per-receiver context has the same context item, but
   * its first parameter is the function object, which the receiver filter would wrongly empty.
   *
   * @param target The callee {@link CGNode}.
   * @param dflt The unfiltered {@link PointerKey} for the target's first parameter.
   * @return A {@link FilteredPointerKey} restricted to the context's receiver when the target is a
   *     trampoline whose context supplies a parameter-0 filter; otherwise {@code dflt}.
   */
  private PointerKey getReceiverFilteredPointerKey(CGNode target, PointerKey dflt) {
    if (!(target.getMethod().getDeclaringClass() instanceof PythonInstanceMethodTrampoline))
      return dflt;
    TypeFilter filter = (TypeFilter) target.getContext().get(ContextKey.PARAMETERS[0]);
    if (filter != null && !filter.isRootFilter())
      return getFilteredPointerKeyForLocal(target, 1, filter);
    return dflt;
  }

  /**
   * Returns true iff the given {@link MethodReference} is a "callable" method, i.e., a method that
   * is used to implement the __call__ functionality of a callable object.
   *
   * @param methodReference The {@link MethodReference} in question.
   * @return True iff the given {@link MethodReference} is a "callable" method.
   */
  private static boolean isCallable(MethodReference methodReference) {
    String name = methodReference.getDeclaringClass().getName().toString();
    return name.endsWith(CALLABLE_METHOD_NAME)
        || name.endsWith(CALLABLE_METHOD_NAME_FOR_KERAS_MODELS);
  }

  /**
   * Returns true iff the given {@link PointerKey} points to at least one instance whose concrete
   * type equals {@link PythonTypes#object}.
   *
   * @param pointerKey The {@link PointerKey} in question.
   * @return True iff the given {@link PointerKey} points to at least one object whose concrete type
   *     equals {@link PythonTypes#object}.,
   */
  protected boolean refersToAnObject(PointerKey pointerKey) {
    PointerAnalysis<InstanceKey> pointerAnalysis = this.getPointerAnalysis();
    OrdinalSet<InstanceKey> pointsToSet = pointerAnalysis.getPointsToSet(pointerKey);

    for (InstanceKey instanceKey : pointsToSet) {
      IClass concreteType = instanceKey.concreteType();
      TypeReference reference = concreteType.getReference();

      // If it's an "object" method.
      if (reference.equals(PythonTypes.object)) return true;

      // Handle synthetic classes (e.g., from XML summaries) which inherit from object
      // but are not functions or trampolines.
      IClassHierarchy cha = pointerAnalysis.getClassHierarchy();
      IClass objClass = cha.lookupClass(PythonTypes.object);
      IClass trampClass = cha.lookupClass(PythonTypes.trampoline);

      if (objClass != null && cha.isSubclassOf(concreteType, objClass)) {
        if (trampClass == null || !cha.isSubclassOf(concreteType, trampClass)) {
          // Do not treat generated trampoline classes (which contain '$') as generic objects
          if (!concreteType.getName().toString().contains("$")) {
            return true;
          }
        }
      }
    }

    return false;
  }

  @Override
  public PythonConstraintVisitor makeVisitor(CGNode node) {
    return new PythonConstraintVisitor(this, node);
  }

  public static class PythonInterestingVisitor extends AstInterestingVisitor
      implements PythonInstructionVisitor {
    public PythonInterestingVisitor(int vn) {
      super(vn);
    }

    @Override
    public void visitBinaryOp(final SSABinaryOpInstruction instruction) {
      bingo = true;
    }

    @Override
    public void visitPythonBinaryOp(PythonBinaryOpInstruction binop) {
      bingo = true;
    }

    @Override
    public void visitPythonInvoke(PythonInvokeInstruction inst) {
      bingo = true;
    }
  }

  @Override
  protected InterestingVisitor makeInterestingVisitor(CGNode node, int vn) {
    return new PythonInterestingVisitor(vn);
  }

  /**
   * The name of the global holding a defaulted parameter's materialized default: {@code
   * <entity>_defaults_<i>} for a Python default, {@code <entity>_click_defaults_<i>} for a {@code
   * @click.option} default (wala/ML#971). A synthesized constructor's trailing formal i mirrors
   * {@code __init__}'s parameter i + 1, and the default globals are written under {@code
   * __init__}'s entity name, so the lookup follows that mapping (wala/ML#762). Every other target
   * reads its own entity's global.
   *
   * @param target The callee.
   * @param i The zero-based parameter index in the callee's formals.
   * @param infix {@code "_defaults_"} or {@code "_click_defaults_"}.
   * @return The global's name, without the {@code global } prefix.
   */
  private static String defaultsGlobalName(CGNode target, int i, String infix) {
    return target.getMethod() instanceof PythonConstructorFunction
        ? target.getMethod().getDeclaringClass().getName()
            + "/"
            + INIT_METHOD_NAME
            + infix
            + (i + 1)
        : target.getMethod().getDeclaringClass().getName() + infix + i;
  }

  /**
   * Resolves a global by name to its field on {@code Root}.
   *
   * @param name The global's name, without the {@code global } prefix.
   * @return The field; never {@code null}, since globals are dynamic fields created on lookup.
   */
  private IField resolveGlobal(String name) {
    FieldReference global =
        FieldReference.findOrCreate(
            PythonTypes.Root, Atom.findOrCreateUnicodeAtom("global " + name), PythonTypes.Root);
    return getClassHierarchy().resolveField(global);
  }

  /**
   * Whether a key is a materialized {@code @click.option} default (wala/ML#971): a constant key of
   * the {@link PythonTypes#clickDefault} or {@link PythonTypes#clickDefaultString} class. Such a
   * key carries its value like any constant key, so a shape or dtype reader takes it as it takes a
   * Python default, but a comparison fold must decline it: the default is the value of the one
   * invocation that passes no option, and every other invocation the command line admits binds the
   * parameter otherwise, so a guard folded from it prunes arms the program runs.
   *
   * @param key The instance key.
   * @return {@code true} for a click default's constant key.
   */
  public static boolean isClickDefault(InstanceKey key) {
    if (!(key instanceof ConstantKey)) return false;
    TypeReference type = key.concreteType().getReference();
    return type.equals(PythonTypes.clickDefault) || type.equals(PythonTypes.clickDefaultString);
  }

  /**
   * Binds a {@code @click.option} default to its parameter under a constant key of the click
   * default's own class (wala/ML#971): each constant key in the click-defaults global becomes the
   * same value under {@link PythonTypes#clickDefaultString} for a string and {@link
   * PythonTypes#clickDefault} otherwise, so {@link #isClickDefault(InstanceKey)} tells it from an
   * ordinary constant while its value reads unchanged. A member that is not a constant key (a list
   * or tuple default) passes through as it is.
   */
  private final class ClickDefaultOperator extends UnaryOperator<PointsToSetVariable> {
    @Override
    public byte evaluate(PointsToSetVariable lhs, PointsToSetVariable rhs) {
      if (rhs.getValue() == null) return NOT_CHANGED;
      MutableIntSet out = IntSetUtil.make();
      rhs.getValue()
          .foreach(
              i -> {
                InstanceKey key = getSystem().getInstanceKey(i);
                if (!(key instanceof ConstantKey) || isClickDefault(key)) {
                  out.add(i);
                  return;
                }
                Object value = ((ConstantKey<?>) key).getValue();
                TypeReference type =
                    value instanceof String
                        ? PythonTypes.clickDefaultString
                        : PythonTypes.clickDefault;
                out.add(
                    getSystem()
                        .findOrCreateIndexForInstanceKey(getInstanceKeyForConstant(type, value)));
              });
      return lhs.addAll(out) ? CHANGED : NOT_CHANGED;
    }

    @Override
    public int hashCode() {
      return ClickDefaultOperator.class.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof ClickDefaultOperator;
    }

    @Override
    public String toString() {
      return "click default";
    }
  }

  /**
   * Whether an instance key is the None constant. {@code None} is one global {@link ConstantKey}
   * whose concrete type is {@code Root}, so the core builder's null-receiver filter, which asks the
   * language whether the key's <em>type</em> is the null type, does not recognize it.
   *
   * @param key The instance key.
   * @return {@code true} iff the key is the None constant.
   */
  public static boolean isNoneConstant(InstanceKey key) {
    return key instanceof ConstantKey && ((ConstantKey<?>) key).getValue() == null;
  }

  /**
   * The pointer analysis makes no field key for the None constant (wala/ML#964): an attribute write
   * on {@code None} raises at run time and so does a read, so neither carries a value. The core put
   * and get operators call this method and skip a {@code null} key. Without this, a write whose
   * receiver set merely includes {@code None} (a function object local also bound to {@code None},
   * a defaulted parameter) lands its value on the one global {@code None} key, and every wildcard
   * element read over a container that may be {@code None} (a {@code zip} or {@code for} over a
   * list that is {@code None} on one arm) reads that value back as an element of an unrelated
   * container. The heap model, which ModRef and other consumers read, still resolves the key
   * through the factory, so no {@code null} leaves the analysis.
   *
   * @param I The instance key.
   * @param field The field.
   * @return The pointer key, or {@code null} for the None constant.
   */
  @Override
  public PointerKey getPointerKeyForInstanceField(InstanceKey I, IField field) {
    if (isNoneConstant(I)) return null;
    return super.getPointerKeyForInstanceField(I, field);
  }

  /**
   * A mapping of script names to wildcard imports included in the script.
   *
   * @return A mapping of script names to wildcard imports included in the corresponding script.
   */
  protected Map<String, Deque<MethodReference>> getScriptToWildcardImports() {
    return scriptToWildcardImports;
  }
}
