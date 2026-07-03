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
import static com.ibm.wala.cast.python.util.Util.IMPORT_WILDCARD_CHARACTER;
import static com.ibm.wala.cast.python.util.Util.MODULE_INITIALIZATION_FILENAME;
import static com.ibm.wala.cast.python.util.Util.PYTHON_FILE_EXTENSION;

import com.google.common.collect.Maps;
import com.ibm.wala.cast.ipa.callgraph.AstSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.ipa.callgraph.GlobalObjectKey;
import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalRead;
import com.ibm.wala.cast.ir.ssa.AstLexicalWrite;
import com.ibm.wala.cast.ir.ssa.AstPropertyRead;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ipa.summaries.PythonInstanceMethodTrampoline;
import com.ibm.wala.cast.python.ir.PythonLanguage;
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
import com.ibm.wala.ipa.callgraph.propagation.FilteredPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.FilteredPointerKey.TypeFilter;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKeyFactory;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.StaticFieldKey;
import com.ibm.wala.ipa.cha.IClassHierarchy;
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
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.IntIterator;
import com.ibm.wala.util.intset.IntSetUtil;
import com.ibm.wala.util.intset.MutableIntSet;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collection;
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

  public static class PythonConstraintVisitor extends AstConstraintVisitor
      implements PythonInstructionVisitor {

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
     *     <p>We also use this mechanism for generators, which put everything in a fake field named
     *     "__contents__" which is read by this mechanism. Generator expressions use the iterator
     *     type and generator functions use CodeBody.
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
    public void visitPythonBinaryOp(PythonBinaryOpInstruction binop) {}

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

      int dflts =
          target.getMethod().getNumberOfParameters()
              - target.getMethod().getNumberOfDefaultParameters();
      for (int i = dflts; i < target.getMethod().getNumberOfParameters(); i++) {
        if (!args.contains(i)) {
          String name = target.getMethod().getDeclaringClass().getName() + "_defaults_" + i;
          FieldReference global =
              FieldReference.findOrCreate(
                  PythonTypes.Root,
                  Atom.findOrCreateUnicodeAtom("global " + name),
                  PythonTypes.Root);
          IField f = getClassHierarchy().resolveField(global);
          PointerKey lval = getPointerKeyForLocal(target, i + 1);
          getSystem().newConstraint(lval, assignOperator, new StaticFieldKey(f));
        }
      }

      // return values
      PointerKey rret = getPointerKeyForReturnValue(target);
      PointerKey lret = getPointerKeyForLocal(caller, call.getReturnValue(0));
      getSystem().newConstraint(lret, assignOperator, rret);

      PointerKey reret = getPointerKeyForExceptionalReturnValue(target);
      PointerKey leret = getPointerKeyForLocal(caller, call.getException());
      getSystem().newConstraint(leret, assignOperator, reret);
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
   * A mapping of script names to wildcard imports included in the script.
   *
   * @return A mapping of script names to wildcard imports included in the corresponding script.
   */
  protected Map<String, Deque<MethodReference>> getScriptToWildcardImports() {
    return scriptToWildcardImports;
  }
}
