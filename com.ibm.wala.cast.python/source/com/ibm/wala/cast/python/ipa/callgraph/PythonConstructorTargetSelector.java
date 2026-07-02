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

import static com.ibm.wala.cast.python.types.PythonTypes.DO_METHOD_NAME;
import static com.ibm.wala.cast.python.types.Util.getGlobalName;
import static com.ibm.wala.cast.python.types.Util.makeGlobalRef;

import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.loader.DynamicCallSiteReference;
import com.ibm.wala.cast.python.ipa.summaries.PythonInstanceMethodTrampoline;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummarizedFunction;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummary;
import com.ibm.wala.cast.python.ir.PythonLanguage;
import com.ibm.wala.cast.python.loader.IPythonClass;
import com.ibm.wala.cast.python.loader.PythonLoader;
import com.ibm.wala.cast.python.loader.PythonLoader.PythonSummaryShellClass;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.MethodTargetSelector;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ipa.summaries.MethodSummary;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAInstructionFactory;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.Pair;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

public class PythonConstructorTargetSelector implements MethodTargetSelector {

  private static final Logger LOGGER =
      Logger.getLogger(PythonConstructorTargetSelector.class.getName());

  private final Map<IClass, IMethod> ctors = HashMapFactory.make();

  private final MethodTargetSelector base;

  public PythonConstructorTargetSelector(MethodTargetSelector base) {
    this.base = base;
  }

  @Override
  public IMethod getCalleeTarget(CGNode caller, CallSiteReference site, IClass receiver) {
    if (receiver != null) {
      LOGGER.finer("Getting callee target for receiver: " + receiver);
      LOGGER.finer("Calling method name is: " + caller.getMethod().getName());

      IClassHierarchy cha = receiver.getClassHierarchy();
      if (cha.isSubclassOf(receiver, cha.lookupClass(PythonTypes.object))
          && (receiver instanceof IPythonClass)) {
        if (!ctors.containsKey(receiver)) {
          TypeReference ctorRef =
              TypeReference.findOrCreate(
                  receiver.getClassLoader().getReference(), receiver.getName() + "/__init__");
          IClass ctorCls = cha.lookupClass(ctorRef);
          IMethod init = ctorCls == null ? null : ctorCls.getMethod(AstMethodReference.fnSelector);
          /*
           * https://github.com/wala/ML/issues/579: a NamedTuple with no explicit __init__ maps
           * positional constructor arguments to its declared fields in declaration order. Collect
           * those fields so the synthesized constructor can populate them on the new instance below.
           */
          List<IField> tupleFields =
              init == null && isPositionalFieldClass(receiver)
                  ? new ArrayList<>(receiver.getDeclaredStaticFields())
                  : Collections.emptyList();
          int params =
              init != null
                  ? init.getNumberOfParameters()
                  : (tupleFields.isEmpty() ? 1 : 1 + tupleFields.size());
          int v = params + 2;
          int pc = 0;
          int inst = v++;
          MethodReference ref =
              MethodReference.findOrCreate(
                  receiver.getReference(), site.getDeclaredTarget().getSelector());
          PythonSummary ctor = new PythonSummary(ref, params);
          SSAInstructionFactory insts = PythonLanguage.Python.instructionFactory();

          // Copy metadata from the original do() method if it exists.
          // This is useful for summarized methods like Dense.do() that carry extra parameters.
          MethodReference originalDoRef =
              MethodReference.findOrCreate(receiver.getReference(), DO_METHOD_NAME, "()LRoot;");
          IClass ctorContainer = cha.lookupClass(originalDoRef.getDeclaringClass());
          IMethod originalDo =
              ctorContainer == null ? null : ctorContainer.getMethod(originalDoRef.getSelector());
          if (originalDo instanceof PythonSummarizedFunction) {
            MethodSummary originalSummary = null;
            try {
              java.lang.reflect.Method getSummary = originalDo.getClass().getMethod("getSummary");
              originalSummary = (MethodSummary) getSummary.invoke(originalDo);
            } catch (Exception e) {
              try {
                java.lang.reflect.Field f =
                    com.ibm.wala.ipa.summaries.SummarizedMethod.class.getDeclaredField("summary");
                f.setAccessible(true);
                originalSummary = (MethodSummary) f.get(originalDo);
              } catch (Exception e2) {
              }
            }

            if (originalSummary != null) {
              // Copy statements, but map parameter uses.
              // Parameters in original do() are 1, 2, ...
              // Parameters in our new ctor are also 1, 2, ...
              // However, we need to ensure the number of parameters matches.
              for (SSAInstruction instOrig : originalSummary.getStatements()) {
                if (instOrig != null
                    && !(instOrig instanceof SSANewInstruction)
                    && !(instOrig instanceof SSAReturnInstruction)) {
                  ctor.addStatement(instOrig);
                  pc++;
                }
              }
              if (originalSummary.getValueNames() != null) {
                ctor.setValueNames(originalSummary.getValueNames());
              }
            }
          }

          ctor.addStatement(
              insts.NewInstruction(pc, inst, NewSiteReference.make(pc, PythonTypes.object)));
          pc++;

          /*
           * https://github.com/wala/ML/issues/579: store each positional constructor argument into
           * the corresponding declared field, in declaration order. Param 1 is the callable; the
           * positional arguments are value numbers 2, 3, ... So field[i] is populated from argument
           * value number 2 + i.
           */
          for (int i = 0; i < tupleFields.size(); i++) {
            ctor.addStatement(
                insts.PutInstruction(
                    pc++,
                    inst,
                    2 + i,
                    FieldReference.findOrCreate(
                        PythonTypes.Root, tupleFields.get(i).getName(), PythonTypes.Root)));
          }

          Collection<TypeReference> innerReferences = Collections.emptyList();
          Collection<MethodReference> methodReferences = new ArrayList<>();

          // Methods inherited from a summary class shell (wala/ML#118) have no function object on
          // the source class object to read; their function classes are bypass-registered and are
          // allocated directly below, mirroring the engine's summary-constructor rewriting.
          Set<MethodReference> summaryDeclaredMethods = new HashSet<>();

          if (receiver instanceof IPythonClass) {
            IPythonClass x = (IPythonClass) receiver;
            innerReferences = x.getInnerReferences();
            // Collect own methods first; they take precedence over inherited methods of the same
            // name (Python override semantics).
            Set<Atom> seenMethodNames = new HashSet<>();
            for (MethodReference m : x.getMethodReferences()) {
              methodReferences.add(m);
              seenMethodNames.add(m.getName());
            }
            // Also stamp methods inherited from supertypes onto the instance, so a call like
            // `c.func(...)` where `class C(D)` and `D` declares `func` resolves through the
            // constructor's per-method trampoline instead of silently dropping the edge. Walks
            // the single supertype chain via `getSuperclass()`; the `seenMethodNames` set keeps
            // own methods winning on name collision, and the IClass model collapses Python
            // multi-inheritance to one `superName` per `PythonClass`, so a true left-first MRO
            // walk isn't representable here without extending the loader. See
            // https://github.com/wala/ML/issues/107.
            for (IClass parent = receiver.getSuperclass();
                parent instanceof IPythonClass;
                parent = parent.getSuperclass()) {
              boolean shell = parent instanceof PythonSummaryShellClass;
              ((IPythonClass) parent)
                  .getMethodReferences().stream()
                      .filter(m -> seenMethodNames.add(m.getName()))
                      .forEach(
                          m -> {
                            methodReferences.add(m);
                            if (shell) summaryDeclaredMethods.add(m);
                          });
            }
          } else {
            for (IMethod m : receiver.getDeclaredMethods()) {
              if (!m.isInit() && !m.isClinit()) {
                methodReferences.add(m.getReference());
              }
            }
          }

          for (TypeReference r : innerReferences) {
            int orig_t = v++;
            String typeName = r.getName().toString();
            typeName = typeName.substring(typeName.lastIndexOf('/') + 1);
            FieldReference inner =
                FieldReference.findOrCreate(
                    PythonTypes.Root, Atom.findOrCreateUnicodeAtom(typeName), PythonTypes.Root);

            ctor.addStatement(insts.GetInstruction(pc, orig_t, 1, inner));
            pc++;

            ctor.addStatement(insts.PutInstruction(pc, inst, orig_t, inner));
            pc++;
          }

          for (MethodReference r : methodReferences) {
            int f = v++;
            ctor.addStatement(
                insts.NewInstruction(
                    pc,
                    f,
                    NewSiteReference.make(
                        pc,
                        PythonInstanceMethodTrampoline.findOrCreate(
                            r.getDeclaringClass(), receiver.getClassHierarchy()))));
            pc++;

            ctor.addStatement(
                insts.PutInstruction(
                    pc,
                    f,
                    inst,
                    FieldReference.findOrCreate(
                        PythonTypes.Root,
                        Atom.findOrCreateUnicodeAtom("$self"),
                        PythonTypes.Root)));
            pc++;

            int orig_f = v++;
            if (summaryDeclaredMethods.contains(r)) {
              // The function object of a shell-inherited summary method is not a field of the
              // source class object; allocate its bypass-registered function class directly.
              ctor.addStatement(
                  insts.NewInstruction(
                      pc, orig_f, NewSiteReference.make(pc, r.getDeclaringClass())));
            } else {
              ctor.addStatement(
                  insts.GetInstruction(
                      pc,
                      orig_f,
                      1,
                      FieldReference.findOrCreate(
                          PythonTypes.Root, r.getName(), PythonTypes.Root)));
            }
            pc++;

            ctor.addStatement(
                insts.PutInstruction(
                    pc,
                    f,
                    orig_f,
                    FieldReference.findOrCreate(
                        PythonTypes.Root,
                        Atom.findOrCreateUnicodeAtom("$function"),
                        PythonTypes.Root)));
            pc++;

            // Add a metadata variable that refers to the declaring class.
            // NOTE: Per https://docs.python.org/3/library/functions.html#classmethod, "[i]f a class
            // method is called for a derived class, the derived class object is passed as the
            // implied first argument."
            int classVar = v++;
            String globalName = getGlobalName(r);
            FieldReference globalRef = makeGlobalRef(receiver.getClassLoader(), globalName);

            ctor.addStatement(new AstGlobalRead(pc++, classVar, globalRef));

            ctor.addStatement(
                insts.PutInstruction(
                    pc++,
                    f,
                    classVar,
                    FieldReference.findOrCreate(
                        PythonTypes.Root,
                        Atom.findOrCreateUnicodeAtom("$class"),
                        PythonTypes.Root)));

            ctor.addStatement(
                insts.PutInstruction(
                    pc,
                    inst,
                    f,
                    FieldReference.findOrCreate(PythonTypes.Root, r.getName(), PythonTypes.Root)));
            pc++;
          }

          if (init != null) {
            int fv = v++;
            ctor.addStatement(
                insts.GetInstruction(
                    pc,
                    fv,
                    1,
                    FieldReference.findOrCreate(
                        PythonTypes.Root,
                        Atom.findOrCreateUnicodeAtom("__init__"),
                        PythonTypes.Root)));
            pc++;

            int numberOfParameters = init.getNumberOfParameters();
            int[] cps = new int[numberOfParameters > 1 ? numberOfParameters : 2];
            cps[0] = fv;
            cps[1] = inst;
            for (int j = 2; j < numberOfParameters; j++) {
              cps[j] = j;
            }

            int result = v++;
            int except = v++;
            CallSiteReference cref = new DynamicCallSiteReference(site.getDeclaredTarget(), pc);
            @SuppressWarnings({"unchecked", "rawtypes"})
            Pair<String, Integer>[] keywordParams = new Pair[0];
            ctor.addStatement(
                new PythonInvokeInstruction(2, result, except, cref, cps, keywordParams));
            pc++;
          }

          ctor.addStatement(insts.ReturnInstruction(pc++, inst, false));

          if (ctor.getValueNames() == null || ctor.getValueNames().isEmpty()) {
            ctor.setValueNames(Collections.singletonMap(1, Atom.findOrCreateUnicodeAtom("self")));
          }

          ctors.put(receiver, new PythonSummarizedFunction(ref, ctor, receiver));
        }

        return ctors.get(receiver);
      }
    }
    return base.getCalleeTarget(caller, site, receiver);
  }

  /**
   * Whether {@code receiver} is a class whose constructor maps positional arguments to declared
   * fields in declaration order &mdash; i.e. a {@code typing.NamedTuple} subclass. Such a class
   * carries no explicit {@code __init__}; its fields come from PEP-526 annotations and are
   * populated positionally (wala/ML#579). Detected via the unresolved {@code NamedTuple} supertype
   * the loader records.
   *
   * @param receiver The class being constructed.
   * @return {@code true} iff positional field population applies.
   */
  private static boolean isPositionalFieldClass(IClass receiver) {
    if (!(receiver instanceof PythonLoader.PythonClass)) return false;
    return ((PythonLoader.PythonClass) receiver)
        .getMissingTypeNames().stream()
            .anyMatch(
                n ->
                    n.equals("NamedTuple")
                        || n.endsWith(".NamedTuple")
                        || n.endsWith("/NamedTuple"));
  }
}
