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
import static com.ibm.wala.cast.python.types.PythonTypes.INIT_METHOD_NAME;
import static com.ibm.wala.cast.python.types.Util.makeGlobalRef;

import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.loader.DynamicCallSiteReference;
import com.ibm.wala.cast.python.ipa.summaries.PythonConstructorFunction;
import com.ibm.wala.cast.python.ipa.summaries.PythonInstanceMethodTrampoline;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummarizedFunction;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummary;
import com.ibm.wala.cast.python.ir.PythonLanguage;
import com.ibm.wala.cast.python.loader.IPythonClass;
import com.ibm.wala.cast.python.loader.PythonLoader;
import com.ibm.wala.cast.python.loader.PythonLoader.PythonSummaryShellClass;
import com.ibm.wala.cast.python.loader.StarFormalDeclaration;
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
import com.ibm.wala.types.annotations.Annotation;
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
                  receiver.getClassLoader().getReference(),
                  receiver.getName() + "/" + INIT_METHOD_NAME);
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
          MethodReference inheritedSummaryInit = null;

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
              for (MethodReference m : ((IPythonClass) parent).getMethodReferences()) {
                Atom pythonLevelName = instanceFieldName(m, shell);
                // An inherited summary initializer is captured independently of the override
                // dedup below: the subclass's own `__init__` overrides it as a method, but its
                // framework effects still apply through the mandatory `super().__init__()`
                // (wala/ML#683).
                if (shell
                    && inheritedSummaryInit == null
                    && pythonLevelName.toString().equals(INIT_METHOD_NAME))
                  inheritedSummaryInit = m;
                // Dedup by the Python-level method name. Shell references are all named by the
                // generic function selector, so using `m.getName()` here would let the first
                // shell method shadow every other one. See wala/ML#667.
                if (!seenMethodNames.add(pythonLevelName)) continue;
                methodReferences.add(m);
                if (shell) summaryDeclaredMethods.add(m);
              }
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
            // A method declared with `@property` is a getter: the instance's attribute of that
            // name is the getter's VALUE, `inst.name = name(inst)`, not a bound method
            // (wala/ML#993). Evaluating it here, at construction, is exact for a flow-insensitive
            // heap: the body's reads of `self` resolve to whatever the initializer ever stores.
            // Only instances built through this synthesized constructor get the value; a read on
            // the class itself, or on an instance made another way, keeps the previous behavior.
            // A property's setter or deleter (`@name.setter`) is not an attribute of the instance:
            // a write to the attribute reaches the field directly, and binding the accessor under
            // the property's name would make a call through the attribute dispatch it beside the
            // getter's value (wala/ML#993). The analysis never invokes the accessor (measured: its
            // body is absent from the call graph unless called directly); the written value
            // reaches the attribute as written.
            if (!summaryDeclaredMethods.contains(r)
                && isPropertyAccessor(r, receiver.getClassHierarchy())) continue;
            if (!summaryDeclaredMethods.contains(r)
                && isProperty(r, receiver.getClassHierarchy())) {
              // The getter is read off the class attribute of its name, where the class body
              // bound it, so its lexical reads of module names resolve through that creator (a
              // fresh allocation here would have no module scope: measured, a getter calling a
              // module function or a library API read as nothing). A setter's definition rebinds
              // that attribute to both functions, so the read is filtered to the getter's own
              // function class; reading it unfiltered ran the setter with the instance as its
              // value (measured).
              int attribute = v++;
              ctor.addStatement(
                  insts.GetInstruction(
                      pc++,
                      attribute,
                      1,
                      FieldReference.findOrCreate(
                          PythonTypes.Root, r.getName(), PythonTypes.Root)));
              int getter = v++;
              ctor.addStatement(
                  insts.CheckCastInstruction(pc++, getter, attribute, r.getDeclaringClass(), true));
              int value = v++;
              int valueException = v++;
              @SuppressWarnings({"unchecked", "rawtypes"})
              Pair<String, Integer>[] noKeywords = new Pair[0];
              ctor.addStatement(
                  new PythonInvokeInstruction(
                      pc,
                      value,
                      valueException,
                      new DynamicCallSiteReference(site.getDeclaredTarget(), pc),
                      new int[] {getter, inst},
                      noKeywords));
              pc++;
              ctor.addStatement(
                  insts.PutInstruction(
                      pc++,
                      inst,
                      value,
                      FieldReference.findOrCreate(
                          PythonTypes.Root, instanceFieldName(r, false), PythonTypes.Root)));
              continue;
            }

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

            // Add a metadata variable that refers to the class the instance is constructed from,
            // which for an inherited method is the derived class, not the declaring one. Per
            // https://docs.python.org/3/library/functions.html#classmethod, "[i]f a class method is
            // called for a derived class, the derived class object is passed as the implied first
            // argument": an inherited `from_config` whose body returns `cls(...)` constructs the
            // derived class, so the rebuilt model runs the derived class's own `call`
            // (wala/ML#997).
            int classVar = v++;
            // The class global is named by the class's own type name, as the method helper names a
            // method's declaring class by its package.
            String globalName = receiver.getReference().getName().toString().substring(1);
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
                    FieldReference.findOrCreate(
                        PythonTypes.Root,
                        instanceFieldName(r, summaryDeclaredMethods.contains(r)),
                        PythonTypes.Root)));
            pc++;
          }

          // The Keras `Model.__init__` contract requires every subclass initializer to invoke
          // `super().__init__()`, which assigns framework state such as
          // `_distribution_strategy` (wala/ML#683). The `super()` machinery cannot dispatch
          // summary-declared initializers (its synthesized `$self` never binds; wala/ML#995), so
          // the synthesized constructor invokes an inherited summary `__init__` on the new
          // instance directly. Over-approximates only for subclasses that unlawfully skip
          // `super().__init__()`.
          if (inheritedSummaryInit != null) {
            int sf = v++;
            ctor.addStatement(
                insts.NewInstruction(
                    pc, sf, NewSiteReference.make(pc, inheritedSummaryInit.getDeclaringClass())));
            pc++;

            int sres = v++;
            int sexc = v++;
            CallSiteReference sref = new DynamicCallSiteReference(site.getDeclaredTarget(), pc);
            @SuppressWarnings({"unchecked", "rawtypes"})
            Pair<String, Integer>[] noKeywords = new Pair[0];
            ctor.addStatement(
                new PythonInvokeInstruction(2, sres, sexc, sref, new int[] {sf, inst}, noKeywords));
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
                        Atom.findOrCreateUnicodeAtom(INIT_METHOD_NAME),
                        PythonTypes.Root)));
            pc++;

            int numberOfParameters = init.getNumberOfParameters();
            // The constructor's formals mirror `__init__`'s shifted by one, and its star formals,
            // declared below on the constructor function, receive what a class call packs: the
            // positional extras as a tuple and the keywords naming no formal as a dict. Forward the
            // two as a starred slot and a `**` keyword, so the unpack at `__init__` binds them as a
            // caller's `Cls(*args, **kwargs)` would; forwarded by position they were nested, or,
            // before the constructor declared them, never packed at all, so a keyword naming no
            // formal of a class never reached `__init__`'s `**kwargs` and a positional argument
            // past
            // `__init__`'s formals never reached its `*args` (wala/ML#188, wala/ML#997).
            int initVarargs =
                init instanceof StarFormalDeclaration d ? d.getVarargsParameter() : -1;
            int initKeywords =
                init instanceof StarFormalDeclaration d ? d.getKeywordsParameter() : -1;
            int positional = initKeywords >= 0 ? initKeywords : numberOfParameters;
            int[] cps = new int[positional > 1 ? positional : 2];
            cps[0] = fv;
            cps[1] = inst;
            for (int j = 2; j < positional; j++) {
              cps[j] = j;
            }
            int[] starred =
                initVarargs >= 2 && initVarargs < positional ? new int[] {initVarargs} : new int[0];
            @SuppressWarnings({"unchecked", "rawtypes"})
            Pair<String, Integer>[] keywordParams =
                initKeywords >= 0 ? new Pair[] {Pair.make("null", initKeywords)} : new Pair[0];

            int result = v++;
            int except = v++;
            CallSiteReference cref = new DynamicCallSiteReference(site.getDeclaredTarget(), pc);
            ctor.addStatement(
                new PythonInvokeInstruction(2, result, except, cref, cps, keywordParams, starred));
            pc++;

            // Declare `__init__`'s parameter names on the constructor's own formals, so a caller's
            // keyword arguments map onto them (the keyword resolver matches callee local names) and
            // flow through the positional forwarding above. Without this, keyword arguments to any
            // source-class constructor were silently dropped: `GCN(units=16)` never reached
            // `__init__`'s `units`, so `self._units`-style fields stayed empty (wala/ML#664).
            // Constructor formal `j` feeds `__init__`'s formal `j + 1` via `cps` (`__init__` has
            // the extra function-object formal in front).
            if ((ctor.getValueNames() == null || ctor.getValueNames().isEmpty())
                && init instanceof AstMethod astInit) {
              String[][] initNames = astInit.debugInfo().getSourceNamesForValues();
              Map<Integer, Atom> ctorValueNames = HashMapFactory.make();
              ctorValueNames.put(1, Atom.findOrCreateUnicodeAtom("self"));
              for (int j = 2; j < numberOfParameters; j++) {
                int initVn = j + 1;
                if (initNames != null
                    && initVn < initNames.length
                    && initNames[initVn] != null
                    && initNames[initVn].length > 0) {
                  ctorValueNames.put(j, Atom.findOrCreateUnicodeAtom(initNames[initVn][0]));
                }
              }
              ctor.setValueNames(ctorValueNames);
            }
          }

          ctor.addStatement(insts.ReturnInstruction(pc++, inst, false));

          if (ctor.getValueNames() == null || ctor.getValueNames().isEmpty()) {
            ctor.setValueNames(Collections.singletonMap(1, Atom.findOrCreateUnicodeAtom("self")));
          }

          ctors.put(
              receiver,
              new PythonConstructorFunction(
                  ref,
                  ctor,
                  receiver,
                  init == null ? 0 : init.getNumberOfDefaultParameters(),
                  init instanceof StarFormalDeclaration
                      ? ((StarFormalDeclaration) init).getNumberOfTrailingNonDefaultableParameters()
                      : 0,
                  // Constructor argument j is `__init__` argument j + 1 (`__init__` has `self` in
                  // front), so its star formals are `__init__`'s shifted by one.
                  init instanceof StarFormalDeclaration d && d.getVarargsParameter() >= 0
                      ? d.getVarargsParameter() - 1
                      : -1,
                  init instanceof StarFormalDeclaration d && d.getKeywordsParameter() >= 0
                      ? d.getKeywordsParameter() - 1
                      : -1));
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

  /**
   * Whether the given method is declared with {@code @property} (wala/ML#993): its class carries
   * the annotation of that name, as a static method carries {@link PythonTypes#STATIC_METHOD}.
   *
   * @param r The method.
   * @param cha The class hierarchy that resolves the method's class.
   * @return {@code true} iff the method is a property getter.
   */
  private static boolean isProperty(MethodReference r, IClassHierarchy cha) {
    IClass cls = cha.lookupClass(r.getDeclaringClass());
    return cls != null
        && cls.getAnnotations() != null
        && cls.getAnnotations().contains(Annotation.make(PythonTypes.PROPERTY));
  }

  /**
   * Whether the given method is a property's setter or deleter (wala/ML#993): its class carries an
   * annotation named {@code <property>.setter} or {@code <property>.deleter}.
   *
   * @param r The method.
   * @param cha The class hierarchy that resolves the method's class.
   * @return {@code true} iff the method is a property accessor other than the getter.
   */
  private static boolean isPropertyAccessor(MethodReference r, IClassHierarchy cha) {
    IClass cls = cha.lookupClass(r.getDeclaringClass());
    if (cls == null || cls.getAnnotations() == null) return false;
    for (Annotation annotation : cls.getAnnotations()) {
      String name = annotation.getType().getName().toString();
      if (name.endsWith(".setter") || name.endsWith(".deleter")) return true;
    }
    return false;
  }

  /**
   * Returns the instance-field name under which the trampoline for the given method reference is
   * stored on the constructed object. A source-class method reference is named by its Python
   * method, so its own name is used directly. A summary-shell method reference is named by the
   * generic function selector (wala/ML#106), so the Python method name is instead the last
   * component of its declaring class's type name (e.g. {@code add_weight} from {@code
   * Ltensorflow/keras/layers/Layer/add_weight}). Without this distinction every shell-inherited
   * method wired under the selector-name field: the first shadowed the rest in the per-name dedup,
   * and none was reachable under its Python name. See wala/ML#667.
   *
   * @param r The method reference being wired.
   * @param shellDeclared Whether the reference was declared by a {@link PythonSummaryShellClass}.
   * @return The atom to use as the instance-field name.
   */
  private static Atom instanceFieldName(MethodReference r, boolean shellDeclared) {
    if (!shellDeclared) return r.getName();
    String typeName = r.getDeclaringClass().getName().toString();
    return Atom.findOrCreateUnicodeAtom(typeName.substring(typeName.lastIndexOf('/') + 1));
  }
}
