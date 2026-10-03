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
import static com.ibm.wala.cast.python.types.PythonTypes.KERAS_BUILD_METHOD_NAME;
import static com.ibm.wala.cast.python.types.PythonTypes.STATIC_METHOD;
import static com.ibm.wala.cast.python.types.Util.getDeclaringClassTypeReference;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.isClassMethod;
import static com.ibm.wala.types.annotations.Annotation.make;

import com.ibm.wala.cast.loader.DynamicCallSiteReference;
import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.ipa.summaries.PythonInstanceMethodTrampoline;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummary;
import com.ibm.wala.cast.python.ir.PythonLanguage;
import com.ibm.wala.cast.python.loader.IPythonClass;
import com.ibm.wala.cast.python.loader.PythonLoader.PythonClass;
import com.ibm.wala.cast.python.loader.PythonLoader.PythonSummaryShellClass;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.classLoader.SyntheticClass;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.MethodTargetSelector;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ipa.summaries.BypassSyntheticClass;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.types.ClassLoaderReference;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.Selector;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.Pair;
import java.util.Map;
import java.util.logging.Logger;

public class PythonInstanceMethodTrampolineTargetSelector<T>
    extends PythonMethodTrampolineTargetSelector<T> {

  private static final Logger LOGGER =
      Logger.getLogger(PythonInstanceMethodTrampolineTargetSelector.class.getName());

  private PythonAnalysisEngine<T> engine;

  public PythonInstanceMethodTrampolineTargetSelector(
      MethodTargetSelector base, PythonAnalysisEngine<T> engine) {
    super(base);
    this.engine = engine;
  }

  @Override
  protected boolean shouldProcess(CGNode caller, CallSiteReference site, IClass receiver) {
    IClassHierarchy cha = receiver.getClassHierarchy();
    return cha.isSubclassOf(receiver, cha.lookupClass(PythonTypes.trampoline))
        || this.isCallable(receiver);
  }

  @Override
  public IMethod getCalleeTarget(CGNode caller, CallSiteReference site, IClass receiver) {
    // TODO: Callable detection may need to be moved. See https://github.com/wala/ML/issues/207. If
    // it stays here, we should further document the receiver swapping process.
    if (isCallable(receiver)) {
      LOGGER.finer("Encountered callable.");

      PythonInvokeInstruction call = this.getCall(caller, site);
      if (call == null) return super.getCalleeTarget(caller, site, receiver);

      // The callable is resolved for the one instance being dispatched (wala/ML#1012). WALA asks
      // for a target once per receiver instance arriving at the site, so each instance dispatches
      // to its own class's callable and a site several classes reach gets each class's target.
      // Reading the callee's points-to set instead made the target depend on how much of the set
      // the solver had computed when the site was first resolved, and an edge added on a partial
      // set is never retracted. The class a program instance belongs to is not its concrete type
      // (its allocation is a plain object) but the class whose constructor allocated it, which
      // the instance key carries and the receiver class does not.
      InstanceKey instance = this.getEngine().getCachedCallGraphBuilder().getDispatchReceiver();
      IClass callable =
          instance == null ? null : callableOf(receiver.getClassHierarchy(), instance);
      if (callable == null) return null; // not found.
      receiver = callable;
      LOGGER.finer("Substituting the receiver with one derived from a callable.");
    }

    return super.getCalleeTarget(caller, site, receiver);
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  @Override
  protected void populate(
      PythonSummary x, int v, IClass receiver, PythonInvokeInstruction call, Logger logger) {
    Map<Integer, Atom> names = HashMapFactory.make();
    IClass filter = ((PythonInstanceMethodTrampoline) receiver).getRealClass();

    x.addStatement(
        PythonLanguage.Python.instructionFactory()
            .GetInstruction(
                0,
                v,
                1,
                FieldReference.findOrCreate(
                    PythonTypes.Root,
                    Atom.findOrCreateUnicodeAtom("$function"),
                    PythonTypes.Root)));

    int v0 = v + 1;

    x.addStatement(
        PythonLanguage.Python.instructionFactory()
            .CheckCastInstruction(1, v0, v, filter.getReference(), true));

    int v1;

    // Are we calling a static method?
    boolean staticMethodReceiver = filter.getAnnotations().contains(make(STATIC_METHOD));
    logger.fine(
        staticMethodReceiver
            ? "Found static method receiver: " + filter
            : "Method is not static: " + filter);

    // Are we calling a class method? If so, it would be using an object instance instead of a
    // class on the LHS.
    boolean classMethodReceiver = isClassMethod(receiver);

    // only add self if the receiver isn't static or a class method.
    if (!staticMethodReceiver && !classMethodReceiver) {
      v1 = v + 2;

      x.addStatement(
          PythonLanguage.Python.instructionFactory()
              .GetInstruction(
                  1,
                  v1,
                  1,
                  FieldReference.findOrCreate(
                      PythonTypes.Root, Atom.findOrCreateUnicodeAtom("$self"), PythonTypes.Root)));

      // The Keras layer-call protocol builds lazily: `Layer.__call__` invokes `self.build(...)`
      // before the first `call`, and user subclasses commonly create their sublayers there (e.g.
      // `self._kernel = tf.keras.layers.Dense(...)`). Nothing else in the analysis invokes
      // `build`, so without this the sublayers those bodies create have empty points-to sets and
      // every value flowing through them unravels (wala/ML#595). Emitting the invocation
      // unconditionally is safe: a class without a `build` method yields an empty points-to set
      // for the field read, and the invoke has no targets.
      if (filter
          .getReference()
          .getName()
          .toString()
          .endsWith("/" + CALLABLE_METHOD_NAME_FOR_KERAS_MODELS)) {
        int buildFunction = v1 + 3;
        x.addStatement(
            PythonLanguage.Python.instructionFactory()
                .GetInstruction(
                    2,
                    buildFunction,
                    v1,
                    FieldReference.findOrCreate(
                        PythonTypes.Root,
                        Atom.findOrCreateUnicodeAtom(KERAS_BUILD_METHOD_NAME),
                        PythonTypes.Root)));
        x.addStatement(
            new PythonInvokeInstruction(
                3,
                v1 + 4,
                v1 + 5,
                new DynamicCallSiteReference(call.getCallSite().getDeclaredTarget(), 3),
                new int[] {buildFunction, v1},
                new Pair[0]));
      }
    } else if (classMethodReceiver) {
      // Add a class reference.
      v1 = v + 2;

      x.addStatement(
          PythonLanguage.Python.instructionFactory()
              .GetInstruction(
                  1,
                  v1,
                  1,
                  FieldReference.findOrCreate(
                      PythonTypes.Root, Atom.findOrCreateUnicodeAtom("$class"), PythonTypes.Root)));

      int v2 = v + 3;
      TypeReference reference = getDeclaringClassTypeReference(filter.getReference());

      x.addStatement(
          PythonLanguage.Python.instructionFactory()
              .CheckCastInstruction(1, v2, v1++, reference, true));
    } else v1 = v + 1;

    int i = 0;
    int paramSize =
        Math.max(
            staticMethodReceiver ? 1 : 2,
            call.getNumberOfPositionalParameters() + (staticMethodReceiver ? 0 : 1));
    int[] params = new int[paramSize];
    params[i++] = v0;

    if (!staticMethodReceiver) params[i++] = v1;

    for (int j = 1; j < call.getNumberOfPositionalParameters(); j++) params[i++] = j + 1;

    int ki = 0, ji = call.getNumberOfPositionalParameters() + 1;
    @SuppressWarnings({"unchecked", "rawtypes"})
    Pair<String, Integer>[] keys = new Pair[0];

    if (call.getKeywords() != null) {
      @SuppressWarnings({"unchecked", "rawtypes"})
      Pair<String, Integer>[] tmp = (Pair<String, Integer>[]) new Pair[call.getKeywords().size()];
      keys = tmp;

      for (String k : call.getKeywords()) {
        names.put(ji, Atom.findOrCreateUnicodeAtom(k));
        keys[ki++] = Pair.<String, Integer>make(k, ji++);
      }
    }

    int result = v1 + 1;
    int except = v1 + 2;

    CallSiteReference ref = new DynamicCallSiteReference(call.getCallSite().getDeclaredTarget(), 2);

    x.addStatement(
        new PythonInvokeInstruction(
            2,
            result,
            except,
            ref,
            params,
            keys,
            shiftedStarredPositions(call, staticMethodReceiver ? 0 : 1)));
    x.addStatement(new SSAReturnInstruction(3, result, false));
    x.setValueNames(names);
  }

  /**
   * The {@code __call__} method class a program-defined base class of the given class declares,
   * nearest first along the superclass chain (wala/ML#994). The walk stops at the first class the
   * program does not define, so a library summary's callable never outranks a program class's own
   * {@code call}.
   *
   * @param cha The class hierarchy.
   * @param loader The program's class loader.
   * @param type The instance's concrete class.
   * @return The inherited {@code __call__} method class, or {@code null} when no program-defined
   *     base declares one.
   */
  private static IClass inheritedDunderCall(
      IClassHierarchy cha, ClassLoaderReference loader, IClass type) {
    for (IClass base = type.getSuperclass();
        base instanceof PythonClass && !(base instanceof PythonSummaryShellClass);
        base = base.getSuperclass()) {
      IClass callable =
          cha.lookupClass(
              TypeReference.findOrCreateClass(
                  loader, "$" + base.getName().toString().substring(1), CALLABLE_METHOD_NAME));
      if (callable != null) return callable;
    }
    return null;
  }

  /**
   * The callable an instance is called through: its class's {@code __call__}, one a program-defined
   * base declares (wala/ML#994), the Keras {@code call} convention, or {@code do}, in Python's
   * lookup order. The class is the instance's concrete type when that names a callable, and
   * otherwise the class whose method allocated the instance, since a program class's instance is
   * allocated in its synthesized constructor.
   *
   * @param cha The class hierarchy.
   * @param o The instance being called.
   * @return The callable's class, or {@code null} when the instance's class names none.
   */
  static IClass callableOf(IClassHierarchy cha, InstanceKey o) {
    AllocationSiteInNode instanceKey = getAllocationSiteInNode(o);
    if (instanceKey != null) {
      CGNode node = instanceKey.getNode();
      IMethod method = node.getMethod();
      IClass declaringClass = method.getDeclaringClass();
      final ClassLoaderReference classLoaderReference =
          declaringClass.getClassLoader().getReference();

      // First, check the concrete type of the allocated object
      IClass concreteType = o.concreteType();
      if (concreteType != null) {
        String concreteTypeName = "$" + concreteType.getName().toString().substring(1);
        IClass concreteCallable =
            cha.lookupClass(
                TypeReference.findOrCreateClass(
                    classLoaderReference, concreteTypeName, CALLABLE_METHOD_NAME));
        // Python finds `__call__` along the method resolution order before a Keras layer's
        // `call` is ever consulted (`Layer.__call__` is what invokes it), so a `__call__` a
        // program-defined base class declares shadows the subclass's own `call` (wala/ML#994).
        if (concreteCallable == null)
          concreteCallable = inheritedDunderCall(cha, classLoaderReference, concreteType);
        if (concreteCallable == null) {
          concreteCallable =
              cha.lookupClass(
                  TypeReference.findOrCreateClass(
                      classLoaderReference,
                      concreteTypeName,
                      CALLABLE_METHOD_NAME_FOR_KERAS_MODELS));
        }
        if (concreteCallable == null) {
          concreteCallable =
              cha.lookupClass(
                  TypeReference.findOrCreateClass(
                      classLoaderReference, concreteTypeName, DO_METHOD_NAME));
        }
        if (concreteCallable != null) return concreteCallable;
      }

      TypeName declaringClassName = declaringClass.getName();
      final String packageName = "$" + declaringClassName.toString().substring(1);

      IClass callable =
          cha.lookupClass(
              TypeReference.findOrCreateClass(
                  classLoaderReference, packageName, CALLABLE_METHOD_NAME));

      if (callable == null) {
        callable =
            cha.lookupClass(
                TypeReference.findOrCreateClass(
                    classLoaderReference,
                    declaringClassName.toString().substring(1),
                    CALLABLE_METHOD_NAME));
      }

      // A `__call__` a program-defined base class declares comes before the class's own `do`
      // and the Keras `call` convention, as in Python's lookup (wala/ML#994). A program class's
      // instance is allocated in its synthesized constructor, so the class reached here is the
      // instance's own.
      if (callable == null)
        callable = inheritedDunderCall(cha, classLoaderReference, declaringClass);

      if (callable == null) {
        callable =
            cha.lookupClass(
                TypeReference.findOrCreateClass(classLoaderReference, packageName, DO_METHOD_NAME));

        if (callable == null) {
          callable =
              cha.lookupClass(
                  TypeReference.findOrCreateClass(
                      classLoaderReference,
                      declaringClassName.toString().substring(1),
                      DO_METHOD_NAME));
        }
      }

      // The Keras `call` convention (https://github.com/wala/ML/issues/106) applies to ANY
      // class with a `call` method, without checking its hierarchy. Although a subclass's
      // summary-modeled base has been resolvable since the class shells of
      // https://github.com/wala/ML/issues/118, gating this on a shell ancestor drops sound
      // dispatch for every subclass whose base does NOT resolve (cross-module imports,
      // https://github.com/wala/ML/issues/571; bare-name collisions,
      // https://github.com/wala/ML/issues/657; unmodeled spellings) and empirically loses 18
      // tests' worth of forward-pass coverage. Tightening is tracked by
      // https://github.com/wala/ML/issues/663.
      if (callable == null) {
        LOGGER.finer(
            "Attempting the Keras `call` convention for"
                + " https://github.com/wala/ML/issues/106.");

        callable =
            cha.lookupClass(
                TypeReference.findOrCreateClass(
                    classLoaderReference, packageName, CALLABLE_METHOD_NAME_FOR_KERAS_MODELS));

        if (callable == null) {
          callable =
              cha.lookupClass(
                  TypeReference.findOrCreateClass(
                      classLoaderReference,
                      declaringClassName.toString().substring(1),
                      CALLABLE_METHOD_NAME_FOR_KERAS_MODELS));
        }

        if (callable != null)
          LOGGER.info(
              "Applying the Keras `call` convention for"
                  + " https://github.com/wala/ML/issues/106.");
      }

      return callable;
    }
    return null;
  }

  public PythonAnalysisEngine<T> getEngine() {
    return engine;
  }

  @Override
  protected Logger getLogger() {
    return LOGGER;
  }

  /**
   * Returns true iff the given {@link IClass} represents a Python callable object.
   *
   * @param receiver The {@link IClass} in question.
   * @return True iff the given {@link IClass} represents a Python callable object.
   */
  private boolean isCallable(IClass receiver) {
    if (receiver == null) return false;
    if (receiver.getReference().equals(PythonTypes.object)
        || receiver instanceof BypassSyntheticClass) {
      return true;
    }
    if (receiver instanceof SyntheticClass
        && (receiver.getMethod(
                    new Selector(
                        Atom.findOrCreateUnicodeAtom(CALLABLE_METHOD_NAME),
                        AstMethodReference.fnDesc))
                != null
            || receiver.getMethod(
                    new Selector(
                        Atom.findOrCreateUnicodeAtom(CALLABLE_METHOD_NAME_FOR_KERAS_MODELS),
                        AstMethodReference.fnDesc))
                != null
            || receiver.getMethod(
                    new Selector(
                        Atom.findOrCreateUnicodeAtom(DO_METHOD_NAME), AstMethodReference.fnDesc))
                != null)) {
      return true;
    }

    // A summary-modeled class whose method references include a `__call__`/`call` function class
    // is callable, whether materialized as an engine-registered synthetic class (for allocatable
    // summary types) or as a summary class shell (`PythonLoader.defineSummaryClassShell`,
    // wala/ML#106). Source classes are deliberately excluded: their instances dispatch through
    // constructor-wired trampolines, and substituting the receiver here instead drops those calls.
    if ((receiver instanceof SyntheticClass || receiver instanceof PythonSummaryShellClass)
        && receiver instanceof IPythonClass) {
      for (MethodReference mr : ((IPythonClass) receiver).getMethodReferences()) {
        String clsName = mr.getDeclaringClass().getName().toString();
        if (clsName.endsWith("/" + CALLABLE_METHOD_NAME)
            || clsName.endsWith("/" + CALLABLE_METHOD_NAME_FOR_KERAS_MODELS)) {
          return true;
        }
      }
    }
    return false;
  }
}
