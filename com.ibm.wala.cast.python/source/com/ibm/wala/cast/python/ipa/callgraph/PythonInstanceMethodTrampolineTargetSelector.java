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
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
    return inheritedMethod(cha, loader, type, CALLABLE_METHOD_NAME);
  }

  /**
   * The method class of the given name the nearest program-defined base class of the given class
   * declares, in Python's method resolution order over every declared base. The walk stops at the
   * first class in that order the program does not define, so a library summary's method, which
   * Python would find there first, is never outranked by a later program base's.
   *
   * @param cha The class hierarchy.
   * @param loader The program's class loader.
   * @param type The instance's class.
   * @param name The method's name.
   * @return The inherited method class, or {@code null} when no program-defined base before the
   *     first library class declares one.
   */
  private static IClass inheritedMethod(
      IClassHierarchy cha, ClassLoaderReference loader, IClass type, String name) {
    List<IClass> order = methodResolutionOrder(cha, type, new HashSet<>());
    for (IClass base : order.subList(1, order.size())) {
      if (!isProgramClass(base)) return null;
      IClass method =
          cha.lookupClass(
              TypeReference.findOrCreateClass(
                  loader, "$" + base.getName().toString().substring(1), name));
      if (method != null) return method;
    }
    return null;
  }

  private static boolean isProgramClass(IClass c) {
    return c instanceof PythonClass && !(c instanceof PythonSummaryShellClass);
  }

  /**
   * The C3 linearization of a class over its declared bases, the class first. A base name that
   * resolves to no class is left out, as the superclass is the first base that resolves; a class
   * the program does not define ends its branch, since only its own position matters to the walk.
   * An inconsistent hierarchy, or a cycle, falls back to a left-to-right depth-first order without
   * repeats.
   *
   * @param cha The class hierarchy.
   * @param type The class.
   * @param onStack The classes being linearized on this path, for the cycle guard.
   * @return The linearization, starting with the class.
   */
  private static List<IClass> methodResolutionOrder(
      IClassHierarchy cha, IClass type, Set<IClass> onStack) {
    List<IClass> bases = declaredBases(cha, type);
    if (!isProgramClass(type) || bases.isEmpty() || !onStack.add(type))
      return new ArrayList<>(List.of(type));
    try {
      List<List<IClass>> sequences = new ArrayList<>();
      for (IClass base : bases) sequences.add(methodResolutionOrder(cha, base, onStack));
      sequences.add(new ArrayList<>(bases));
      // The merge consumes its sequences, so it gets copies: the fallback needs them whole.
      List<List<IClass>> consumed = new ArrayList<>();
      for (List<IClass> sequence : sequences) consumed.add(new ArrayList<>(sequence));
      List<IClass> merged = c3Merge(consumed);
      List<IClass> order = new ArrayList<>();
      order.add(type);
      if (merged != null) order.addAll(merged);
      else
        for (List<IClass> sequence : sequences)
          for (IClass c : sequence) if (!order.contains(c)) order.add(c);
      return order;
    } finally {
      onStack.remove(type);
    }
  }

  /**
   * The classes a class's declared base names resolve to, in declaration order.
   *
   * @param cha The class hierarchy.
   * @param type The class.
   * @return The resolved bases.
   */
  private static List<IClass> declaredBases(IClassHierarchy cha, IClass type) {
    List<IClass> bases = new ArrayList<>();
    if (type instanceof IPythonClass python)
      for (TypeName baseName : python.getBaseTypeNames()) {
        IClass base =
            cha.lookupClass(
                TypeReference.findOrCreate(type.getClassLoader().getReference(), baseName));
        if (base != null && !bases.contains(base)) bases.add(base);
      }
    if (bases.isEmpty() && type.getSuperclass() != null) bases.add(type.getSuperclass());
    // `object` ends every class's order in Python, after every other base; a library class's own
    // bases are not expanded here, so keeping it would place it before a later base's branch.
    bases.removeIf(
        base ->
            base.getReference().equals(PythonTypes.object)
                || base.getReference().equals(PythonTypes.Root));
    return bases;
  }

  /**
   * The C3 merge of the given sequences.
   *
   * @param sequences The bases' linearizations followed by the bases themselves; consumed.
   * @return The merge, or {@code null} when no consistent order exists.
   */
  private static List<IClass> c3Merge(List<List<IClass>> sequences) {
    List<IClass> result = new ArrayList<>();
    while (true) {
      sequences.removeIf(List::isEmpty);
      if (sequences.isEmpty()) return result;
      IClass head = null;
      for (List<IClass> sequence : sequences) {
        IClass candidate = sequence.get(0);
        boolean inTail = false;
        for (List<IClass> other : sequences)
          if (other.indexOf(candidate) > 0) {
            inTail = true;
            break;
          }
        if (!inTail) {
          head = candidate;
          break;
        }
      }
      if (head == null) return null;
      result.add(head);
      for (List<IClass> sequence : sequences) if (sequence.get(0).equals(head)) sequence.remove(0);
    }
  }

  /**
   * The callable an instance is called through: its class's {@code __call__}, one a program-defined
   * base declares (wala/ML#994), its class's {@code do} or Keras {@code call}, or a {@code call} or
   * {@code do} a program-defined base declares, in Python's lookup order, the bases in method
   * resolution order. The class is the instance's concrete type when that names a callable, and
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

      // `Layer.__call__` finds `call` along the method resolution order, so a subclass that
      // inherits its `call` (or `do`) from a program-defined base is called through the base's.
      // The lookups above name the instance's own class only.
      if (callable == null)
        callable =
            inheritedMethod(
                cha, classLoaderReference, declaringClass, CALLABLE_METHOD_NAME_FOR_KERAS_MODELS);
      if (callable == null)
        callable = inheritedMethod(cha, classLoaderReference, declaringClass, DO_METHOD_NAME);

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
