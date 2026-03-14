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
package com.ibm.wala.cast.python.client;

import static java.util.Collections.emptyList;
import static java.util.logging.Level.SEVERE;

import com.ibm.wala.cast.ipa.callgraph.AstCFAPointerKeys;
import com.ibm.wala.cast.ipa.callgraph.AstContextInsensitiveSSAContextInterpreter;
import com.ibm.wala.cast.ir.ssa.AstIRFactory;
import com.ibm.wala.cast.loader.AstDynamicField;
import com.ibm.wala.cast.python.ipa.callgraph.PythonClassMethodTrampolineTargetSelector;
import com.ibm.wala.cast.python.ipa.callgraph.PythonConstructorTargetSelector;
import com.ibm.wala.cast.python.ipa.callgraph.PythonInstanceMethodTrampolineTargetSelector;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ipa.callgraph.PythonScopeMappingInstanceKeys;
import com.ibm.wala.cast.python.ipa.summaries.BuiltinFunctions;
import com.ibm.wala.cast.python.ipa.summaries.PythonComprehensionTrampolines;
import com.ibm.wala.cast.python.ipa.summaries.PythonInstanceMethodTrampoline;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummarizedFunction;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummary;
import com.ibm.wala.cast.python.ipa.summaries.PythonSuper;
import com.ibm.wala.cast.python.ir.PythonLanguage;
import com.ibm.wala.cast.python.loader.IPythonClass;
import com.ibm.wala.cast.python.loader.PythonLoaderFactory;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.cast.util.Util;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IClassLoader;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.classLoader.Module;
import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.classLoader.SyntheticClass;
import com.ibm.wala.client.AbstractAnalysisEngine;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.AnalysisCacheImpl;
import com.ibm.wala.ipa.callgraph.AnalysisOptions;
import com.ibm.wala.ipa.callgraph.AnalysisScope;
import com.ibm.wala.ipa.callgraph.ClassTargetSelector;
import com.ibm.wala.ipa.callgraph.Entrypoint;
import com.ibm.wala.ipa.callgraph.IAnalysisCacheView;
import com.ibm.wala.ipa.callgraph.MethodTargetSelector;
import com.ibm.wala.ipa.callgraph.impl.ClassHierarchyClassTargetSelector;
import com.ibm.wala.ipa.callgraph.impl.ClassHierarchyMethodTargetSelector;
import com.ibm.wala.ipa.callgraph.impl.ContextInsensitiveSelector;
import com.ibm.wala.ipa.callgraph.impl.DefaultEntrypoint;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.cfa.ZeroXInstanceKeys;
import com.ibm.wala.ipa.callgraph.propagation.cfa.nCFAContextSelector;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ipa.cha.SeqClassHierarchyFactory;
import com.ibm.wala.ipa.summaries.BypassClassTargetSelector;
import com.ibm.wala.ipa.summaries.BypassMethodTargetSelector;
import com.ibm.wala.ipa.summaries.BypassSyntheticClassLoader;
import com.ibm.wala.ipa.summaries.MethodSummary;
import com.ibm.wala.ipa.summaries.XMLMethodSummaryReader;
import com.ibm.wala.shrike.shrikeBT.Constants;
import com.ibm.wala.ssa.IRFactory;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAInstructionFactory;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SSAOptions;
import com.ibm.wala.ssa.SSAOptions.DefaultValues;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.ClassLoaderReference;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.Selector;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.WalaException;
import com.ibm.wala.util.WalaRuntimeException;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.HashSetFactory;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

public abstract class PythonAnalysisEngine<T>
    extends AbstractAnalysisEngine<InstanceKey, PythonSSAPropagationCallGraphBuilder, T> {

  private static final Logger logger = Logger.getLogger(PythonAnalysisEngine.class.getName());

  /** Library summaries to load. */
  private static final String[] LIBRARIES =
      new String[] {
        "flask.xml", "pandas.xml", "functools.xml", "pytest.xml", "click.xml", "abseil.xml"
      };

  protected PythonSSAPropagationCallGraphBuilder builder;

  static {
    try {
      @SuppressWarnings("unchecked")
      Class<PythonLoaderFactory> j4 =
          (Class<PythonLoaderFactory>)
              Class.forName("com.ibm.wala.cast.python.loader.JepPythonLoaderFactory");
      PythonAnalysisEngine.setLoaderFactory(j4);
    } catch (UnsatisfiedLinkError | ClassNotFoundException e2) {
      try {
        @SuppressWarnings("unchecked")
        Class<? extends PythonLoaderFactory> j3 =
            (Class<? extends PythonLoaderFactory>)
                Class.forName("com.ibm.wala.cast.python.loader.Python3LoaderFactory");
        PythonAnalysisEngine.setLoaderFactory(j3);
      } catch (ClassNotFoundException e) {
        try {
          @SuppressWarnings("unchecked")
          Class<? extends PythonLoaderFactory> j2 =
              (Class<? extends PythonLoaderFactory>)
                  Class.forName("com.ibm.wala.cast.python.loader.Python2LoaderFactory");
          PythonAnalysisEngine.setLoaderFactory(j2);
        } catch (ClassNotFoundException e1) {
          assert false : e.getMessage() + ", then " + e1.getMessage();
        }
      }
    }
  }

  private static Class<? extends PythonLoaderFactory> loaders;

  public static void setLoaderFactory(Class<? extends PythonLoaderFactory> lf) {
    assert loaders == null;
    loaders = lf;
  }

  protected PythonLoaderFactory loader;

  private final IRFactory<IMethod> irs = AstIRFactory.makeDefaultFactory();

  public PythonAnalysisEngine(List<File> pythonPath) {
    PythonLoaderFactory f;
    try {
      f = loaders.getConstructor(java.util.List.class).newInstance(pythonPath);
    } catch (InstantiationException
        | IllegalAccessException
        | InvocationTargetException
        | NoSuchMethodException e) {
      f = null;
      assert false : e.getMessage();
    }
    loader = f;
  }

  public PythonAnalysisEngine() {
    this(emptyList());
  }

  @Override
  public void buildAnalysisScope() throws IOException {
    scope =
        new AnalysisScope(Collections.singleton(PythonLanguage.Python)) {
          {
            loadersByName.put(PythonTypes.pythonLoaderName, PythonTypes.pythonLoader);
            loadersByName.put(
                SYNTHETIC,
                new ClassLoaderReference(
                    SYNTHETIC, PythonLanguage.Python.getName(), PythonTypes.pythonLoader));
          }
        };

    for (Module o : moduleFiles) {
      scope.addToScope(PythonTypes.pythonLoader, o);
    }
  }

  @Override
  public IClassHierarchy buildClassHierarchy() {
    IClassHierarchy cha = null;
    try {
      cha = SeqClassHierarchyFactory.make(scope, loader);
    } catch (ClassHierarchyException e) {
      final String msg = "Failed to build class hierarchy.";
      logger.log(SEVERE, msg, e);
      throw new WalaRuntimeException(msg, e);
    }

    try {
      Util.checkForFrontEndErrors(cha);
    } catch (WalaException e) {
      logger.log(Level.WARNING, e, () -> "Encountered WALA exception: " + e.getLocalizedMessage());
    }

    setClassHierarchy(cha);
    return cha;
  }

  private abstract class AbstractPythonSyntheticClass extends SyntheticClass
      implements IPythonClass {
    public AbstractPythonSyntheticClass(TypeReference T, IClassHierarchy cha) {
      super(T, cha);
    }
  }

  protected void addSummaryBypassLogic(AnalysisOptions options, String summary) {
    IClassHierarchy cha = getClassHierarchy();
    XMLMethodSummaryReader xml =
        new XMLMethodSummaryReader(getClass().getClassLoader().getResourceAsStream(summary), scope);

    Map<MethodReference, MethodSummary> summaries = new HashMap<>(xml.getSummaries());
    BypassSyntheticClassLoader ldr =
        (BypassSyntheticClassLoader) cha.getLoader(scope.getSyntheticLoader());

    Map<TypeReference, List<MethodReference>> classToFunDoRefs = new HashMap<>();

    // Pass 1: Transform methods to function classes
    for (MethodSummary s : xml.getSummaries().values()) {
      MethodReference mr = s.getMethod();
      String methodName = mr.getName().toString();
      if (!methodName.equals("do")
          && !methodName.equals("import")
          && !methodName.equals("__init__")) {
        TypeReference t = mr.getDeclaringClass();
        TypeReference funClsRef =
            TypeReference.findOrCreate(
                t.getClassLoader(), t.getName().toString() + "/" + methodName);
        MethodReference funDoRef =
            MethodReference.findOrCreate(funClsRef, AstMethodReference.fnSelector);

        List<MethodReference> funDoRefs =
            classToFunDoRefs.computeIfAbsent(t, k -> new ArrayList<>());
        funDoRefs.add(funDoRef);

        PythonSummary funSummary = new PythonSummary(funDoRef, s.getNumberOfParameters());
        for (SSAInstruction inst : s.getStatements())
          if (inst != null) funSummary.addStatement(inst);
        funSummary.setValueNames(s.getValueNames());
        summaries.put(funDoRef, funSummary);
        summaries.remove(mr);

        ldr.registerClass(
            funClsRef.getName(),
            new SyntheticClass(funClsRef, cha) {
              @Override
              public IClassLoader getClassLoader() {
                return ldr;
              }

              @Override
              public boolean isPublic() {
                return true;
              }

              @Override
              public int getModifiers() {
                return Constants.ACC_PUBLIC;
              }

              @Override
              public IClass getSuperclass() {
                return cha.lookupClass(PythonTypes.CodeBody);
              }

              @Override
              public Collection<? extends IClass> getDirectInterfaces() {
                return Collections.emptySet();
              }

              @Override
              public Collection<IClass> getAllImplementedInterfaces() {
                return Collections.emptySet();
              }

              @Override
              public IMethod getMethod(Selector selector) {
                return selector.equals(AstMethodReference.fnSelector)
                    ? new PythonSummarizedFunction(funDoRef, funSummary, this)
                    : null;
              }

              @Override
              public IField getField(Atom name) {
                return null;
              }

              @Override
              public IMethod getClassInitializer() {
                return null;
              }

              @Override
              public Collection<? extends IMethod> getDeclaredMethods() {
                return Collections.singleton(getMethod(AstMethodReference.fnSelector));
              }

              @Override
              public Collection<IField> getAllInstanceFields() {
                return Collections.emptySet();
              }

              @Override
              public Collection<IField> getAllStaticFields() {
                return Collections.emptySet();
              }

              @Override
              public Collection<IField> getAllFields() {
                return Collections.emptySet();
              }

              @Override
              public Collection<? extends IMethod> getAllMethods() {
                return getDeclaredMethods();
              }

              @Override
              public Collection<IField> getDeclaredInstanceFields() {
                return Collections.emptySet();
              }

              @Override
              public Collection<IField> getDeclaredStaticFields() {
                return Collections.emptySet();
              }

              @Override
              public boolean isReferenceType() {
                return true;
              }

              @Override
              public boolean isPrivate() {
                return false;
              }
            });
      }
    }

    // Pass 2: Identify and rewrite constructors to inject trampolines
    for (MethodReference mr : new ArrayList<>(summaries.keySet())) {
      if (mr.getName().toString().equals("do")) {
        MethodSummary s = summaries.get(mr);
        int returnVn = -1;
        TypeReference instanceType = null;
        for (SSAInstruction inst : s.getStatements()) {
          if (inst != null) {
            if (inst instanceof SSANewInstruction) {
              instanceType = ((SSANewInstruction) inst).getNewSite().getDeclaredType();
              returnVn = ((SSANewInstruction) inst).getDef();
            } else if (inst.toString().contains(" = new <")) {
              try {
                java.lang.reflect.Method m = inst.getClass().getMethod("getNewSite");
                NewSiteReference site = (NewSiteReference) m.invoke(inst);
                instanceType = site.getDeclaredType();
                m = inst.getClass().getMethod("getDef");
                returnVn = (Integer) m.invoke(inst);
              } catch (Exception e) {
              }
            }
          }
        }

        boolean isAutoGenerated = false;
        int stmtCount = 0;
        for (SSAInstruction inst : s.getStatements()) if (inst != null) stmtCount++;
        if (stmtCount == 2 && instanceType != null && instanceType.equals(PythonTypes.object)) {
          isAutoGenerated = true;
        }

        if (isAutoGenerated && xml.getAllocatableClasses().contains(mr.getDeclaringClass())) {
          // Remove auto-generated 'do' methods
          summaries.remove(mr);
        } else if (instanceType != null && classToFunDoRefs.containsKey(instanceType)) {
          // It's an explicit constructor for a class with methods, rewrite it to inject trampolines
          PythonSummary newSummary = new PythonSummary(s.getMethod(), s.getNumberOfParameters());
          newSummary.setValueNames(s.getValueNames());
          SSAReturnInstruction returnInst = null;
          for (SSAInstruction inst : s.getStatements()) {
            if (inst instanceof SSAReturnInstruction) returnInst = (SSAReturnInstruction) inst;
            else if (inst != null) newSummary.addStatement(inst);
          }

          int v = 2000, pc = newSummary.getNumberOfStatements();
          SSAInstructionFactory insts = PythonLanguage.Python.instructionFactory();
          for (MethodReference funDoRef : classToFunDoRefs.get(instanceType)) {
            TypeReference funClsRef = funDoRef.getDeclaringClass();
            int funObjVn = v++, trampVn = v++;
            newSummary.addStatement(
                insts.NewInstruction(pc++, funObjVn, NewSiteReference.make(pc, funClsRef)));
            newSummary.addStatement(
                insts.NewInstruction(
                    pc++,
                    trampVn,
                    NewSiteReference.make(
                        pc, PythonInstanceMethodTrampoline.findOrCreate(funClsRef, cha))));
            newSummary.addStatement(
                insts.PutInstruction(
                    pc++,
                    trampVn,
                    returnVn,
                    FieldReference.findOrCreate(
                        PythonTypes.Root,
                        Atom.findOrCreateUnicodeAtom("$self"),
                        PythonTypes.Root)));
            newSummary.addStatement(
                insts.PutInstruction(
                    pc++,
                    trampVn,
                    funObjVn,
                    FieldReference.findOrCreate(
                        PythonTypes.Root,
                        Atom.findOrCreateUnicodeAtom("$function"),
                        PythonTypes.Root)));

            String fieldName =
                funClsRef
                    .getName()
                    .toString()
                    .substring(funClsRef.getName().toString().lastIndexOf('/') + 1);
            newSummary.addStatement(
                insts.PutInstruction(
                    pc++,
                    returnVn,
                    trampVn,
                    FieldReference.findOrCreate(
                        PythonTypes.Root,
                        Atom.findOrCreateUnicodeAtom(fieldName),
                        PythonTypes.Root)));

            // For callable objects, also bind the trampoline to the 'do' field to handle implicit
            // c() calls which are often translated to c.do()
            if (fieldName.equals("__call__") || fieldName.equals("call")) {
              newSummary.addStatement(
                  insts.PutInstruction(
                      pc++,
                      returnVn,
                      trampVn,
                      FieldReference.findOrCreate(
                          PythonTypes.Root, Atom.findOrCreateUnicodeAtom("do"), PythonTypes.Root)));
            }
          }
          if (returnInst != null) newSummary.addStatement(returnInst);
          summaries.put(s.getMethod(), newSummary);
        }
      }
    }

    // Pass 3: Register classes. Use AbstractPythonSyntheticClass only for classes with methods
    // (requiring trampolines)
    for (TypeReference t : xml.getAllocatableClasses()) {
      if (classToFunDoRefs.containsKey(t) && !classToFunDoRefs.get(t).isEmpty()) {
        ldr.registerClass(
            t.getName(),
            new AbstractPythonSyntheticClass(t, cha) {
              private final Map<Atom, IField> fields = HashMapFactory.make();

              @Override
              public IClassLoader getClassLoader() {
                return ldr;
              }

              @Override
              public boolean isPublic() {
                return true;
              }

              @Override
              public boolean isPrivate() {
                return false;
              }

              @Override
              public int getModifiers() {
                return Constants.ACC_PUBLIC;
              }

              @Override
              public IClass getSuperclass() {
                return cha.lookupClass(PythonTypes.object);
              }

              @Override
              public Collection<? extends IClass> getDirectInterfaces() {
                return Collections.emptySet();
              }

              @Override
              public Collection<IClass> getAllImplementedInterfaces() {
                return Collections.emptySet();
              }

              @Override
              public IMethod getMethod(Selector selector) {
                for (IMethod m : getDeclaredMethods())
                  if (m.getSelector().equals(selector)) return m;
                return null;
              }

              @Override
              public IField getField(Atom name) {
                if (!fields.containsKey(name))
                  fields.put(
                      name,
                      new AstDynamicField(
                          false, cha.lookupClass(PythonTypes.Root), name, PythonTypes.Root));
                return fields.get(name);
              }

              @Override
              public IMethod getClassInitializer() {
                return null;
              }

              @Override
              public Collection<? extends IMethod> getDeclaredMethods() {
                Set<IMethod> methods = new HashSet<>();
                for (MethodSummary s : summaries.values())
                  if (s.getMethod().getDeclaringClass().equals(getReference()))
                    methods.add(new PythonSummarizedFunction(s.getMethod(), s, this));
                return methods;
              }

              @Override
              public Collection<IField> getAllInstanceFields() {
                return fields.values();
              }

              @Override
              public Collection<IField> getAllStaticFields() {
                return Collections.emptySet();
              }

              @Override
              public Collection<IField> getAllFields() {
                return fields.values();
              }

              @Override
              public Collection<? extends IMethod> getAllMethods() {
                return getDeclaredMethods();
              }

              @Override
              public Collection<IField> getDeclaredInstanceFields() {
                return fields.values();
              }

              @Override
              public Collection<IField> getDeclaredStaticFields() {
                return Collections.emptySet();
              }

              @Override
              public boolean isReferenceType() {
                return true;
              }

              @Override
              public Collection<MethodReference> getMethodReferences() {
                return classToFunDoRefs.getOrDefault(getReference(), Collections.emptyList());
              }

              @Override
              public Collection<TypeReference> getInnerReferences() {
                return Collections.emptySet();
              }
            });
      } else {
        ldr.registerClass(
            t.getName(),
            new SyntheticClass(t, cha) {
              private final Map<Atom, IField> fields = HashMapFactory.make();

              @Override
              public IClassLoader getClassLoader() {
                return ldr;
              }

              @Override
              public boolean isPublic() {
                return true;
              }

              @Override
              public boolean isPrivate() {
                return false;
              }

              @Override
              public int getModifiers() {
                return Constants.ACC_PUBLIC;
              }

              @Override
              public IClass getSuperclass() {
                return cha.lookupClass(PythonTypes.CodeBody);
              }

              @Override
              public Collection<? extends IClass> getDirectInterfaces() {
                return Collections.emptySet();
              }

              @Override
              public Collection<IClass> getAllImplementedInterfaces() {
                return Collections.emptySet();
              }

              @Override
              public IMethod getMethod(Selector selector) {
                return null;
              }

              @Override
              public IField getField(Atom name) {
                if (!fields.containsKey(name))
                  fields.put(
                      name,
                      new AstDynamicField(
                          false, cha.lookupClass(PythonTypes.Root), name, PythonTypes.Root));
                return fields.get(name);
              }

              @Override
              public IMethod getClassInitializer() {
                return null;
              }

              @Override
              public Collection<? extends IMethod> getDeclaredMethods() {
                return null;
              }

              @Override
              public Collection<IField> getAllInstanceFields() {
                return fields.values();
              }

              @Override
              public Collection<IField> getAllStaticFields() {
                return Collections.emptySet();
              }

              @Override
              public Collection<IField> getAllFields() {
                return fields.values();
              }

              @Override
              public Collection<? extends IMethod> getAllMethods() {
                return null;
              }

              @Override
              public Collection<IField> getDeclaredInstanceFields() {
                return fields.values();
              }

              @Override
              public Collection<IField> getDeclaredStaticFields() {
                return Collections.emptySet();
              }

              @Override
              public boolean isReferenceType() {
                return true;
              }
            });
      }
    }

    MethodTargetSelector targetSelector = options.getMethodTargetSelector();
    targetSelector =
        new BypassMethodTargetSelector(targetSelector, summaries, xml.getIgnoredPackages(), cha);
    options.setSelector(targetSelector);

    ClassTargetSelector cs =
        new BypassClassTargetSelector(
            options.getClassTargetSelector(), xml.getAllocatableClasses(), cha, ldr);
    options.setSelector(cs);
  }

  protected void addBypassLogic(IClassHierarchy cha, AnalysisOptions options) {
    options.setSelector(
        new PythonInstanceMethodTrampolineTargetSelector<T>(
            new PythonClassMethodTrampolineTargetSelector<T>(
                new PythonConstructorTargetSelector(
                    new PythonComprehensionTrampolines(options.getMethodTargetSelector()))),
            this));

    BuiltinFunctions builtins = new BuiltinFunctions(cha);
    options.setSelector(builtins.builtinClassTargetSelector(options.getClassTargetSelector()));

    // load the library summaries.
    Arrays.stream(LIBRARIES).forEach(l -> addSummaryBypassLogic(options, l));
  }

  @Override
  public IClassHierarchy getClassHierarchy() {
    return super.getClassHierarchy();
  }

  protected Iterable<Entrypoint> makeDefaultEntrypoints(IClassHierarchy cha) {
    Set<Entrypoint> result = HashSetFactory.make();
    cha.forEach(
        entry -> {
          if (entry.getName().toString().endsWith(".py")) {
            MethodReference er =
                MethodReference.findOrCreate(entry.getReference(), AstMethodReference.fnSelector);
            result.add(new DefaultEntrypoint(er, cha));
          }
        });
    return result;
  }

  public PythonSSAPropagationCallGraphBuilder getCachedCallGraphBuilder() {
    return this.builder;
  }

  @Override
  protected PythonSSAPropagationCallGraphBuilder getCallGraphBuilder(
      IClassHierarchy cha, AnalysisOptions options, IAnalysisCacheView cache2) {
    IAnalysisCacheView cache = new AnalysisCacheImpl(irs, options.getSSAOptions());

    options.setSelector(new ClassHierarchyClassTargetSelector(cha));
    options.setSelector(new ClassHierarchyMethodTargetSelector(cha));

    addBypassLogic(cha, options);

    options.setUseConstantSpecificKeys(true);

    SSAOptions ssaOptions = options.getSSAOptions();
    ssaOptions.setDefaultValues(
        new DefaultValues() {
          @Override
          public int getDefaultValue(SymbolTable symtab, int valueNumber) {
            return symtab.getNullConstant();
          }
        });
    options.setSSAOptions(ssaOptions);

    PythonSSAPropagationCallGraphBuilder builder = makeBuilder(cha, options, cache);

    AstContextInsensitiveSSAContextInterpreter interpreter =
        new AstContextInsensitiveSSAContextInterpreter(options, cache);
    builder.setContextInterpreter(interpreter);

    builder.setContextSelector(new nCFAContextSelector(1, new ContextInsensitiveSelector()));

    builder.setInstanceKeys(
        new PythonScopeMappingInstanceKeys(
            builder,
            new ZeroXInstanceKeys(options, cha, interpreter, ZeroXInstanceKeys.ALLOCATIONS)));

    new PythonSuper(cha).handleSuperCalls(builder, options);

    return this.builder = builder;
  }

  protected PythonSSAPropagationCallGraphBuilder makeBuilder(
      IClassHierarchy cha, AnalysisOptions options, IAnalysisCacheView cache) {
    return new PythonSSAPropagationCallGraphBuilder(cha, options, cache, new AstCFAPointerKeys());
  }

  public abstract T performAnalysis(PropagationCallGraphBuilder builder) throws CancelException;
}
