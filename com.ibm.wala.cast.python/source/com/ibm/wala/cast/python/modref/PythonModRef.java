package com.ibm.wala.cast.python.modref;

import com.google.common.collect.MapMaker;
import com.ibm.wala.cast.ipa.callgraph.AstHeapModel;
import com.ibm.wala.cast.ipa.modref.AstModRef;
import com.ibm.wala.cast.ir.ssa.AstGlobalWrite;
import com.ibm.wala.cast.ir.ssa.AstPropertyWrite;
import com.ibm.wala.cast.python.ssa.PythonInstructionVisitor;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.StaticFieldKey;
import com.ibm.wala.ipa.modref.ExtendedHeapModel;
import com.ibm.wala.util.collections.HashMapFactory;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class PythonModRef extends AstModRef<InstanceKey> {

  /**
   * The fields of each pointer analysis's {@link StaticFieldKey}s, by name. An analysis's entry is
   * built from one read of its pointer keys, the first time one of its global writes is visited,
   * and every visitor of that analysis shares it, whichever {@code makeModVisitor} overload or
   * {@code getMod} path made the visitor (https://github.com/wala/ML/issues/1063). The keys are
   * weak and compared by identity, and an entry holds only fields, so it goes once its analysis is
   * unreachable. The index assumes the analysis is final, as it is once its call graph is built.
   */
  private static final Map<PointerAnalysis<?>, Map<String, List<IField>>> STATIC_FIELDS =
      new MapMaker().weakKeys().makeMap();

  /**
   * The fields of the given analysis's static field keys with the given name.
   *
   * @param pa The pointer analysis.
   * @param name The name of a global variable.
   * @return The fields named {@code name}.
   */
  private static List<IField> staticFieldsNamed(PointerAnalysis<?> pa, String name) {
    Map<String, List<IField>> byName = STATIC_FIELDS.get(pa);
    if (byName == null)
      // Build each analysis's index once, even when visitors on several threads ask for it first.
      synchronized (STATIC_FIELDS) {
        byName = STATIC_FIELDS.get(pa);
        if (byName == null) {
          byName = indexStaticFields(pa);
          STATIC_FIELDS.put(pa, byName);
        }
      }
    return byName.getOrDefault(name, Collections.emptyList());
  }

  /**
   * Indexes the fields of the given analysis's static field keys by name.
   *
   * @param pa The pointer analysis.
   * @return The fields of {@code pa}'s {@link StaticFieldKey}s, by name, unmodifiable.
   */
  private static Map<String, List<IField>> indexStaticFields(PointerAnalysis<?> pa) {
    Map<String, List<IField>> byName = HashMapFactory.make();
    for (PointerKey pk : pa.getPointerKeys())
      if (pk instanceof StaticFieldKey) {
        IField field = ((StaticFieldKey) pk).getField();
        byName.computeIfAbsent(field.getName().toString(), k -> new ArrayList<>()).add(field);
      }
    Map<String, List<IField>> ret = HashMapFactory.make();
    byName.forEach((name, fields) -> ret.put(name, List.copyOf(fields)));
    return Collections.unmodifiableMap(ret);
  }

  public static class PythonRefVisitor<T extends InstanceKey> extends AstRefVisitor<T>
      implements PythonInstructionVisitor {

    public PythonRefVisitor(
        CGNode n, Collection<PointerKey> result, PointerAnalysis<T> pa, ExtendedHeapModel h) {
      super(n, result, pa, (AstHeapModel) h);
    }
  }

  @Override
  protected RefVisitor<InstanceKey, ? extends ExtendedHeapModel> makeRefVisitor(
      CGNode n,
      Collection<PointerKey> result,
      PointerAnalysis<InstanceKey> pa,
      ExtendedHeapModel h) {
    return new PythonRefVisitor<>(n, result, pa, h);
  }

  public static class PythonModVisitor<T extends InstanceKey> extends AstModVisitor<T>
      implements PythonInstructionVisitor {

    public PythonModVisitor(
        CGNode n,
        Collection<PointerKey> result,
        ExtendedHeapModel h,
        PointerAnalysis<T> pa,
        boolean ignoreAllocHeapDefs) {
      super(n, result, (AstHeapModel) h, pa);
    }

    @Override
    public void visitAstGlobalWrite(AstGlobalWrite instruction) {
      super.visitAstGlobalWrite(instruction);

      // find the pointer keys corresponding to this global.
      for (IField field : staticFieldsNamed(this.pa, instruction.getGlobalName()))
        this.result.add(this.h.getPointerKeyForStaticField(field));
    }

    @Override
    public void visitPropertyWrite(AstPropertyWrite instruction) {
      super.visitPropertyWrite(instruction);

      // prune "writes" to ConstantKeys. See https://github.com/wala/ML/issues/103.
      for (Iterator<PointerKey> it = this.result.iterator(); it.hasNext(); ) {
        PointerKey pointerKey = it.next();

        if (pointerKey instanceof InstanceFieldPointerKey) {
          InstanceFieldPointerKey instanceFieldPointerKey = (InstanceFieldPointerKey) pointerKey;
          InstanceKey instanceKey = instanceFieldPointerKey.getInstanceKey();

          if (instanceKey instanceof ConstantKey) it.remove();
        }
      }
    }
  }

  @Override
  protected ModVisitor<InstanceKey, ? extends ExtendedHeapModel> makeModVisitor(
      CGNode n,
      Collection<PointerKey> result,
      PointerAnalysis<InstanceKey> pa,
      ExtendedHeapModel h,
      boolean ignoreAllocHeapDefs) {
    return new PythonModVisitor<>(n, result, h, pa, ignoreAllocHeapDefs);
  }
}
