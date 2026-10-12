package com.ibm.wala.cast.python.modref;

import com.ibm.wala.cast.ipa.callgraph.AstHeapModel;
import com.ibm.wala.cast.ipa.modref.AstModRef;
import com.ibm.wala.cast.ir.ssa.AstGlobalWrite;
import com.ibm.wala.cast.ir.ssa.AstPropertyWrite;
import com.ibm.wala.cast.python.ssa.PythonInstructionVisitor;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.StaticFieldKey;
import com.ibm.wala.ipa.modref.ExtendedHeapModel;
import com.ibm.wala.ipa.slicer.HeapExclusions;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class PythonModRef extends AstModRef<InstanceKey> {

  /**
   * The fields of a pointer analysis's {@link StaticFieldKey}s, indexed by name on first use, so
   * that a global write finds its keys without reading every pointer key of the analysis
   * (https://github.com/wala/ML/issues/1063).
   */
  private static final class StaticFields {

    /** The pointer analysis whose static fields these are. */
    private final PointerAnalysis<?> pa;

    /** The fields by name, or {@code null} before the first lookup. */
    private Map<String, List<IField>> byName;

    private StaticFields(PointerAnalysis<?> pa) {
      this.pa = pa;
    }

    /**
     * The fields of the analysis's static field keys with the given name.
     *
     * @param name The name of a global variable.
     * @return The fields named {@code name}.
     */
    private List<IField> named(String name) {
      if (this.byName == null) {
        this.byName = HashMapFactory.make();
        for (PointerKey pk : this.pa.getPointerKeys())
          if (pk instanceof StaticFieldKey) {
            IField field = ((StaticFieldKey) pk).getField();
            this.byName
                .computeIfAbsent(field.getName().toString(), k -> new ArrayList<>())
                .add(field);
          }
      }
      return this.byName.getOrDefault(name, Collections.emptyList());
    }
  }

  /**
   * The static fields of the analysis of the {@link #computeMod} under way, shared by the visitors
   * of all its nodes (https://github.com/wala/ML/issues/1063).
   */
  private StaticFields staticFields;

  @Override
  public Map<CGNode, OrdinalSet<PointerKey>> computeMod(
      CallGraph cg, PointerAnalysis<InstanceKey> pa, HeapExclusions heapExclude) {
    this.staticFields = new StaticFields(pa);
    return super.computeMod(cg, pa, heapExclude);
  }

  /**
   * {@inheritDoc}
   *
   * <p>This is the factory {@link #computeMod} makes each node's visitor with. It hands a {@link
   * PythonModVisitor} the static fields of the analysis under way, so that a subclass overriding
   * the factory that this one delegates to shares them too.
   */
  @Override
  protected ModVisitor<InstanceKey, ?> makeModVisitor(
      CGNode n,
      Collection<PointerKey> result,
      PointerAnalysis<InstanceKey> pa,
      ExtendedHeapModel h) {
    ModVisitor<InstanceKey, ?> visitor = super.makeModVisitor(n, result, pa, h);
    if (visitor instanceof PythonModVisitor
        && this.staticFields != null
        && this.staticFields.pa == pa)
      ((PythonModVisitor<?>) visitor).staticFields = this.staticFields;
    return visitor;
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

    /**
     * The analysis's static fields: those of the {@link PythonModRef#computeMod} that made this
     * visitor, else, made on first use, this visitor's own.
     */
    private StaticFields staticFields;

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
      if (this.staticFields == null) this.staticFields = new StaticFields(this.pa);
      for (IField field : this.staticFields.named(instruction.getGlobalName()))
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
