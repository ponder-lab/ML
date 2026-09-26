package com.ibm.wala.cast.python.ipa.callgraph;

import com.ibm.wala.cast.ipa.callgraph.AstCFAPointerKeys;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import java.util.Collections;
import java.util.Iterator;

/**
 * Pointer keys for Python: the {@link AstCFAPointerKeys} with no field, catalog, or reflected field
 * key for the None constant.
 *
 * <p>{@code None} is one global {@link ConstantKey} whose concrete type is {@code Root}, so the
 * core builder's null-receiver filter, which asks the language whether the key's <em>type</em> is
 * the null type, does not recognize it, and the reflected write records field names in any
 * receiver's object catalog. A write whose receiver set merely includes {@code None} (a function
 * object local also bound to {@code None}, a defaulted parameter) then lands its value on the
 * {@code None} key, and every wildcard element read over a container that may be {@code None} (a
 * {@code zip} or a {@code for} over a list that is {@code None} on one arm) reads that value back
 * as an element of an unrelated container. An attribute write on {@code None} raises at run time,
 * and so does a read, so neither carries a value: declining the keys removes edges the program
 * cannot exercise and adds none. Both operators null-guard the keys they are handed.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class PythonPointerKeys extends AstCFAPointerKeys {

  /**
   * Whether an instance key is the None constant.
   *
   * @param key The instance key.
   * @return {@code true} iff the key is the None constant.
   */
  public static boolean isNoneConstant(InstanceKey key) {
    return key instanceof ConstantKey && ((ConstantKey<?>) key).getValue() == null;
  }

  @Override
  public PointerKey getPointerKeyForInstanceField(InstanceKey I, IField f) {
    if (isNoneConstant(I)) return null;
    return super.getPointerKeyForInstanceField(I, f);
  }

  @Override
  public PointerKey getPointerKeyForObjectCatalog(InstanceKey I) {
    if (isNoneConstant(I)) return null;
    return super.getPointerKeyForObjectCatalog(I);
  }

  @Override
  public Iterator<PointerKey> getPointerKeysForReflectedFieldRead(InstanceKey I, InstanceKey F) {
    if (isNoneConstant(I)) return Collections.emptyIterator();
    return super.getPointerKeysForReflectedFieldRead(I, F);
  }

  @Override
  public Iterator<PointerKey> getPointerKeysForReflectedFieldWrite(InstanceKey I, InstanceKey F) {
    if (isNoneConstant(I)) return Collections.emptyIterator();
    return super.getPointerKeysForReflectedFieldWrite(I, F);
  }
}
