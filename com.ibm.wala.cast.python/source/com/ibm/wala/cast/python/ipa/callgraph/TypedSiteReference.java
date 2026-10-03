package com.ibm.wala.cast.python.ipa.callgraph;

import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.types.TypeReference;

/**
 * The site of an allocation the builder synthesizes, identified by its program counter AND its
 * type. A {@link NewSiteReference} is equal to another at the same program counter whatever their
 * types, since an instruction allocates one object. The builder allocates several at one
 * instruction: an array operation's result beside the methods attached to it, a slice of a tensor
 * beside a slice of an array, a list beside a tuple. Under the program counter alone those were one
 * instance key, whose type was whichever allocation registered first, so a method field could hold
 * another method's object, and which one depended on iteration order (wala/ML#1009).
 */
final class TypedSiteReference extends NewSiteReference {

  private TypedSiteReference(int programCounter, TypeReference declaredType) {
    super(programCounter, declaredType);
  }

  /**
   * The site of an allocation of the given type that the builder synthesizes at an instruction.
   *
   * @param programCounter The instruction's index.
   * @param declaredType The allocated type.
   * @return The site.
   */
  static NewSiteReference at(int programCounter, TypeReference declaredType) {
    return new TypedSiteReference(programCounter, declaredType);
  }

  @Override
  public boolean equals(Object obj) {
    return obj instanceof TypedSiteReference other
        && other.getProgramCounter() == this.getProgramCounter()
        && other.getDeclaredType().equals(this.getDeclaredType());
  }

  @Override
  public int hashCode() {
    return this.getProgramCounter() * 8191 + this.getDeclaredType().hashCode();
  }
}
