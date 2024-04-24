package com.ibm.wala.cast.python.types;

import com.ibm.wala.cast.types.AstTypeReference;
import com.ibm.wala.classLoader.IClassLoader;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;

public class Util {

  private static final String GLOBAL_KEYWORD = "global";

  /**
   * Returns the filename portion of the given {@link TypeName} representing a Python type.
   *
   * @param typeName A {@link TypeName} of a Python type.
   * @return The filename portion of the given {@link TypeName}.
   * @apiNote Python types include a file in their {@link TypeName}s in Ariadne.
   */
  public static String getFilename(final TypeName typeName) {
    String ret = typeName.toString();
    ret = ret.substring("Lscript ".length());

    if (ret.indexOf('/') != -1) ret = ret.substring(0, ret.indexOf('/'));

    return ret;
  }

  /**
   * creates a reference to a global named globalName. the declaring type and type of the global are
   * both the root type.
   */
  public static FieldReference makeGlobalRef(IClassLoader loader, String globalName) {
    TypeReference rootTypeRef =
        TypeReference.findOrCreate(loader.getReference(), AstTypeReference.rootTypeName);
    return FieldReference.findOrCreate(
        rootTypeRef, Atom.findOrCreateUnicodeAtom(GLOBAL_KEYWORD + " " + globalName), rootTypeRef);
  }

  private Util() {}
}
