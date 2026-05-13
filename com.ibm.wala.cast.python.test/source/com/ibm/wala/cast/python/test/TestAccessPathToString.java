package com.ibm.wala.cast.python.test;

import static org.junit.Assert.assertEquals;

import com.ibm.wala.cast.python.analysis.ap.GlobalVarAP;
import com.ibm.wala.cast.python.analysis.ap.PropertyPathElement;
import org.junit.Test;

/**
 * Unit tests for the {@code toString()} methods on {@link GlobalVarAP} and {@link
 * PropertyPathElement}. The surrounding {@code equals}/{@code hashCode} on these classes admit
 * {@code null} field values, so {@code toString()} routes through {@link
 * java.util.Objects#toString} to satisfy {@link Object#toString}'s non-null contract for both
 * branches.
 */
public class TestAccessPathToString {

  @Test
  public void globalVarAPToStringReturnsName() {
    assertEquals("x", GlobalVarAP.createGlobalVarAP("x").toString());
  }

  @Test
  public void globalVarAPToStringHandlesNull() {
    assertEquals("null", GlobalVarAP.createGlobalVarAP(null).toString());
  }

  @Test
  public void propertyPathElementToStringReturnsField() {
    assertEquals("f", PropertyPathElement.createFieldPathElement("f").toString());
  }

  @Test
  public void propertyPathElementToStringHandlesNull() {
    assertEquals("null", PropertyPathElement.createFieldPathElement(null).toString());
  }
}
