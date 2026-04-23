package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;

public class RandomNormalCall extends Normal {

  public RandomNormalCall(PointsToSetVariable source) {
    super(source);
  }

  public RandomNormalCall(CGNode node) {
    super(node);
  }

  @Override
  protected int getShapeParameterPosition() {
    return 1;
  }

  @Override
  protected String getShapeParameterName() {
    return "shape";
  }

  @Override
  protected int getDTypeParameterPosition() {
    return 2;
  }

  @Override
  protected String getDTypeParameterName() {
    return "dtype";
  }
}
