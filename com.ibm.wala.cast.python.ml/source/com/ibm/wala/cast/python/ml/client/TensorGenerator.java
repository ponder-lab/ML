package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

public abstract class TensorGenerator {

  protected static final Logger LOGGER = Logger.getLogger(TensorGenerator.class.getName());

  protected PointsToSetVariable source;

  protected CGNode node;

  public TensorGenerator(PointsToSetVariable source, CGNode node) {
    this.source = source;
    this.node = node;
  }

  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> shapes = getShapes(builder);
    EnumSet<DType> dTypes = getDTypes(builder);

    Set<TensorType> ret = HashSetFactory.make();

    // Create a tensor type for each possible shape and dtype combination.
    for (List<Dimension<?>> dimensionList : shapes)
      for (DType dtype : dTypes) ret.add(new TensorType(dtype.name().toLowerCase(), dimensionList));

    return ret;
  }

  protected abstract Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder);

  protected abstract EnumSet<DType> getDTypes(PropagationCallGraphBuilder builder);
}
