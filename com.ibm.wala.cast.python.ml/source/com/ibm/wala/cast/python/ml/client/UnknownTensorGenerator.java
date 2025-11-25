package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TYPE_REFERENCE_TO_SIGNATURE;
import static com.ibm.wala.cast.python.util.Util.getFunction;
import static java.lang.Integer.parseInt;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.TypeReference;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

public class UnknownTensorGenerator extends TensorGenerator {

  private static final double TEMPERATURE = 0.0;

  private static final String API_KEY_ENV_KEY = "GOOGLE_API_KEY";

  private static final String API_KEY = System.getenv(API_KEY_ENV_KEY);

  private static final String MODEL_NAME = "gemini-2.5-flash-lite";

  private static final ChatModel MODEL =
      GoogleAiGeminiChatModel.builder()
          .apiKey(API_KEY)
          .modelName(MODEL_NAME)
          .temperature(TEMPERATURE)
          .build();

  public UnknownTensorGenerator(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  protected int getShapeParameterPosition() {
    // TODO: Handle keyword arguments.
    String answer =
        MODEL.chat(
            "Does `"
                + this.getSignature()
                + "` have a parameter named \"shape?\" If so, respond with its position"
                + " (0-indexed)? If not, respond with -1. Only respond with the integer.");
    return parseInt(answer);
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    // TODO: Handle keyword arguments.
    String answer =
        MODEL.chat(
            "Does `"
                + this.getSignature()
                + "` have a parameter named \"dtype?\" If so, respond with its position"
                + " (0-indexed)? If not, respond with -1. Only respond with the integer.");
    return parseInt(answer);
  }

  @Override
  protected String getSignature() {
    TypeReference function = getFunction(this.getSource());
    return TYPE_REFERENCE_TO_SIGNATURE.get(function);
  }
}
