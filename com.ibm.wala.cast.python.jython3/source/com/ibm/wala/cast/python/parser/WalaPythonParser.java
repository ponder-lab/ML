package com.ibm.wala.cast.python.parser;

import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.antlr.runtime.CharStream;
import org.antlr.runtime.CommonToken;
import org.antlr.runtime.CommonTokenStream;
import org.antlr.runtime.Token;
import org.antlr.runtime.TokenRewriteStream;
import org.python.antlr.AnalyzingParser;
import org.python.antlr.PythonLexer;
import org.python.antlr.PythonParser;
import org.python.antlr.PythonTokenSource;
import org.python.antlr.WalaErrorHandler;

public class WalaPythonParser extends AnalyzingParser {
  private String text = null;
  private final Set<Exception> errors = HashSetFactory.make();

  public WalaPythonParser(CharStream stream, String filename, String encoding) {
    super(stream, filename, encoding);
  }

  private TokenRewriteStream tokens;

  public void recordError(Exception e) {
    errors.add(e);
  }

  public Set<Exception> getErrors() {
    return errors;
  }

  public String getText(int start, int end) {
    int e = Math.min(end, this.getOriginalText().length() - 1);
    if (start >= e) {
      return "";
    } else {
      return text.substring(start, e);
    }
  }

  /**
   * The whole text the parser read.
   *
   * @return The text, every character of it.
   */
  public String getOriginalText() {
    if (text == null) {
      text = tokens.toOriginalString();
    }
    return text;
  }

  /**
   * The string literal tokens the parser joins into one string from the token starting at a
   * character index: that token and every string literal token that follows it with only hidden
   * tokens, such as whitespace, comments and line breaks inside brackets, between them. The parser
   * keeps only the first token of such a run on the string it makes (wala/ML#1231).
   *
   * @param charStartIndex The index of the first token's first character.
   * @return The run's tokens in order, or an empty list if no string literal token starts there.
   */
  public List<CommonToken> getStringRun(int charStartIndex) {
    if (stringTokensByStart == null) {
      stringTokensByStart = HashMapFactory.make();
      List<?> all = tokens.getTokens();
      for (int i = 0; i < all.size(); i++)
        if (all.get(i) instanceof CommonToken t && t.getType() == PythonLexer.STRING)
          stringTokensByStart.put(t.getStartIndex(), i);
    }
    Integer first = stringTokensByStart.get(charStartIndex);
    if (first == null) return Collections.emptyList();
    List<?> all = tokens.getTokens();
    List<CommonToken> ret = new ArrayList<>();
    for (int i = first; i < all.size(); i++) {
      CommonToken t = (CommonToken) all.get(i);
      if (t.getType() == PythonLexer.STRING) ret.add(t);
      else if (t.getChannel() != Token.HIDDEN_CHANNEL) break;
    }
    return ret;
  }

  /** The index in the token stream of each string literal token, by its first character's index. */
  private Map<Integer, Integer> stringTokensByStart;

  protected PythonParser setupParser(boolean single) {
    WalaErrorHandler weh = new WalaErrorHandler(errorHandler, this);
    PythonLexer lexer = new PythonLexer(charStream);
    lexer.setErrorHandler(weh);
    lexer.single = single;
    CommonTokenStream tokens = this.tokens = new TokenRewriteStream(lexer);
    PythonTokenSource indentedSource = new PythonTokenSource(tokens, filename, single);
    tokens = new CommonTokenStream(indentedSource);
    PythonParser parser = new PythonParser(tokens, encoding);
    parser.setErrorHandler(weh);
    parser.setTreeAdaptor(new AnalyzerTreeAdaptor());
    return parser;
  }
}
