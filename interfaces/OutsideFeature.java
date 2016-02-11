package interfaces;

import java.util.Stack;

import edu.berkeley.nlp.syntax.Tree;

public interface OutsideFeature {

	public String getFeature(Stack<Tree<String>> foottoroot);

}
