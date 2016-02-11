package interfaces;

import edu.berkeley.nlp.syntax.Tree;

public interface InsideFeature {

	String getFeature(Tree<String> insideTree, boolean isPreterminal);

}
