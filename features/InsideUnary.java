package features;

import edu.berkeley.nlp.syntax.Tree;
import interfaces.InsideFeature;

public class InsideUnary implements InsideFeature {

	@Override
	public String getFeature(Tree<String> insideTree, boolean isPreterminal) {
		if (isPreterminal && !insideTree.getChildren().isEmpty()) {
			return (insideTree.getLabel() + "->" + insideTree.getChildren().get(0).getLabel().toLowerCase());
		} else {
			return "NOTVALID";
		}
	}

}
