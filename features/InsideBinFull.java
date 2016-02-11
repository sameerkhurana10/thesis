package features;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import interfaces.InsideFeature;

public class InsideBinFull implements InsideFeature {

	/**
	 * The method extracts the feature a -> b c
	 */
	@Override
	public String getFeature(Tree<String> insideTree, boolean isPreterminal) {
		/*
		 * You can extract a -> b c only if the node a is not a pre-terminal of
		 * course
		 */
		if (!isPreterminal) {
			/*
			 * Getting the children trees i.e. the node trees for the children
			 * nodes of the node under scrutiny
			 */
			List<Tree<String>> children = insideTree.getChildren();
			/*
			 * The if condition is added because the code was throwing error for
			 * nodes that do not have long enough inside trees
			 */
			if (children.size() > 1) {
				return (insideTree.getLabel() + "->" + children.get(0).getLabel() + "," + children.get(1).getLabel());
			}
		}
		/*
		 * If the feature does not exist for a particular node then return
		 * NOTVALID and hence nothing would be added to the the dictionary in
		 * this case
		 */
		return "NOTVALID";
	}

}
