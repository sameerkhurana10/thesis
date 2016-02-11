package features;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import interfaces.InsideFeature;

public class InsideBinLeft implements InsideFeature {

	/**
	 * Extracts the feature of the type
	 */
	@Override
	public String getFeature(Tree<String> insideTree, boolean isPreterminal) {
		/*
		 * Feature of the form a->b can be extracted only if the node is not a
		 * pre-terminal
		 */
		if (!isPreterminal) {
			/*
			 * Getting the trees of all the child nodes of the node under
			 * scrutiny
			 */
			List<Tree<String>> children = insideTree.getChildren();
			/*
			 * The list should not be empty
			 */
			if (!(children.isEmpty())) {
				return (insideTree.getLabel() + "->" + children.get(0).getLabel());
			}
		}
		/*
		 * If the feature does not exist then return the below String
		 */
		return "NOTVALID";
	}

}
