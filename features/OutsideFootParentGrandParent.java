package features;

import java.util.Stack;

import edu.berkeley.nlp.syntax.Tree;
import interfaces.OutsideFeature;

public class OutsideFootParentGrandParent implements OutsideFeature {

	/**
	 * For description see the interface description
	 */
	@Override
	public String getFeature(Stack<Tree<String>> foottoroot) {
		/*
		 * Extract the feature TODO
		 */
		String feature = null;
		if (foottoroot.size() >= 3) {
			Tree<String> footTree = foottoroot.pop();
			Tree<String> parentTree = foottoroot.pop();
			Tree<String> grandparentTree = foottoroot.pop();

			/*
			 * TODO
			 */
			feature = footTree.getLabel() + "," + parentTree.getLabel() + "," + grandparentTree.getLabel();

			/*
			 * TODO
			 */
			foottoroot.push(grandparentTree);
			foottoroot.push(parentTree);
			foottoroot.push(footTree);
		} else {
			/*
			 * TODO
			 */
			feature = "NOTVALID";
		}
		return feature;
	}

}
