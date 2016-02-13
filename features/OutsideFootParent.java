package features;

import java.util.Stack;

import edu.berkeley.nlp.syntax.Tree;
import interfaces.OutsideFeature;
import superclass.VSMThesis;

public class OutsideFootParent extends VSMThesis implements OutsideFeature {

	/**
	 * For description see the interface description
	 */
	@Override
	public String getFeature(Stack<Tree<String>> foottoroot) {
		/*
		 * Extracting the outside feature TODO
		 */
		String feature = null;
		if (foottoroot.size() >= 2) {
			/*
			 * TODO
			 */
			Tree<String> footTree = foottoroot.pop();
			Tree<String> parentTree = foottoroot.pop();

			feature = footTree.getLabel() + "," + parentTree.getLabel();

			// Putting them back
			foottoroot.push(parentTree);
			foottoroot.push(footTree);
		} else {
			feature = "NOTVALID";
		}
		return feature;
	}

}
