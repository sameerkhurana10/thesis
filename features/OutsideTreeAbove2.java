package features;

import java.util.Stack;

import edu.berkeley.nlp.syntax.Tree;
import interfaces.OutsideFeature;
import superclass.VSMThesis;
import utils.CommonUtil;

public class OutsideTreeAbove2 extends VSMThesis implements OutsideFeature {

	/**
	 * For description see the interface description
	 */
	@Override
	public String getFeature(Stack<Tree<String>> foottoroot) {

		/*
		 * Extracting the feature TODO The piece of code is implemented by Dr
		 * Shay Cohen
		 */
		String feature = null;
		if (foottoroot.size() >= 3) {
			Tree<String> footTree = foottoroot.pop();
			Tree<String> parentTree = foottoroot.pop();
			Tree<String> grandparentTree = foottoroot.pop();

			/*
			 * The function that extracts the feature The function is developed
			 * by Dr Shay Cohen
			 */
			feature = CommonUtil.getStringFromGrandparent(grandparentTree, parentTree, footTree);

			/*
			 * Putting them back TODO
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
