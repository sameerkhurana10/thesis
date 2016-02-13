package features;

import java.util.Stack;

import edu.berkeley.nlp.syntax.Tree;
import interfaces.OutsideFeature;
import superclass.VSMThesis;
import utils.CommonUtil;

public class OutsideTreeabove1 extends VSMThesis implements OutsideFeature {

	/**
	 * For description see the interface description
	 */
	@Override
	public String getFeature(Stack<Tree<String>> foottoroot) {

		/*
		 * 
		 * extracting the feature TODO. The code is written by Dr Shay Cohen
		 */
		String feature = null;
		if (foottoroot.size() >= 2) {
			Tree<String> footTree = foottoroot.pop();
			Tree<String> parentTree = foottoroot.pop();

			/*
			 * TODO
			 */
			feature = CommonUtil.getStringFromParent(parentTree, footTree);

			/*
			 * Putting them back TODO
			 */
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
