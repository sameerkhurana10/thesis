package features;

import java.util.Stack;

import edu.berkeley.nlp.syntax.Tree;
import interfaces.OutsideFeature;
import utils.CommonUtil;

public class OutsideTreeAbove3 implements OutsideFeature {

	/**
	 * For description see the interface description
	 */
	@Override
	public String getFeature(Stack<Tree<String>> foottoroot) {

		/*
		 * Getting the feature TODO
		 */
		String feature = null;
		if (foottoroot.size() >= 4) {
			Tree<String> footTree = foottoroot.pop();
			Tree<String> parentTree = foottoroot.pop();
			Tree<String> grandparentTree = foottoroot.pop();
			Tree<String> greatgrandparentTree = foottoroot.pop();

			/*
			 * The function written by Dr Shay Cohen
			 */
			feature = CommonUtil.getStringFromGreatgrandparent(greatgrandparentTree, grandparentTree, parentTree,
					footTree);

			// Putting them back
			foottoroot.push(greatgrandparentTree);
			foottoroot.push(grandparentTree);
			foottoroot.push(parentTree);
			foottoroot.push(footTree);
		} else {
			feature = "NOTVALID";
		}
		return feature;
	}

}
