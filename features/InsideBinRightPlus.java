package features;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import interfaces.InsideFeature;
import utils.CommonUtil;

public class InsideBinRightPlus implements InsideFeature {

	/**
	 * The feature that this method is extracting is of the form a -> b (c -> .)
	 */
	@Override
	public String getFeature(Tree<String> insideTree, boolean isPreterminal) {
		if (!isPreterminal) {
			/*
			 * Getting the trees of the child nodes
			 */
			List<Tree<String>> children = insideTree.getChildren();
			/*
			 * Checking the list. It should have the required size
			 */
			if ((!children.isEmpty()) && children.size() > 1) {
				Tree<String> right = children.get(1);
				/*
				 * Getting the required feature string from the right tree c ->
				 * .
				 */
				String rightstr = CommonUtil.getTreeString(right);
				return (insideTree.getLabel() + "->" + children.get(0).getLabel() + ",(" + rightstr + ")");
			}
		}
		/*
		 * The feature might not exist at all
		 */
		return "NOTVALID";
	}

}
