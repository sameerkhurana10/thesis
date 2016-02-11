package features;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import interfaces.InsideFeature;
import utils.CommonUtil;

public class InsideBinLeftPlus implements InsideFeature {

	/**
	 * The feature that we are trying to extract here is of the form a -> (b ->
	 * d,e), c
	 */
	@Override
	public String getFeature(Tree<String> insideTree, boolean isPreterminal) {
		if (!isPreterminal) {
			/*
			 * Getting the trees of all the child nodes of the node under
			 * scrutiny
			 */
			List<Tree<String>> children = insideTree.getChildren();

			/*
			 * Check the size of the children tree to make sure that the feature
			 * that we are trying to extract exists in the inside tree
			 */
			if ((!children.isEmpty()) && children.size() > 1) {
				Tree<String> left = children.get(0);
				/*
				 * Getting the left String i.e. the a->(b->d,e),c
				 */
				String leftString = CommonUtil.getTreeString(left);

				if (leftString != null) {
					return (insideTree.getLabel() + "->(" + leftString + ")," + children.get(1).getLabel());
				}
			}
		}
		return "NOTVALID";
	}

}
