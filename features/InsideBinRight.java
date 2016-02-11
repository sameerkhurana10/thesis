package features;

import java.util.List;

import edu.berkeley.nlp.syntax.Tree;
import interfaces.InsideFeature;

/**
 * This is the InsideBinFull feature object class. The object of this class
 * holds a dictionary that stores the features of the type a -> b c. The object
 * of this class is also used to call the method that extracts the feature, a ->
 * c
 * 
 * @author sameerkhurana10
 *
 */
public class InsideBinRight implements InsideFeature {

	@Override
	public String getFeature(Tree<String> insideTree, boolean isPreterminal) {
		/*
		 * If the inside tree is a pre-terminal then we cannot extract the
		 * desired feature
		 */
		if (!isPreterminal) {
			/*
			 * Getting all the node trees for all the child nodes of the node
			 * under scrutiny
			 */
			List<Tree<String>> children = insideTree.getChildren();
			/*
			 * Checking the size of the list here, because the elements that we
			 * are trying to extract must exist. The earlier code did not have
			 * this condition and hence this line was throwing error
			 */
			if ((!children.isEmpty()) && children.size() > 1) {
				return (insideTree.getLabel() + "->" + children.get(1).getLabel());
			}
		}
		/*
		 * Return the below string if the feature does not exist
		 */
		return "NOTVALID";
	}

}
