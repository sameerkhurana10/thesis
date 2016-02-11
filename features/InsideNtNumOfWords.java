package features;

import edu.berkeley.nlp.syntax.Tree;
import interfaces.InsideFeature;

public class InsideNtNumOfWords implements InsideFeature {

	public static int length;

	/**
	 * The function extracts the feature of the form (a,num). For more
	 * information see the interface documentation
	 */
	@Override
	public String getFeature(Tree<String> insideTree, boolean isPreterminal) {

		return (insideTree.getLabel() + "," + Integer.toString(length));
	}

}
