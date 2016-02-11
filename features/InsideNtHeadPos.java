package features;

import edu.berkeley.nlp.syntax.Tree;
import interfaces.InsideFeature;
import utils.AbstractHeadFinder;
import utils.PennTreebankCollinsHeadFinder;

public class InsideNtHeadPos implements InsideFeature {

	/**
	 * The method extracts the feature (a, pos)
	 */
	@Override
	public String getFeature(Tree<String> insideTree, boolean isPreterminal) {
		/*
		 * Using the Collins head finder
		 */
		AbstractHeadFinder headfinder = new PennTreebankCollinsHeadFinder();
		/*
		 * Getting the required POS tag only if the inside tree is not a leaf
		 */
		if (!insideTree.isLeaf()) {
			String headpos = headfinder.getHeadPartOfSpeech(insideTree);
			return (insideTree.getLabel() + "," + headpos);
		} else {
			return "NOTVALID";
		}
	}

}
