package features;

import java.util.Stack;

import edu.berkeley.nlp.syntax.Tree;
import interfaces.OutsideFeature;
import utils.AbstractHeadFinder;
import utils.PennTreebankCollinsHeadFinder;

public class OutsideOtherheadposAbove implements OutsideFeature {

	/**
	 * For description see the interface description TODO
	 */
	@Override
	public String getFeature(Stack<Tree<String>> foottoroot) {
		/*
		 * TODO
		 */
		AbstractHeadFinder headfinder = new PennTreebankCollinsHeadFinder();
		Stack<Tree<String>> tempstack = new Stack<Tree<String>>();

		Tree<String> footTree = foottoroot.pop();
		tempstack.push(footTree);
		String footpos = headfinder.getHeadPartOfSpeech(footTree);

		String feature = "NOTVALID";

		// Pop and check
		while (!foottoroot.empty()) {
			Tree<String> parentTree = foottoroot.pop();
			tempstack.push(parentTree);

			String headpos = headfinder.getHeadPartOfSpeech(parentTree);
			if (!headpos.equals(footpos)) {
				feature = headpos;
				break;
			}
		}

		// Push back
		while (!tempstack.empty()) {
			Tree<String> item = tempstack.pop();
			foottoroot.push(item);
		}

		return feature;
	}

}
