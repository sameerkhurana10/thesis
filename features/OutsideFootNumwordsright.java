package features;

import java.util.Stack;

import edu.berkeley.nlp.syntax.Tree;
import interfaces.OutsideFeature;

public class OutsideFootNumwordsright implements OutsideFeature {

	public static int outsideWordsRight;

	/**
	 * For description see the interface description
	 */
	@Override
	public String getFeature(Stack<Tree<String>> foottoroot) {
		/*
		 * Getting the top element of the stack without removing it. This
		 * statement gives us the inside tree or the foottree as we call it
		 */
		Tree<String> footTree = foottoroot.peek();
		String footlabel = footTree.getLabel();

		/*
		 * We can access the static variable in a non-static method
		 */
		return (footlabel + "," + outsideWordsRight);
	}

}
