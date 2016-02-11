package utils;

import java.util.ArrayList;

import edu.berkeley.nlp.PCFGLA.Binarization;
import edu.berkeley.nlp.PCFGLA.TreeAnnotations;
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.syntax.Trees;

public class PTBTreeNormaliser extends PTBTreeProcessor {
	private int markovizationH;
	private int markovizationV;
	private Binarization binarization;

	private boolean isNormalize;

	public static boolean keepUnary = false;
	public static boolean extraOutput = true;

	public PTBTreeNormaliser(boolean isNormalize) {
		this.markovizationH = 0;
		this.markovizationV = 1;
		this.binarization = Binarization.RIGHT;
		this.isNormalize = isNormalize;
	}

	@Override
	public Tree<String> process(Tree<String> tree) {
		Tree<String> normalizedTree;

		if (isNormalize) {

			// System.err.println(tree.toString());

			Trees.TreeTransformer<String> treeTransformer = new Trees.StandardTreeNormalizer();
			tree = treeTransformer.transformTree(tree);

			// System.err.println(tree.toString());

			//normalizedTree = TreeAnnotations.processTree(tree, markovizationV,
				//	markovizationH, binarization, false);

			// System.err.println(normalizedTree.toString());

			//collapseUnaryRules(normalizedTree);

			// System.err.println(normalizedTree.toString());

			normalizedTree = removeTOPBracketing(tree);

			// System.err.println(normalizedTree.toString());

		} else {
			normalizedTree = tree;
		}
		return normalizedTree;
	}

	// (TOP (S (NP () ()) (...))) -> (S (NP () ()) (...))
	private Tree<String> removeTOPBracketing(Tree<String> tree) {
		if (tree.getLabel().equals("TOP") || tree.getLabel().equals("ROOT")
				|| tree.getLabel().equals("S1")) {
			Tree<String> finalTree = tree.getChildren().get(0);
			return finalTree;
		}
		return tree;
	}

	// Note: If you have a tree like (S (NP (NN dog)) (...))
	// then you will get a tree like (S (NP|NN dog) (...))
	// which is a bit strange (i.e. collapsing unary rules up to POS tags)
	// but seems necessary if we are interested in removing unary rules
	public static void collapseUnaryRules(Tree<String> tree) {
		boolean changed = false;

		if (tree.isPreTerminal())
			return;

		if (keepUnary) {
			if ((tree.getChildren().size() == 1)
					&& ((tree.getChildren().get(0).isPreTerminal()))) {
				return;
			}
		}

		ArrayList<Tree<String>> newChildren = new ArrayList<Tree<String>>();
		for (int i = 0; i < tree.getChildren().size(); i++) {
			if ((tree.getChildren().get(i).getChildren().size() == 1)
					&& ((!keepUnary) || (!tree.getChildren().get(i)
							.getChildren().get(0).isPreTerminal()))
					&& ((!(tree.getChildren().get(i).isPreTerminal())))) {
				newChildren.add(tree.getChildren().get(i).getChildren().get(0));
				tree.getChildren()
						.get(i)
						.getChildren()
						.get(0)
						.setLabel(
								tree.getChildren().get(i).getLabel()
										+ "|"
										+ tree.getChildren().get(i)
												.getChildren().get(0)
												.getLabel());
				changed = true;
			} else {
				newChildren.add(tree.getChildren().get(i));
			}
		}

		tree.setChildren(newChildren);

		if (changed) {
			collapseUnaryRules(tree);
		} else {
			for (Tree<String> child : tree.getChildren()) {
				collapseUnaryRules(child);
			}
		}
	}
}
