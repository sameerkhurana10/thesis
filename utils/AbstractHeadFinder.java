package utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;
import java.util.concurrent.ConcurrentHashMap;

import edu.berkeley.nlp.syntax.Tree;

public class AbstractHeadFinder {
	private static final boolean DEBUG = false;
	// protected final TreebankLanguagePack tlp;
	protected ConcurrentHashMap<String, String[][]> nonTerminalInfo;

	/**
	 * Default direction if no rule is found for category. Subclasses can turn
	 * it on if they like. If they don't it is an error if no rule is defined
	 * for a category (null is returned).
	 */
	protected String[] defaultRule; // = null;

	/**
	 * These are built automatically from categoriesToAvoid and used in a fairly
	 * different fashion from defaultRule (above). These are used for categories
	 * that do have defined rules but where none of them have matched. Rather
	 * than picking the rightmost or leftmost child, we will use these to pick
	 * the the rightmost or leftmost child which isn't in categoriesToAvoid.
	 */
	private String[] defaultLeftRule;
	private String[] defaultRightRule;

	// This is taken from TreeAnnotations of the Berkeley parser
	public static String transformLabelIsLeaf(String treeLabel) {
		String transformedLabel = treeLabel;
		int cutIndex = transformedLabel.indexOf('-');
		int cutIndex2 = transformedLabel.indexOf('=');
		final int cutIndex3 = transformedLabel.indexOf('^');
		if (cutIndex3 > 0 && (cutIndex3 < cutIndex2 || cutIndex2 == -1))
			cutIndex2 = cutIndex3;
		if (cutIndex2 > 0 && (cutIndex2 < cutIndex || cutIndex <= 0))
			cutIndex = cutIndex2;
		if (cutIndex > 0 && true /* !tree.isLeaf() */) {
			transformedLabel = new String(transformedLabel.substring(0,
					cutIndex));
		}

		// correct for unspliced nodes (different than Berkely parser), added by
		// Shay
		if (transformedLabel.startsWith("@")) {
			transformedLabel = new String(transformedLabel.substring(1));
		}

		return transformedLabel;
	}

	public static String basicTag(String tag) {
		return transformLabelIsLeaf(stripUnaryChain(tag));
	}

	public static String stripUnaryChain(String treeLabel) {
		String pos = "";
		StringTokenizer tokenizer = new StringTokenizer(treeLabel, "|");

		while (tokenizer.hasMoreElements()) {
			pos = tokenizer.nextToken();
		}

		return pos;
	}

	// This is taken from TreeAnnotations of the Berkeley parser
	public static String transformLabel(Tree<String> tree) {
		String transformedLabel = tree.getLabel();
		int cutIndex = transformedLabel.indexOf('-');
		int cutIndex2 = transformedLabel.indexOf('=');
		final int cutIndex3 = transformedLabel.indexOf('^');
		if (cutIndex3 > 0 && (cutIndex3 < cutIndex2 || cutIndex2 == -1))
			cutIndex2 = cutIndex3;
		if (cutIndex2 > 0 && (cutIndex2 < cutIndex || cutIndex <= 0))
			cutIndex = cutIndex2;
		if (cutIndex > 0 && !tree.isLeaf()) {
			transformedLabel = new String(transformedLabel.substring(0,
					cutIndex));
		}

		// correct for unspliced nodes (different than Berkely parser), added by
		// Shay
		if (transformedLabel.startsWith("@")) {
			transformedLabel = new String(transformedLabel.substring(1));
		}

		return transformedLabel;
	}

	/**
	 * Set categories which, if it comes to last resort processing (i.e. none of
	 * the rules matched), will be avoided as heads. In last resort processing,
	 * it will attempt to match the leftmost or rightmost constituent not in
	 * this set but will fall back to the left or rightmost constituent if
	 * necessary.
	 * 
	 * @param categoriesToAvoid
	 *            list of constituent types to avoid
	 */
	protected void setCategoriesToAvoid(String[] categoriesToAvoid) {
		// automatically build defaultLeftRule, defaultRightRule
		ArrayList<String> asList = new ArrayList<String>(
				Arrays.asList(categoriesToAvoid));
		asList.add(0, "leftexcept");
		defaultLeftRule = new String[asList.size()];
		defaultRightRule = new String[asList.size()];
		asList.toArray(defaultLeftRule);
		asList.set(0, "rightexcept");
		asList.toArray(defaultRightRule);
	}

	/**
	 * A way for subclasses for corpora with explicit head markings to return
	 * the explicitly marked head
	 * 
	 * @param t
	 *            a tree to find the head of
	 * @return the marked head-- null if no marked head
	 */
	// to be overridden in subclasses for corpora
	//
	protected Tree<String> findMarkedHead(Tree<String> t) {
		return null;
	}

	// public Tree<String> determineHeadAbove(ExtendedTree<String> t,
	// ExtendedTree<String> tHead)
	// {
	// ExtendedTree<String> parent = t.getParent();
	// Tree<String> head = null;
	//
	// while (parent != null)
	// {
	// head = determinePercolatedHead(parent);
	//
	// if (head != tHead) {
	// return head;
	// }
	//
	// if (head == null) { break; }
	//
	// parent = parent.getParent();
	// }
	//
	// return null;
	// }

	/**
	 * Determine which daughter of the current parse tree is the head.
	 * 
	 * @param t
	 *            The parse tree to examine the daughters of. If this is a leaf,
	 *            <code>null</code> is returned
	 * @return The daughter parse tree that is the head of <code>t</code>
	 * @see Tree#percolateHeads(HeadFinder) for a routine to call this and
	 *      spread heads throughout a tree
	 */
	public Tree<String> determineHead(Tree<String> t) {
		return determineHead(t, null);
	}

	public Tree<String> determinePercolatedHead(Tree<String> tree) {
		Tree<String> head = tree;

		while ((head != null) && (!head.isPreTerminal())) {
			head = determineHead(head);
		}
		return head;
	}

	/**
	 * Determine which daughter of the current parse tree is the head.
	 * 
	 * @param t
	 *            The parse tree to examine the daughters of. If this is a leaf,
	 *            <code>null</code> is returned
	 * @param parent
	 *            The parent of t
	 * @return The daughter parse tree that is the head of <code>t</code>.
	 *         Returns null for leaf nodes.
	 * @see Tree#percolateHeads(HeadFinder) for a routine to call this and
	 *      spread heads throughout a tree
	 */
	public Tree<String> determineHead(Tree<String> t, Tree<String> parent) {
		if (nonTerminalInfo == null) {
			throw new RuntimeException(
					"Classes derived from AbstractCollinsHeadFinder must"
							+ " create and fill ConcurrentHashMap nonTerminalInfo.");
		}
		if (t == null || t.isLeaf()) {
			return null;
		}

		if (t.isPreTerminal()) {
			return t;
		}

		if (DEBUG) {
			System.err.println("determineHead for " + t);
		}

		List<Tree<String>> kids = t.getChildren();

		Tree<String> theHead;
		// first check if subclass found explicitly marked head
		if ((theHead = findMarkedHead(t)) != null) {
			if (DEBUG) {
				System.err.println("Find marked head method returned "
						+ theHead.getLabel() + " as head of " + t.getLabel());
			}
			return theHead;
		}

		// if the node is a unary, then that kid must be the head
		// it used to special case preterminal and ROOT/TOP case
		// but that seemed bad (especially hardcoding string "ROOT")
		if (kids.size() == 1) {
			if (DEBUG) {
				System.err.println("Only one child determines "
						+ kids.get(0).getLabel() + " as head of "
						+ t.getLabel());
			}
			return kids.get(0);
		}

		return determineNonTrivialHead(t, parent);
	}

	/**
	 * Called by determineHead and may be overridden in subclasses if special
	 * treatment is necessary for particular categories.
	 */
	protected Tree<String> determineNonTrivialHead(Tree<String> t,
			Tree<String> parent) {
		Tree<String> theHead = null;
		// String motherCat = tlp.basicCategory(t.label().value());
		String motherCat = transformLabel(t);

		if (DEBUG) {
			System.err.println("Looking for head of " + t.getLabel()
					+ "; value is |" + t.getLabel() + "|, " + " baseCat is |"
					+ motherCat + '|');
		}
		// We know we have nonterminals underneath
		// (a bit of a Penn Treebank assumption, but).

		// Look at label.
		// a total special case....
		// first look for POS tag at end
		// this appears to be redundant in the Collins case since the rule
		// already would do that
		// Tree lastDtr = t.lastChild();
		// if (tlp.basicCategory(lastDtr.label().value()).equals("POS")) {
		// theHead = lastDtr;
		// } else {
		String[][] how = nonTerminalInfo.get(motherCat);
		if (how == null) {
			if (DEBUG) {
				System.err.println("Warning: No rule found for " + motherCat
						+ " (first char: " + motherCat.charAt(0) + ')');
				System.err.println("Known nonterms are: "
						+ nonTerminalInfo.keySet());
			}
			if (defaultRule != null) {
				if (DEBUG) {
					System.err.println("  Using defaultRule");
				}
				return traverseLocate(t.getChildren(), defaultRule, true);
			} else {
				return null;
			}
		}
		for (int i = 0; i < how.length; i++) {
			boolean lastResort = (i == how.length - 1);
			theHead = traverseLocate(t.getChildren(), how[i], lastResort);
			if (theHead != null) {
				break;
			}
		}
		if (DEBUG) {
			System.err.println("  Chose " + theHead.getLabel());
		}
		return theHead;
	}

	/**
	 * Attempt to locate head daughter tree from among daughters. Go through
	 * daughterTrees looking for things from a set found by looking up the
	 * motherkey specifier in a hash map, and if you do not find one, take
	 * leftmost or rightmost thing iff lastResort is true, otherwise return
	 * <code>null</code>.
	 */
	protected Tree<String> traverseLocate(List<Tree<String>> daughterTrees,
			String[] how, boolean lastResort) {
		int headIdx = 0;
		String childCat;
		boolean found = false;

		if (how[0].equals("left")) {
			twoloop: for (int i = 1; i < how.length; i++) {
				for (headIdx = 0; headIdx < daughterTrees.size(); headIdx++) {
					// childCat =
					// tlp.basicCategory(daughterTrees.get(headIdx).getLabel());
					childCat = transformLabel(daughterTrees.get(headIdx));
					if (how[i].equals(childCat)) {
						found = true;
						break twoloop;
					}
				}
			}
		} else if (how[0].equals("leftdis")) {
			twoloop: for (headIdx = 0; headIdx < daughterTrees.size(); headIdx++) {
				// childCat = tlp.basicCategory(daughterTrees[headIdx].label()
				// .value());
				childCat = transformLabel(daughterTrees.get(headIdx));
				for (int i = 1; i < how.length; i++) {
					if (how[i].equals(childCat)) {
						found = true;
						break twoloop;
					}
				}
			}
		} else if (how[0].equals("right")) {
			// from right
			twoloop: for (int i = 1; i < how.length; i++) {
				for (headIdx = daughterTrees.size() - 1; headIdx >= 0; headIdx--) {
					childCat = transformLabel(daughterTrees.get(headIdx));
					if (how[i].equals(childCat)) {
						found = true;
						break twoloop;
					}
				}
			}
		} else if (how[0].equals("rightdis")) {
			// from right, but search for any, not in turn
			twoloop: for (headIdx = daughterTrees.size() - 1; headIdx >= 0; headIdx--) {
				// childCat = tlp.basicCategory(daughterTrees[headIdx].label()
				// .value());
				childCat = transformLabel(daughterTrees.get(headIdx));
				for (int i = 1; i < how.length; i++) {
					if (how[i].equals(childCat)) {
						found = true;
						break twoloop;
					}
				}
			}
		} else if (how[0].equals("leftexcept")) {
			for (headIdx = 0; headIdx < daughterTrees.size(); headIdx++) {
				// childCat = tlp.basicCategory(daughterTrees[headIdx].label()
				// .value());
				childCat = transformLabel(daughterTrees.get(headIdx));
				found = true;
				for (int i = 1; i < how.length; i++) {
					if (how[i].equals(childCat)) {
						found = false;
					}
				}
				if (found) {
					break;
				}
			}
		} else if (how[0].equals("rightexcept")) {
			for (headIdx = daughterTrees.size() - 1; headIdx >= 0; headIdx--) {
				// childCat = tlp.basicCategory(daughterTrees[headIdx].label()
				// .value());
				childCat = transformLabel(daughterTrees.get(headIdx));
				found = true;
				for (int i = 1; i < how.length; i++) {
					if (how[i].equals(childCat)) {
						found = false;
					}
				}
				if (found) {
					break;
				}
			}
		} else {
			throw new RuntimeException("ERROR: invalid direction type "
					+ how[0]
					+ " to nonTerminalInfo map in AbstractCollinsHeadFinder.");
		}

		// what happens if our rule didn't match anything
		if (!found) {
			if (lastResort) {
				// use the default rule to try to match anything except
				// categoriesToAvoid
				// if that doesn't match, we'll return the left or rightmost
				// child (by
				// setting headIdx). We want to be careful to ensure that
				// postOperationFix
				// runs exactly once.
				String[] rule;
				if (how[0].startsWith("left")) {
					headIdx = 0;
					rule = defaultLeftRule;
				} else {
					headIdx = daughterTrees.size() - 1;
					rule = defaultRightRule;
				}
				Tree<String> child = (Tree<String>) traverseLocate(
						daughterTrees, rule, false);
				if (child != null) {
					return child;
				}
			} else {
				// if we're not the last resort, we can return null to let the
				// next rule try to match
				return null;
			}
		}

		headIdx = postOperationFix(headIdx, daughterTrees);

		return daughterTrees.get(headIdx);
	}

	/**
	 * A way for subclasses to fix any heads under special conditions The
	 * default does nothing.
	 * 
	 * @param headIdx
	 *            the index of the proposed head
	 * @param daughterTrees
	 *            the array of daughter trees
	 * @return the new headIndex
	 */
	protected int postOperationFix(int headIdx, List<Tree<String>> daughterTrees) {
		if (headIdx >= 2) {
			// String prevLab = tlp.basicCategory(daughterTrees[headIdx -
			// 1].value());
			String prevLab = transformLabel(daughterTrees.get(headIdx - 1));
			if (prevLab.equals("CC") || prevLab.equals("CONJP")) {
				int newHeadIdx = headIdx - 2;
				Tree<String> t = daughterTrees.get(newHeadIdx);
				while (newHeadIdx >= 0 && t.isPreTerminal()
						&& isPunctuationTag(t.getLabel())) {
					newHeadIdx--;
				}
				if (newHeadIdx >= 0) {
					headIdx = newHeadIdx;
				}
			}
		}
		return headIdx;
	}

	public boolean isPunctuationTag(String label) {
		System.err.println("Warning: undefined punctuation tag identifier.");

		return false;
	}

	public String getHeadPartOfSpeech(Tree<String> tree) {
		String headPOS = null;
		while (true) {
			if (tree.isPreTerminal()) {
				headPOS = tree.getLabel();
				break;
			}

			tree = determineHead(tree);
		}
		return headPOS;
	}

}
