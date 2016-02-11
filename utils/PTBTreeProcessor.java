package utils;

import edu.berkeley.nlp.syntax.Tree;

public abstract class PTBTreeProcessor {
	public abstract Tree<String> process(Tree<String> tree);
}
