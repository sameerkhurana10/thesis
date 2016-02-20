package beans;

import java.io.Serializable;
import java.util.Stack;

import edu.berkeley.nlp.syntax.Tree;
import utils.VSMSparseVector;

public class FeatureVector implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private Tree<String> insideTree;
	private Tree<String> syntaxTree;
	private Stack<Tree<String>> footToRoot;
	private VSMSparseVector featureVec;
	private String treeFileName;
	private int treeFileIdx;

	public Tree<String> getInsideTree() {
		return insideTree;
	}

	public void setInsideTree(Tree<String> insideTree) {
		this.insideTree = insideTree;
	}

	public Tree<String> getSyntaxTree() {
		return syntaxTree;
	}

	public void setSyntaxTree(Tree<String> syntaxTree) {
		this.syntaxTree = syntaxTree;
	}

	public Stack<Tree<String>> getFootToRoot() {
		return footToRoot;
	}

	public void setFootToRoot(Stack<Tree<String>> footToRoot) {
		this.footToRoot = footToRoot;
	}

	public VSMSparseVector getFeatureVec() {
		return featureVec;
	}

	public void setFeatureVec(VSMSparseVector featureVec) {
		this.featureVec = featureVec;
	}

	public String getTreeFileName() {
		return treeFileName;
	}

	public void setTreeFileName(String treeFileName) {
		this.treeFileName = treeFileName;
	}

	public int getTreeFileIdx() {
		return treeFileIdx;
	}

	public void setTreeFileIdx(int treeFileIdx) {
		this.treeFileIdx = treeFileIdx;
	}

}
