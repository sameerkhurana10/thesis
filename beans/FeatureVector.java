package beans;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.Stack;

import dictionary.Alphabet;
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
	private int vectorDimensions;
	private long treeIdx;
	private Alphabet sampleDictionary;
	private Alphabet fullDictionary;
	private LinkedList<String> featureList;

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

	public long getTreeIdx() {
		return treeIdx;
	}

	public void setTreeIdx(int treeFileIdx) {
		this.treeIdx = treeFileIdx;
	}

	public int getVectorDimensions() {
		return vectorDimensions;
	}

	public void setVectorDimensions(int vectorDimensions) {
		this.vectorDimensions = vectorDimensions;
	}

	public Alphabet getSampleDictionary() {
		return sampleDictionary;
	}

	public void setSampleDictionary(Alphabet sampleDictionary) {
		this.sampleDictionary = sampleDictionary;
	}

	public Alphabet getFullDictionary() {
		return fullDictionary;
	}

	public void setFullDictionary(Alphabet fullDictionary) {
		this.fullDictionary = fullDictionary;
	}

	public LinkedList<String> getFeatureList() {
		return featureList;
	}

	public void setFeatureList(LinkedList<String> featureList) {
		this.featureList = featureList;
	}

}
