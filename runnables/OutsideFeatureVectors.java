package runnables;

import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Stack;

import org.apache.log4j.Logger;

import beans.FeatureVector;
import dictionary.Alphabet;
import edu.berkeley.nlp.syntax.Constituent;
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.syntax.Trees.PennTreeReader;
import interfaces.InsideFeature;
import interfaces.OutsideFeature;
import jeigen.SparseMatrixLil;
import main.FeatureDictionary;
import utils.CommonUtil;
import utils.VSMSparseVector;

public class OutsideFeatureVectors implements Runnable {

	String featureVectorsStoragePath;
	Alphabet dictionary;
	InsideFeature featureObject;
	Logger logger;
	String nonTerminal;
	int dprime;
	Tree<String> insideTree;
	String treeFile;
	boolean isPreTerminal;
	LinkedList<FeatureVector> sparseVectors;
	String dictionaryPath;

	int M;
	String featureDictionaries;
	LinkedList<OutsideFeature> features;
	Alphabet sampleDictionary;
	int k;
	SparseMatrixLil Psi;

	public OutsideFeatureVectors(String treeFile, String featureVectorsStoragePath, Logger logger, String nonTerminal,
			int M, String featureDictionaries, LinkedList<OutsideFeature> features, int k) {
		this.treeFile = treeFile;
		this.featureVectorsStoragePath = featureVectorsStoragePath;
		this.logger = logger;
		this.M = M;
		this.featureDictionaries = featureDictionaries;
		this.features = features;
		this.nonTerminal = nonTerminal;
		sparseVectors = new LinkedList<FeatureVector>();
		this.k = k;
		sampleDictionary = new Alphabet();
		sampleDictionary.allowGrowth();
		sampleDictionary.turnOnCounts();
		Psi = new SparseMatrixLil(dprime, M);
	}

	@Override
	public void run() {

		System.out.println("MEMORY USED: " + Runtime.getRuntime().maxMemory());

		dictionaryInit();
		PennTreeReader treeParser = null;
		try {
			treeParser = CommonUtil.getTreeReaderBz(treeFile);

		} catch (Exception e) {
			e.printStackTrace();
		}
		int i = 0;
		int size = 0;
		mainloop: while (treeParser.hasNext()) {
			i++;
			System.out.println(i);
			Tree<String> tree = FeatureDictionary.getNormalizedTree(treeParser.next());
			Iterator<Tree<String>> treeNodeItr = null;
			Map<Tree<String>, Constituent<String>> constituentsMap = null;

			if (tree != null) {
				treeNodeItr = FeatureDictionary.getTreeNodeIterator(tree);
				constituentsMap = tree.getConstituents();

			}

			while (treeNodeItr != null && treeNodeItr.hasNext()) {

				// Function psi follows

				insideTree = treeNodeItr.next();
				isPreTerminal = FeatureDictionary.checkIsPreterminal(insideTree);
				Stack<Tree<String>> footToRoot = new Stack<Tree<String>>();
				CommonUtil.updateFoottorootPath(footToRoot, tree, insideTree, constituentsMap);

				CommonUtil.getNumberOfOutsideWordsLeft(insideTree, constituentsMap, tree);
				CommonUtil.getNumberOfOutsideWordsRight(insideTree, constituentsMap, tree);

				if (!insideTree.isLeaf() && insideTree.getLabel().equalsIgnoreCase(nonTerminal)) {

					FeatureVector bean = new FeatureVector();
					VSMSparseVector psi = new VSMSparseVector(dprime);
					LinkedList<String> featureList = new LinkedList<String>();

					for (OutsideFeature outsideFeatureObj : features) {
						String feature = outsideFeatureObj.getFeature(footToRoot);
						if (!feature.equalsIgnoreCase("NOTVALID")) {
							int featureIndex = dictionary.lookupIndex(feature);
							if (featureIndex != -1) {
								psi.add(featureIndex, 1.0);
							} else {
								int notFrequentFeatureIndex = dictionary.lookupIndex("NOTFREQUENT");
								psi.add(notFrequentFeatureIndex, 1.0);

							}

							// For the scaling factor calculation
							sampleDictionary.lookupIndex(feature);
							featureList.add(feature);

						}
					}

					bean.setFeatureVec(psi);
					bean.setInsideTree(insideTree);
					bean.setSyntaxTree(tree);
					bean.setFootToRoot(footToRoot);
					bean.setTreeIdx(i);
					bean.setFeatureList(featureList);
					size = sparseVectors.size();

					if (sparseVectors.size() < M)
						sparseVectors.add(bean);
					else
						break mainloop;

				}
			}

		}
		sampleDictionary.stopGrowth();
		CommonUtil.scaleFeatures(sparseVectors, sampleDictionary, dictionary, M, k);
		logger.info("size of the vectors list is given by: " + sparseVectors.size());
		CommonUtil.serializeVec(sparseVectors, nonTerminal, featureVectorsStoragePath, "outside");
		logger.info("Memory USED NOW: " + Runtime.getRuntime().totalMemory());
		CommonUtil.serializeDictionary(dictionary, nonTerminal, featureVectorsStoragePath, "outside");
		logger.info("MEMORY USED: " + Runtime.getRuntime().totalMemory());
		CommonUtil.writeDictionaryToDisk(dictionary, sampleDictionary, nonTerminal, featureVectorsStoragePath,
				"outside");
		CommonUtil.writeStatisticsToDisk(sparseVectors, nonTerminal, featureVectorsStoragePath, "outside");

	}

	private void dictionaryInit() {
		try {
			LinkedList<Alphabet> dictionaries = CommonUtil.getDictionaries(featureDictionaries, nonTerminal, "outside");
			dprime = CommonUtil.getVectorDimensions(dictionaries);
			dictionary = CommonUtil.combineDictionaries(dictionaries, dprime);
			dictionary.stopGrowth();
		} catch (IOException e) {
			logger.error(e);
		} catch (ClassNotFoundException e) {
			logger.error(e);
		}

	}

}