package runnables;

import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;

import org.apache.log4j.Logger;

import beans.FeatureVector;
import dictionary.Alphabet;
import edu.berkeley.nlp.syntax.Constituent;
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.syntax.Trees.PennTreeReader;
import interfaces.InsideFeature;
import main.FeatureDictionary;
import utils.CommonUtil;
import utils.VSMSparseVector;

public class InsideFeatureVectors implements Runnable {

	String featureVectorsStoragePath;
	Alphabet dictionary;
	InsideFeature featureObject;
	Logger logger;
	String nonTerminal;
	int d;
	Tree<String> insideTree;
	String treeFile;
	boolean isPreTerminal;
	LinkedList<FeatureVector> sparseVectors;
	String dictionaryPath;

	int M;
	String featureDictionaries;
	LinkedList<InsideFeature> features;
	Alphabet sampleDictionary;
	int k;

	public InsideFeatureVectors(String treeFile, String featureVectorsStoragePath, Logger logger, String nonTerminal,
			int M, String featureDictionaries, LinkedList<InsideFeature> features, int k) {
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
	}

	@Override
	public void run() {

		dictionaryInit();
		PennTreeReader treeParser = null;
		try {
			treeParser = CommonUtil.getTreeReaderBz(treeFile);

		} catch (Exception e) {
			e.printStackTrace();
		}
		int i = 0;
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

				// Function phi follows

				insideTree = treeNodeItr.next();
				isPreTerminal = FeatureDictionary.checkIsPreterminal(insideTree);

				CommonUtil.setConstituentLength(constituentsMap.get(insideTree));

				if (!insideTree.isLeaf() && insideTree.getLabel().equalsIgnoreCase(nonTerminal)) {

					FeatureVector bean = new FeatureVector();
					VSMSparseVector phi = new VSMSparseVector(d);
					LinkedList<String> featureList = new LinkedList<String>();

					for (InsideFeature insideFeatureObj : features) {
						String feature = insideFeatureObj.getFeature(insideTree, isPreTerminal);
						if (!feature.equalsIgnoreCase("NOTVALID")) {
							int featureIndex = dictionary.lookupIndex(feature);
							if (featureIndex != -1) {
								phi.add(featureIndex, 1.0);
							} else {
								int notFrequentFeatureIndex = dictionary.lookupIndex("NOTFREQUENT");
								phi.add(notFrequentFeatureIndex, 1.0);

							}

							// For the scaling factor calculation
							sampleDictionary.lookupIndex(feature);
							featureList.add(feature);

						}
					}

					bean.setFeatureVec(phi);
					bean.setInsideTree(insideTree);
					bean.setSyntaxTree(tree);
					bean.setTreeIdx(i);
					bean.setFeatureList(featureList);

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
		CommonUtil.serializeVec(sparseVectors, nonTerminal, featureVectorsStoragePath, "inside");
		CommonUtil.serializeDictionary(dictionary, nonTerminal, featureVectorsStoragePath, "inside");
		CommonUtil.writeDictionaryToDisk(dictionary, sampleDictionary, nonTerminal, featureVectorsStoragePath,
				"inside");
		CommonUtil.writeStatisticsToDisk(sparseVectors, nonTerminal, featureVectorsStoragePath, "inside");

	}

	private void dictionaryInit() {
		try {
			LinkedList<Alphabet> dictionaries = CommonUtil.getDictionaries(featureDictionaries, nonTerminal, "inside");
			d = CommonUtil.getVectorDimensions(dictionaries);
			dictionary = CommonUtil.combineDictionaries(dictionaries, d);
			dictionary.stopGrowth();
		} catch (IOException e) {
			logger.error(e);
		} catch (ClassNotFoundException e) {
			logger.error(e);
		}

	}

}