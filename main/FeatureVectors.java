package main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Stack;

import org.apache.commons.cli.Options;
import org.apache.log4j.Logger;

import beans.FeatureVector;
import dictionary.Alphabet;
import edu.berkeley.nlp.syntax.Constituent;
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.syntax.Trees.PennTreeReader;
import interfaces.InsideFeature;
import utils.CommonUtil;
import utils.VSMSparseVector;

public class FeatureVectors {

	/*
	 * Thread for extracting inside feature vectors
	 */
	class InsideFeatureVectors implements Runnable {

		String featureVectorsStoragePath;
		Alphabet dictionary;
		InsideFeature featureObject;
		Logger logger;
		String nonTerminal;
		int d;
		Tree<String> insideTree;
		List<String> treeFiles;
		boolean isPreTerminal;
		LinkedList<VSMSparseVector> sparseVectors;
		String dictionaryPath;
		String featureName;
		int numOfVecs;

		public InsideFeatureVectors(List<String> treeFiles, InsideFeature featureObject, String featureDictionariesPath,
				String featureVectorsStoragePath, Logger logger, String nonTerminal, String dictionaryPath,
				String featureName, int numOfVecs) {
			this.treeFiles = treeFiles;
			this.featureVectorsStoragePath = featureVectorsStoragePath;
			this.logger = logger;
			this.nonTerminal = nonTerminal;
			this.featureObject = featureObject;
			this.dictionaryPath = dictionaryPath;
			this.featureName = featureName;
			this.numOfVecs = numOfVecs;
		}

		@Override
		public void run() {

			dictionaryInit();

			mainLoop: for (String treeFile : treeFiles) {
				PennTreeReader treeParser = null;
				try {
					treeParser = CommonUtil.getTreeReader(treeFile);

				} catch (Exception e) {
					e.printStackTrace();
				}

				while (treeParser.hasNext()) {
					Tree<String> tree = FeatureDictionary.getNormalizedTree(treeParser.next());
					Iterator<Tree<String>> treeNodeItr = null;
					Map<Tree<String>, Constituent<String>> constituentsMap = null;

					if (tree != null) {
						treeNodeItr = FeatureDictionary.getTreeNodeIterator(tree);
						constituentsMap = tree.getConstituents();

					}

					while (treeNodeItr != null && treeNodeItr.hasNext()) {
						insideTree = treeNodeItr.next();
						isPreTerminal = FeatureDictionary.checkIsPreterminal(insideTree);

						CommonUtil.setConstituentLength(constituentsMap.get(insideTree));

						if (!insideTree.isLeaf() && insideTree.getLabel().equalsIgnoreCase(nonTerminal)) {
							VSMSparseVector vec = new VSMSparseVector(d);
							String feature = featureObject.getFeature(insideTree, isPreTerminal);
							int featureIndex = dictionary.lookupIndex(feature);
							int featureCount = dictionary.countMap.get(feature);

							if (featureIndex != -1) {
								double tfIdf = Math.log(1.0 / (double) featureCount);
								vec.add(featureIndex, tfIdf);
							}

							if (sparseVectors.size() < numOfVecs)
								sparseVectors.add(vec);
							else
								break mainLoop;
						}
					}

				}

			}
			logger.info("size of the vectors list is given by: " + sparseVectors.size());
			serializeVec(sparseVectors);

		}

		private void serializeVec(LinkedList<VSMSparseVector> sparseVectors) {
			try {
				File file = new File(featureVectorsStoragePath + "/" + nonTerminal + "/" + "inside" + "/" + featureName
						+ "vecs.ser");
				if (!file.getParentFile().exists()) {
					file.getParentFile().mkdirs();
				}
				ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(file));
				os.writeObject(sparseVectors);
				os.flush();
				os.close();
			} catch (IOException e) {
				// TODO
			}
		}

		private void dictionaryInit() {
			try {
				ObjectInputStream ois = new ObjectInputStream(new FileInputStream(dictionaryPath));
				dictionary = (Alphabet) ois.readObject();
				dictionary.stopGrowth();
				d = dictionary.size();
			} catch (IOException e) {
				// TODO
			} catch (ClassNotFoundException e) {
				// TODO
			}

		}

	}

	/*
	 * Thread for extracting outside feature vectors
	 */
	class OutsideFeatureVectors implements Runnable {

		String featureVectorsStoragePath;
		Alphabet dictionary;
		InsideFeature featureObject;
		Logger logger;
		String nonTerminal;
		int dprime;
		Tree<String> insideTree;
		List<String> treeFiles;
		boolean isPreTerminal;
		LinkedList<FeatureVector> sparseVectors;
		String dictionaryPath;
		String featureName;
		int numOfVecs;

		public OutsideFeatureVectors(List<String> treeFiles, InsideFeature featureObject,
				String featureDictionariesPath, String featureVectorsStoragePath, Logger logger, String nonTerminal,
				String dictionaryPath, String featureName, int numOfVecs) {
			this.treeFiles = treeFiles;
			this.featureVectorsStoragePath = featureVectorsStoragePath;
			this.logger = logger;
			this.nonTerminal = nonTerminal;
			this.featureObject = featureObject;
			this.dictionaryPath = dictionaryPath;
			this.featureName = featureName;
			this.numOfVecs = numOfVecs;
		}

		@Override
		public void run() {

			dictionaryInit();
			if (dprime == 1) {
				logger.info("Exiting the thread because the feature dictionary size is 1 i.e no feature of the type: "
						+ featureName + " exists for the non-terminal " + nonTerminal);
				System.exit(-1);
			}

			mainLoop: for (int i = 0; i < treeFiles.size(); i++) {
				String treeFile = treeFiles.get(i);
				PennTreeReader treeParser = null;
				try {
					treeParser = CommonUtil.getTreeReader(treeFile);

				} catch (Exception e) {
					e.printStackTrace();
				}

				while (treeParser.hasNext()) {
					Tree<String> tree = FeatureDictionary.getNormalizedTree(treeParser.next());
					Iterator<Tree<String>> treeNodeItr = null;
					Map<Tree<String>, Constituent<String>> constituentsMap = null;

					if (tree != null) {
						treeNodeItr = FeatureDictionary.getTreeNodeIterator(tree);
						constituentsMap = tree.getConstituents();

					}

					while (treeNodeItr != null && treeNodeItr.hasNext()) {
						insideTree = treeNodeItr.next();
						isPreTerminal = FeatureDictionary.checkIsPreterminal(insideTree);
						Stack<Tree<String>> footToRoot = new Stack<Tree<String>>();
						CommonUtil.updateFoottorootPath(footToRoot, tree, insideTree, constituentsMap);

						CommonUtil.getNumberOfOutsideWordsLeft(insideTree, constituentsMap, tree);
						CommonUtil.getNumberOfOutsideWordsRight(insideTree, constituentsMap, tree);

						if (!insideTree.isLeaf() && insideTree.getLabel().equalsIgnoreCase(nonTerminal)) {
							FeatureVector bean = new FeatureVector();
							VSMSparseVector vec = new VSMSparseVector(dprime);
							String feature = featureObject.getFeature(insideTree, isPreTerminal);
							int featureIndex = dictionary.lookupIndex(feature);
							int featureCount = dictionary.countMap.get(feature);

							if (featureIndex != -1) {
								double tfIdf = Math.log(1.0 / (double) featureCount);
								vec.add(featureIndex, tfIdf);
							} else {
								int notFrequentFeatureIndex = dictionary.lookupIndex("NOTFREQUENT");
								int featureCountNF = dictionary.countMap.get("NOTFREQUENT");
								double tfIdf = Math.log(1.0 / (double) featureCountNF);
								vec.add(notFrequentFeatureIndex, tfIdf);

							}

							bean.setFeatureVec(vec);
							bean.setInsideTree(insideTree);
							bean.setSyntaxTree(tree);
							bean.setFootToRoot(footToRoot);
							bean.setTreeFileName(treeFile);
							bean.setTreeFileIdx(i);

							if (sparseVectors.size() < numOfVecs)
								sparseVectors.add(bean);
							else
								break mainLoop;
						}
					}

				}

			}
			logger.info("size of the vectors list is given by: " + sparseVectors.size());
			serializeVec(sparseVectors);

		}

		private void serializeVec(LinkedList<FeatureVector> featureVecs) {
			try {
				File file = new File(featureVectorsStoragePath + "/" + nonTerminal + "/" + "outside" + "/" + featureName
						+ "vecs.ser");
				if (!file.getParentFile().exists()) {
					file.getParentFile().mkdirs();
				}
				ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(file));
				os.writeObject(featureVecs);
				os.flush();
				os.close();
			} catch (IOException e) {
				// TODO
			}
		}

		private void dictionaryInit() {
			try {
				ObjectInputStream ois = new ObjectInputStream(new FileInputStream(dictionaryPath));
				dictionary = (Alphabet) ois.readObject();
				dictionary.stopGrowth();
				dprime = dictionary.size();
			} catch (IOException e) {
				// TODO
			} catch (ClassNotFoundException e) {
				// TODO
			}

		}

	}

	private static Options options;
	private static Alphabet insideBinFull;
	private static Alphabet insideBinLeft;
	private static Alphabet insideBinRight;
	private static Alphabet insideBinLeftPlus;
	private static Alphabet insideBinRightPlus;
	private static Alphabet insideNtHeadPos;
	private static Alphabet insideNtNumOdWords;
	private static Alphabet insideUnary;

	static {
		options.addOption("d", true, "destination folder for the feature vectors");
		options.addOption("nt", true, "feature vectors for which non-terminal?");
		options.addOption("M", true, "number of samples for a non-terminal");
		options.addOption("dict", true, "feature dictionary path");
	}

	public static void main(String[] args) {

		parse(args);

	}

	private static void parse(String[] args) {
		// TODO

	}

}
