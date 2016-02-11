package main;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Stack;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.log4j.Logger;

import dictionary.Alphabet;
import edu.berkeley.nlp.syntax.Constituent;
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.syntax.Trees.PennTreeReader;
import utils.CommonUtil;
import utils.PTBTreeNormaliser;

public class FeatureDictionary {

	private static Alphabet insideFeatures;
	private static Alphabet filteredInsideFeatures;
	private static Alphabet outsideFeatures;
	private static Alphabet filteredOutsideFeatures;
	private static Tree<String> insideTree;
	private static boolean isPreterminal;
	private static PTBTreeNormaliser treeNormalizer;
	private static Options options = new Options();
	private static String nonTerminal;
	private static String numOfTrees;
	private static String corpusRootDirectory;
	private static int threshHoldCount;
	private static String dictionaryWritePath;
	private static int nullTreeCounts;

	final static Logger logger = Logger.getLogger(FeatureDictionary.class);

	static {

		insideFeatures = new Alphabet();
		filteredInsideFeatures = new Alphabet();
		outsideFeatures = new Alphabet();
		filteredOutsideFeatures = new Alphabet();

		insideFeatures.turnOnCounts();
		filteredInsideFeatures.turnOnCounts();
		insideFeatures.allowGrowth();
		filteredInsideFeatures.allowGrowth();

		outsideFeatures.turnOnCounts();
		filteredOutsideFeatures.turnOnCounts();
		outsideFeatures.allowGrowth();
		filteredOutsideFeatures.allowGrowth();

		treeNormalizer = new PTBTreeNormaliser(true);

		options.addOption("n", true, "number of trees to parse for dictionary preparation");
		options.addOption("r", true,
				"Set the path of the root directory containing all the tree files. The files should contain only the trees and nothing else i.e. extracted trees from the BLLIP corpus");
		options.addOption("nt", true, "give the non-terminal for which you want to extract the feature dictionaries");
		options.addOption("p", true, "give the path of the directory where you want to store the feature dictionaries");
		options.addOption("t", true,
				"threshhold count for feature selection. All the features that have count less than the threshhold will be counted as NOTFREQUENT");
	}

	public static void main(String... args) {

		parse(args);

		logger.info("FEATURE EXTRACTION FOR NON TERMINAL: " + nonTerminal);
		long programStartTime = System.currentTimeMillis() / 60000;
		logger.info("START TIME: " + nonTerminal + " IS " + programStartTime);
		Thread updateInsideFeatureDict = null;
		Thread updateOutsideFeatureDict = null;

		List<String> treeFiles = CommonUtil.getTreeFilePaths(corpusRootDirectory, numOfTrees);
		logger.info("++ NUMBER OF TREE FILES:  " + treeFiles.size());

		for (String treeFile : treeFiles) {
			PennTreeReader treeParser = null;
			try {
				treeParser = CommonUtil.getTreeReader(treeFile);
			} catch (Exception e) {
				e.printStackTrace();
			}

			while (treeParser.hasNext()) {

				Tree<String> tree = getNormalizedTree(treeParser.next());
				Map<Tree<String>, Constituent<String>> constituentsMap = null;
				Iterator<Tree<String>> treeNodeItr = null;

				if (tree != null) {
					treeNodeItr = getTreeNodeIterator(tree);
					constituentsMap = tree.getConstituents();

				} else {
					nullTreeCounts++;
				}

				while (treeNodeItr != null && treeNodeItr.hasNext()) {

					insideTree = treeNodeItr.next();
					isPreterminal = checkIsPreterminal(insideTree);

					Stack<Tree<String>> footToRoot = new Stack<Tree<String>>();
					CommonUtil.updateFoottorootPath(footToRoot, tree, insideTree, constituentsMap);
					CommonUtil.getNumberOfOutsideWordsLeft(insideTree, constituentsMap, tree);
					CommonUtil.getNumberOfOutsideWordsRight(insideTree, constituentsMap, tree);

					if (!insideTree.isLeaf() && insideTree.getLabel().equalsIgnoreCase(nonTerminal)) {

						updateInsideFeatureDict = new Thread(new Runnable() {

							@Override
							public void run() {
								List<String> insideFeatureList = CommonUtil.getInsideFeatures(insideTree,
										isPreterminal);
								updateInsideFeatureDict(insideFeatureList);
							}
						});

						updateOutsideFeatureDict = new Thread(new Runnable() {

							@Override
							public void run() {
								List<String> outsideFeatureList = CommonUtil.getOutsideFeatures(footToRoot);
								updateOutsideFeatureDict(outsideFeatureList);

							}
						});

						updateInsideFeatureDict.start();
						updateOutsideFeatureDict.start();

						try {
							updateInsideFeatureDict.join();
							updateOutsideFeatureDict.join();
						} catch (InterruptedException e) {
							e.printStackTrace();
						}

					}

				}
			}
			if (nullTreeCounts > 0)
				logger.info("NULL TREES FOR FILE: " + treeFile + " ARE " + Integer.toString(nullTreeCounts) + " NT: "
						+ nonTerminal);
		}

		insideFeatures.stopGrowth();
		outsideFeatures.stopGrowth();

		filterFeatureDictionaries();
		writeDictionariesToDisk();
		writeDictionaryObjectsToDisk();

		long endTime = System.currentTimeMillis() / 60000;
		logger.info(
				"END TIME FOR: " + nonTerminal + " IS " + endTime + " DURATION IS: " + (endTime - programStartTime));

	}

	private static void writeDictionaryObjectsToDisk() {
		Thread t1 = new Thread(new Runnable() {

			@Override
			public void run() {
				ObjectOutputStream os = null;
				FileOutputStream fos = null;

				try {
					fos = new FileOutputStream(
							new File(dictionaryWritePath + "/" + nonTerminal + "/" + nonTerminal + "dictin.ser"));
					os = new ObjectOutputStream(fos);
					os.writeObject(filteredInsideFeatures);

				} catch (IOException e) {
					e.printStackTrace();
				}

			}
		});

		t1.start();

		Thread t2 = new Thread(new Runnable() {

			@Override
			public void run() {
				ObjectOutputStream os = null;
				FileOutputStream fos = null;
				try {
					fos = new FileOutputStream(
							new File(dictionaryWritePath + "/" + nonTerminal + "/" + nonTerminal + "dictout.ser"));
					os = new ObjectOutputStream(fos);
					os.writeObject(filteredOutsideFeatures);
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		});

		t2.start();

		try {
			t1.join();
			t2.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

	}

	private static void writeDictionariesToDisk() {
		Thread t1 = new Thread(new Runnable() {

			@Override
			public void run() {
				BufferedWriter bw = null;
				logger.info(
						"INSIDE FEATURE DICTIONARY SIZE FOR " + nonTerminal + " IS " + filteredInsideFeatures.size());
				for (Object obj : filteredInsideFeatures.map.keys()) {
					String feature = (String) obj;
					File dictionaryDir = new File(dictionaryWritePath + "/" + nonTerminal);
					if (!dictionaryDir.exists()) {
						dictionaryDir.mkdirs();
					}
					try {
						bw = new BufferedWriter(new FileWriter(dictionaryDir + "/" + nonTerminal + "dictin.txt", true));
						bw.write(feature + "\t" + filteredInsideFeatures.countMap.get(feature) + "\n");
						bw.close();
					} catch (IOException e) {
						e.printStackTrace();
					}

				}

			}
		});

		t1.start();

		Thread t2 = new Thread(new Runnable() {

			@Override
			public void run() {
				BufferedWriter bw = null;
				logger.info("OUTSIDE FEATURE DICT SIZE FOR " + nonTerminal + " IS " + filteredOutsideFeatures.size());
				for (Object obj : filteredOutsideFeatures.map.keys()) {
					String feature = (String) obj;
					File dictionaryDir = new File(dictionaryWritePath + "/" + nonTerminal);
					if (!dictionaryDir.exists()) {
						dictionaryDir.mkdirs();
					}
					try {
						bw = new BufferedWriter(
								new FileWriter(dictionaryDir + "/" + nonTerminal + "dictout.txt", true));
						bw.write(feature + "\t" + filteredOutsideFeatures.countMap.get(feature) + "\n");
						bw.close();
					} catch (IOException e) {
						e.printStackTrace();
					}

				}

			}
		});

		t2.start();

		try {
			t1.join();
			t2.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

	}

	private static void parse(String[] args) {
		CommandLineParser parser = new BasicParser();

		CommandLine cmd = null;

		try {
			cmd = parser.parse(options, args);
			if (cmd.hasOption("r") && cmd.hasOption("n") && cmd.hasOption("nt") && cmd.hasOption("p")
					&& cmd.hasOption("t")) {
				corpusRootDirectory = cmd.getOptionValue("r");
				numOfTrees = cmd.getOptionValue("n");
				nonTerminal = cmd.getOptionValue("nt");
				dictionaryWritePath = cmd.getOptionValue("p");
				threshHoldCount = Integer.parseInt(cmd.getOptionValue("t"));
			} else {
				help();
			}

		} catch (ParseException e) {

		}

	}

	private static void help() {
		HelpFormatter formater = new HelpFormatter();
		formater.printHelp("FeatureDictionary", options);
		System.exit(0);
	}

	private static void filterFeatureDictionaries() {
		Thread t1 = new Thread(new Runnable() {

			@Override
			public void run() {
				Object[] features = insideFeatures.map.keys();
				filteredInsideFeatures.lookupIndex("NOTFREQUENT");
				for (Object obj : features) {
					String feature = (String) obj;
					int featureFreq = insideFeatures.countMap.get(feature);
					if (featureFreq >= threshHoldCount) {
						filteredInsideFeatures.lookupIndex(feature);
						filteredInsideFeatures.countMap.put(feature, featureFreq);
					} else {
						filteredInsideFeatures.countMap.put("NOTFREQUENT",
								filteredInsideFeatures.countMap.get("NOTFREQUENT") + featureFreq);
					}
				}

			}
		});

		Thread t2 = new Thread(new Runnable() {

			@Override
			public void run() {
				Object[] features = outsideFeatures.map.keys();
				filteredOutsideFeatures.lookupIndex("NOTFREQUENT");
				for (Object obj : features) {
					String feature = (String) obj;
					int featureFreq = outsideFeatures.countMap.get(feature);
					if (featureFreq >= threshHoldCount) {
						filteredOutsideFeatures.lookupIndex(feature);
						filteredOutsideFeatures.countMap.put(feature, featureFreq);
					} else {
						filteredOutsideFeatures.countMap.put("NOTFREQUENT",
								filteredOutsideFeatures.countMap.get("NOTFREQUENT") + featureFreq);
					}
				}
			}
		});

		t1.start();
		t2.start();

		try {
			t1.join();
			t2.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

	}

	private static Tree<String> getNormalizedTree(Tree<String> tree) {
		Tree<String> syntaxTree = null;
		try {
			syntaxTree = treeNormalizer.process(tree);
		} catch (RuntimeException e) {

		}
		return syntaxTree;
	}

	private static Iterator<Tree<String>> getTreeNodeIterator(Tree<String> tree) {
		return tree.iterator();
	}

	private static void updateOutsideFeatureDict(List<String> outsideFeatureList) {
		for (String feature : outsideFeatureList) {
			outsideFeatures.lookupIndex(feature);
		}

	}

	private static void updateInsideFeatureDict(List<String> insideFeatureList) {
		for (String feature : insideFeatureList) {
			if (!feature.equalsIgnoreCase("notvalid")) {
				insideFeatures.lookupIndex(feature);
			}
		}
	}

	private static boolean checkIsPreterminal(Tree<String> insideTree2) {
		return insideTree.isPreTerminal();
	}

}
