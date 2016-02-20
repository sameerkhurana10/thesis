package main;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Iterator;
import java.util.LinkedHashMap;
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
import features.InsideBinFull;
import features.InsideBinLeft;
import features.InsideBinLeftPlus;
import features.InsideBinRight;
import features.InsideBinRightPlus;
import features.InsideNtHeadPos;
import features.InsideNtNumOfWords;
import features.InsideUnary;
import features.OutsideFootNumwordsleft;
import features.OutsideFootNumwordsright;
import features.OutsideFootParent;
import features.OutsideFootParentGrandParent;
import features.OutsideOtherheadposAbove;
import features.OutsideTreeAbove2;
import features.OutsideTreeAbove3;
import features.OutsideTreeabove1;
import interfaces.InsideFeature;
import interfaces.OutsideFeature;
import utils.CommonUtil;
import utils.PTBTreeNormaliser;

class ExtractInsideFeatureDictionary implements Runnable {

	Tree<String> insideTree;
	List<String> treeFiles;
	String nonTerminal;
	Alphabet featureDictionary;
	Alphabet filteredDictionary;
	InsideFeature featureObject;
	boolean isPreTerminal;
	String dictionaryDiskPath;
	String featureType;
	LinkedHashMap<String, String> treeMap;

	public ExtractInsideFeatureDictionary(List<String> treeFiles, String nonTerminal, Alphabet dictionary,
			Alphabet filteredDictionary, InsideFeature featureObject, String dictionaryDiskPath, String featureType) {
		this.treeFiles = treeFiles;
		this.nonTerminal = nonTerminal;
		this.featureObject = featureObject;
		this.featureDictionary = dictionary;
		this.filteredDictionary = filteredDictionary;
		this.dictionaryDiskPath = dictionaryDiskPath;
		this.featureType = featureType;
		treeMap = new LinkedHashMap<String, String>();
	}

	@Override
	public void run() {

		for (String treeFile : treeFiles) {
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
					/*
					 * Important to get the insideNtNumOfWords feature of
					 * course. Need to set the static variable
					 */
					CommonUtil.setConstituentLength(constituentsMap.get(insideTree));

					if (!insideTree.isLeaf() && insideTree.getLabel().equalsIgnoreCase(nonTerminal)) {
						String feature = featureObject.getFeature(insideTree, isPreTerminal);
						if (!feature.equalsIgnoreCase("NOTVALID")) {
							featureDictionary.lookupIndex(feature);
							treeMap.put(feature, insideTree.toString());
						}
					}
				}

			}

		}

		/*
		 * Filter the dictionary based on the threshold value given by the user
		 */
		FeatureDictionary.filterDictionary(featureDictionary, filteredDictionary);

		/*
		 * Write the dictionary to the disk
		 */
		FeatureDictionary.writeDictToDisk(nonTerminal, dictionaryDiskPath, filteredDictionary, featureType, "inside",
				treeMap);

	}

}

class ExtractOutsideFeatureDictionary implements Runnable {

	Tree<String> insideTree;
	List<String> treeFiles;
	String nonTerminal;
	Alphabet featureDictionary;
	Alphabet filteredDictionary;
	OutsideFeature featureObject;
	boolean isPreTerminal;
	String dictionaryDiskPath;
	String featureType;
	LinkedHashMap<String, String> treeMap;

	public ExtractOutsideFeatureDictionary(List<String> treeFiles, String nonTerminal, Alphabet dictionary,
			Alphabet filteredDictionary, OutsideFeature featureObject, String dictionaryDiskPath, String featureType) {
		this.treeFiles = treeFiles;
		this.nonTerminal = nonTerminal;
		this.featureObject = featureObject;
		this.featureDictionary = dictionary;
		this.filteredDictionary = filteredDictionary;
		this.dictionaryDiskPath = dictionaryDiskPath;
		this.featureType = featureType;
		treeMap = new LinkedHashMap<String, String>();
	}

	@Override
	public void run() {

		for (String treeFile : treeFiles) {
			PennTreeReader treeParser = null;
			try {
				treeParser = CommonUtil.getTreeReader(treeFile);
			} catch (Exception e) {
				e.printStackTrace();
			}

			while (treeParser.hasNext()) {
				Tree<String> tree = FeatureDictionary.getNormalizedTree(treeParser.next());
				Map<Tree<String>, Constituent<String>> constituentsMap = null;
				Iterator<Tree<String>> treeNodeItr = null;

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
						String feature = featureObject.getFeature(footToRoot);
						if (!feature.equalsIgnoreCase("NOTVALID")) {
							featureDictionary.lookupIndex(feature);
							treeMap.put(feature, footToRoot.toString());
						}
					}
				}

			}

		}

		/*
		 * Filter the dictionary after it has been created
		 */
		FeatureDictionary.filterDictionary(featureDictionary, filteredDictionary);

		/*
		 * Write dictionary to disk
		 */
		FeatureDictionary.writeDictToDisk(nonTerminal, dictionaryDiskPath, filteredDictionary, featureType, "outside",
				treeMap);
	}

}

public class FeatureDictionary {

	private static Alphabet insideFeatures;
	private static Alphabet filteredInsideFeatures;
	private static Alphabet outsideFeatures;
	private static Alphabet filteredOutsideFeatures;
	private static Tree<String> insideTree;
	private static boolean isPreterminal;
	private static PTBTreeNormaliser treeNormalizer;
	private static Options options = new Options();
	static String nonTerminal;
	private static String numOfTrees;
	private static String corpusRootDirectory;
	private static int threshHoldCount;
	private static String dictionaryWritePath;
	private static int nullTreeCounts;

	/*
	 * Feature dictionaries for each of the features that we want to extract for
	 * a given tree node
	 */
	static Alphabet insideBinFull;
	private static Alphabet insideBinLeft;
	private static Alphabet insideBinLeftPlus;
	private static Alphabet insideBinRight;
	private static Alphabet insideBinRightPlus;
	private static Alphabet insideNtHeadPos;
	private static Alphabet insideNtNumOfWords;
	private static Alphabet insideUnary;
	private static Alphabet outsideFootNumWordsLeft;
	private static Alphabet outsideFootNumWordsRight;
	private static Alphabet outsideFootParent;
	private static Alphabet outsideFootParentGrandParent;
	private static Alphabet outsideOtherHeadPosAbove;
	private static Alphabet outsideTreeAbove1;
	private static Alphabet outsideTreeAbove2;
	private static Alphabet outsideTreeAbove3;

	/*
	 * The filtered dictionaries are the ones that are finally stored to the
	 * disk. It has features that are threshholded based on their counts
	 */
	private static Alphabet filteredInsideBinFull;
	private static Alphabet filteredInsideBinLeft;
	private static Alphabet filteredInsideBinLeftPlus;
	private static Alphabet filteredInsideBinRight;
	private static Alphabet filteredInsideBinRightPlus;
	private static Alphabet filteredInsideNtHeadPos;
	private static Alphabet filteredInsideNtNumOfWords;
	private static Alphabet filteredInsideUnary;
	private static Alphabet filteredOutsideFootNumWordsLeft;
	private static Alphabet filteredOutsideFootNumWordsRight;
	private static Alphabet filteredOutsideFootParent;
	private static Alphabet filteredOutsideFootParentGrandParent;
	private static Alphabet filteredOutsideOtherHeadPosAbove;
	private static Alphabet filteredOutsideTreeAbove1;
	private static Alphabet filteredOutsideTreeAbove2;
	private static Alphabet filteredOutsideTreeAbove3;

	final static Logger logger = Logger.getLogger(FeatureDictionary.class);

	/*
	 * static initialization block
	 */

	static {

		/*
		 * instantiating the feature dictionaries
		 */
		insideBinFull = new Alphabet();
		filteredInsideBinFull = new Alphabet();
		insideBinFull.allowGrowth();
		insideBinFull.turnOnCounts();
		filteredInsideBinFull.allowGrowth();
		filteredInsideBinFull.turnOnCounts();

		insideBinLeft = new Alphabet();
		insideBinLeft.allowGrowth();
		insideBinLeft.turnOnCounts();
		filteredInsideBinLeft = new Alphabet();
		filteredInsideBinLeft.allowGrowth();
		filteredInsideBinLeft.turnOnCounts();

		insideBinLeftPlus = new Alphabet();
		insideBinLeftPlus.allowGrowth();
		insideBinLeftPlus.turnOnCounts();
		filteredInsideBinLeftPlus = new Alphabet();
		filteredInsideBinLeftPlus.allowGrowth();
		filteredInsideBinLeftPlus.turnOnCounts();

		insideBinRight = new Alphabet();
		insideBinRight.allowGrowth();
		insideBinRight.turnOnCounts();
		filteredInsideBinRight = new Alphabet();
		filteredInsideBinRight.allowGrowth();
		filteredInsideBinRight.turnOnCounts();

		insideBinRightPlus = new Alphabet();
		insideBinRightPlus.allowGrowth();
		insideBinRightPlus.turnOnCounts();
		filteredInsideBinRightPlus = new Alphabet();
		filteredInsideBinRightPlus.allowGrowth();
		filteredInsideBinRightPlus.turnOnCounts();

		insideNtHeadPos = new Alphabet();
		insideNtHeadPos.allowGrowth();
		insideNtHeadPos.turnOnCounts();
		filteredInsideNtHeadPos = new Alphabet();
		filteredInsideNtHeadPos.allowGrowth();
		filteredInsideNtHeadPos.turnOnCounts();

		insideNtNumOfWords = new Alphabet();
		insideNtNumOfWords.allowGrowth();
		insideNtNumOfWords.turnOnCounts();
		filteredInsideNtNumOfWords = new Alphabet();
		filteredInsideNtNumOfWords.allowGrowth();
		filteredInsideNtNumOfWords.turnOnCounts();

		insideUnary = new Alphabet();
		insideUnary.allowGrowth();
		insideUnary.turnOnCounts();
		filteredInsideUnary = new Alphabet();
		filteredInsideUnary.allowGrowth();
		filteredInsideUnary.turnOnCounts();

		outsideFootNumWordsLeft = new Alphabet();
		outsideFootNumWordsLeft.allowGrowth();
		outsideFootNumWordsLeft.turnOnCounts();
		filteredOutsideFootNumWordsLeft = new Alphabet();
		filteredOutsideFootNumWordsLeft.allowGrowth();
		filteredOutsideFootNumWordsLeft.turnOnCounts();

		outsideFootNumWordsRight = new Alphabet();
		outsideFootNumWordsRight.allowGrowth();
		outsideFootNumWordsRight.turnOnCounts();
		filteredOutsideFootNumWordsRight = new Alphabet();
		filteredOutsideFootNumWordsRight.allowGrowth();
		filteredOutsideFootNumWordsRight.turnOnCounts();

		outsideFootParent = new Alphabet();
		outsideFootParent.allowGrowth();
		outsideFootParent.turnOnCounts();
		filteredOutsideFootParent = new Alphabet();
		filteredOutsideFootParent.allowGrowth();
		filteredOutsideFootParent.turnOnCounts();

		outsideFootParentGrandParent = new Alphabet();
		outsideFootParentGrandParent.allowGrowth();
		outsideFootParentGrandParent.turnOnCounts();
		filteredOutsideFootParentGrandParent = new Alphabet();
		filteredOutsideFootParentGrandParent.allowGrowth();
		filteredOutsideFootParentGrandParent.turnOnCounts();

		outsideOtherHeadPosAbove = new Alphabet();
		outsideOtherHeadPosAbove.allowGrowth();
		outsideOtherHeadPosAbove.turnOnCounts();
		filteredOutsideOtherHeadPosAbove = new Alphabet();
		filteredOutsideOtherHeadPosAbove.allowGrowth();
		filteredOutsideOtherHeadPosAbove.turnOnCounts();

		outsideTreeAbove1 = new Alphabet();
		outsideTreeAbove1.allowGrowth();
		outsideTreeAbove1.turnOnCounts();
		filteredOutsideTreeAbove1 = new Alphabet();
		filteredOutsideTreeAbove1.allowGrowth();
		filteredOutsideTreeAbove1.turnOnCounts();

		outsideTreeAbove2 = new Alphabet();
		outsideTreeAbove2.allowGrowth();
		outsideTreeAbove2.turnOnCounts();
		filteredOutsideTreeAbove2 = new Alphabet();
		filteredOutsideTreeAbove2.allowGrowth();
		filteredOutsideTreeAbove2.turnOnCounts();

		outsideTreeAbove3 = new Alphabet();
		outsideTreeAbove3.allowGrowth();
		outsideTreeAbove3.turnOnCounts();
		filteredOutsideTreeAbove3 = new Alphabet();
		filteredOutsideTreeAbove3.allowGrowth();
		filteredOutsideTreeAbove3.turnOnCounts();

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

		List<String> treeFiles = CommonUtil.getTreeFilePaths(corpusRootDirectory, numOfTrees);
		logger.info("++ NUMBER OF TREE FILES:  " + treeFiles.size());

		/*
		 * First creating the inside dictionaries concurrently. Not putting too
		 * much load on the system. To avoid the "too many open files" exception
		 * //
		 */
		Thread insideBinFullDictionary = new Thread(new ExtractInsideFeatureDictionary(treeFiles, nonTerminal,
				insideBinFull, filteredInsideBinFull, new InsideBinFull(), dictionaryWritePath, "insideBinFull"));

		Thread insideBinRightDictionary = new Thread(new ExtractInsideFeatureDictionary(treeFiles, nonTerminal,
				insideBinRight, filteredInsideBinRight, new InsideBinRight(), dictionaryWritePath, "insideBinRight"));

		Thread insideBinRightPlusDictionary = new Thread(new ExtractInsideFeatureDictionary(treeFiles, nonTerminal,
				insideBinRightPlus, filteredInsideBinRightPlus, new InsideBinRightPlus(), dictionaryWritePath,
				"insideBinRightPlus"));

		Thread insideBinLeftDictionary = new Thread(new ExtractInsideFeatureDictionary(treeFiles, nonTerminal,
				insideBinLeft, filteredInsideBinLeft, new InsideBinLeft(), dictionaryWritePath, "insideBinLeft"));

		insideBinFullDictionary.start();
		insideBinRightDictionary.start();
		insideBinRightPlusDictionary.start();
		insideBinLeftDictionary.start();

		try {
			insideBinFullDictionary.join();
			insideBinRightDictionary.join();
			insideBinRightPlusDictionary.join();
			insideBinLeftDictionary.join();
		} catch (InterruptedException e1) {
			logger.error("exception while joining the top 4 featuredictionaries");
		}

		Thread insideBinLeftPlusDictionary = new Thread(
				new ExtractInsideFeatureDictionary(treeFiles, nonTerminal, insideBinLeftPlus, filteredInsideBinLeftPlus,
						new InsideBinLeftPlus(), dictionaryWritePath, "insideBinLeftPlus"));

		Thread ntHeadPosDictionary = new Thread(
				new ExtractInsideFeatureDictionary(treeFiles, nonTerminal, insideNtHeadPos, filteredInsideNtHeadPos,
						new InsideNtHeadPos(), dictionaryWritePath, "insideNtHeadPos"));

		Thread ntNumOfWordsDictionary = new Thread(new ExtractInsideFeatureDictionary(treeFiles, nonTerminal,
				insideNtNumOfWords, filteredInsideNtNumOfWords, new InsideNtNumOfWords(), dictionaryWritePath,
				"insideNtNumOfWords"));

		Thread insideUnaryDictionary = new Thread(new ExtractInsideFeatureDictionary(treeFiles, nonTerminal,
				insideUnary, filteredInsideUnary, new InsideUnary(), dictionaryWritePath, "insideUnary"));

		insideBinLeftPlusDictionary.start();
		ntHeadPosDictionary.start();
		ntNumOfWordsDictionary.start();
		insideUnaryDictionary.start();

		try {
			insideBinLeftPlusDictionary.join();
			ntHeadPosDictionary.join();
			ntNumOfWordsDictionary.join();
			insideUnaryDictionary.join();
		} catch (InterruptedException e) {
			logger.error("error while joining the inside feature dictionary threads" + e);
		}

		Thread outsideFootNumWordsLeftDictionary = new Thread(new ExtractOutsideFeatureDictionary(treeFiles,
				nonTerminal, outsideFootNumWordsLeft, filteredOutsideFootNumWordsLeft, new OutsideFootNumwordsleft(),
				dictionaryWritePath, "outsideFootNumWordsLeft"));

		Thread outsideFootNumWordsRightDictionary = new Thread(new ExtractOutsideFeatureDictionary(treeFiles,
				nonTerminal, outsideFootNumWordsRight, filteredOutsideFootNumWordsRight, new OutsideFootNumwordsright(),
				dictionaryWritePath, "outsideFootNumWordsRight"));

		Thread outsideFootParentDictionary = new Thread(
				new ExtractOutsideFeatureDictionary(treeFiles, nonTerminal, outsideFootParent,
						filteredOutsideFootParent, new OutsideFootParent(), dictionaryWritePath, "outsideFootParent"));

		Thread outsideFootParentGrandParentDictionary = new Thread(new ExtractOutsideFeatureDictionary(treeFiles,
				nonTerminal, outsideFootParentGrandParent, filteredOutsideFootParentGrandParent,
				new OutsideFootParentGrandParent(), dictionaryWritePath, "outsideFootParentGrandParent"));

		outsideFootNumWordsRightDictionary.start();
		outsideFootNumWordsLeftDictionary.start();
		outsideFootParentDictionary.start();
		outsideFootParentGrandParentDictionary.start();

		try {
			outsideFootNumWordsLeftDictionary.join();
			outsideFootNumWordsRightDictionary.join();
			outsideFootParentDictionary.join();
			outsideFootParentGrandParentDictionary.join();
		} catch (InterruptedException e1) {
			logger.error("error while joining the top 4 outside feature dictionary threads");
		}

		Thread outsideOtherHeadPosAboveDictionary = new Thread(new ExtractOutsideFeatureDictionary(treeFiles,
				nonTerminal, outsideOtherHeadPosAbove, filteredOutsideOtherHeadPosAbove, new OutsideOtherheadposAbove(),
				dictionaryWritePath, "outsideOtherHeadPosAbove"));

		Thread outsideTreeAbove1Dictionary = new Thread(
				new ExtractOutsideFeatureDictionary(treeFiles, nonTerminal, outsideTreeAbove1,
						filteredOutsideTreeAbove1, new OutsideTreeabove1(), dictionaryWritePath, "outsideTreeAbove1"));

		Thread outsideTreeAbove2Dictionary = new Thread(
				new ExtractOutsideFeatureDictionary(treeFiles, nonTerminal, outsideTreeAbove2,
						filteredOutsideTreeAbove2, new OutsideTreeAbove2(), dictionaryWritePath, "outsideTreeAbove2"));

		Thread outsideTreeAbove3Dictionary = new Thread(
				new ExtractOutsideFeatureDictionary(treeFiles, nonTerminal, outsideTreeAbove3,
						filteredOutsideTreeAbove3, new OutsideTreeAbove3(), dictionaryWritePath, "outsideTreeAbove3"));

		outsideOtherHeadPosAboveDictionary.start();
		outsideTreeAbove1Dictionary.start();
		outsideTreeAbove2Dictionary.start();
		outsideTreeAbove3Dictionary.start();

		try {

			outsideOtherHeadPosAboveDictionary.join();
			outsideTreeAbove1Dictionary.join();
			outsideTreeAbove2Dictionary.join();
			outsideTreeAbove3Dictionary.join();

		} catch (InterruptedException e) {
			logger.error("Error while joining the outside dictionary threads: " + e);
		}

		long endTime = System.currentTimeMillis() / 60000;
		logger.info(
				"END TIME FOR: " + nonTerminal + " IS " + endTime + " DURATION IS: " + (endTime - programStartTime));

	}

	public static void writeDictToDisk(String nonTerminal, String dictionaryDiskPath, Alphabet dictionary,
			String featureType, String dictionaryType, LinkedHashMap<String, String> treeMap) {

		ObjectOutputStream os1 = null;
		ObjectOutputStream os2 = null;
		FileOutputStream fos1 = null;
		FileOutputStream fos2 = null;

		BufferedWriter bw = null;

		File dictionaryDirObj = new File(dictionaryDiskPath + "/" + nonTerminal + "/ser/" + dictionaryType);
		if (!dictionaryDirObj.exists()) {
			dictionaryDirObj.mkdirs();
		}
		File dictionaryDirTxt = new File(dictionaryDiskPath + "/" + nonTerminal + "/txt/" + dictionaryType);
		if (!dictionaryDirTxt.exists()) {
			dictionaryDirTxt.mkdirs();
		}

		try {

			fos1 = new FileOutputStream(
					new File(dictionaryDirObj.getAbsolutePath() + "/" + nonTerminal + "." + featureType + ".ser"));
			fos2 = new FileOutputStream(
					new File(dictionaryDirObj.getAbsolutePath() + "/" + nonTerminal + "." + featureType + ".map.ser"));

			os1 = new ObjectOutputStream(fos1);
			os2 = new ObjectOutputStream(fos2);

			os1.writeObject(dictionary);
			os2.writeObject(treeMap);

			os1.flush();
			os1.close();
			os2.flush();
			os2.close();

			if (dictionary.size() > 0) {
				logger.info("Writing dictionary " + dictionaryDirTxt.getAbsolutePath() + "/" + nonTerminal + "."
						+ featureType);
				for (Object obj : dictionary.map.keys()) {
					bw = new BufferedWriter(new FileWriter(
							dictionaryDirTxt.getAbsolutePath() + "/" + nonTerminal + "." + featureType + ".txt", true));
					String feature = (String) obj;
					String count = Integer.toString(dictionary.countMap.get(feature));
					bw.write(feature + "\t" + count + "\n");
					bw.flush();
					bw.close();
				}
			} else {
				bw = new BufferedWriter(new FileWriter(
						dictionaryDirTxt.getAbsolutePath() + "/" + nonTerminal + "." + featureType + ".empty.txt",
						true));
				bw.write("EMPTY FEATURE DICTIONARY");
				bw.flush();
				bw.close();
				logger.info(
						"EMPTY DICTIONARY FOR THE NON TERMINAL** " + nonTerminal + " **FEATURE TYPE** " + featureType);
			}
		} catch (FileNotFoundException e) {
			logger.error("Exception while writing the dictionary to the disk** " + e);
		} catch (IOException e) {
			logger.error("Exception while writing the dictionary to the disk** " + e);
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

	static void filterDictionary(Alphabet dictionary, Alphabet filteredDictionary) {
		dictionary.stopGrowth();
		Object[] features = dictionary.map.keys();
		filteredDictionary.lookupIndex("NOTFREQUENT");
		for (Object obj : features) {
			String feature = (String) obj;
			int featureFreq = dictionary.countMap.get(feature);
			if (featureFreq >= threshHoldCount) {
				filteredDictionary.lookupIndex(feature);
				filteredDictionary.countMap.put(feature, featureFreq);
			} else {
				filteredDictionary.countMap.put("NOTFREQUENT",
						filteredDictionary.countMap.get("NOTFREQUENT") + featureFreq);
			}
		}

	}

	static Tree<String> getNormalizedTree(Tree<String> tree) {
		Tree<String> syntaxTree = null;
		try {
			syntaxTree = treeNormalizer.process(tree);
		} catch (RuntimeException e) {

		}
		return syntaxTree;
	}

	static Iterator<Tree<String>> getTreeNodeIterator(Tree<String> tree) {
		return tree.iterator();
	}

	static boolean checkIsPreterminal(Tree<String> insideTree) {
		return insideTree.isPreTerminal();
	}

}
