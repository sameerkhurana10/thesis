package utils;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Stack;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.collections.map.MultiValueMap;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.commons.lang3.StringUtils;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLDouble;

import Jama.Matrix;
import beans.FeatureVector;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
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
import jeigen.DenseMatrix;
import jeigen.SparseMatrixLil;
import main.FeatureDictionary;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.VectorEntry;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import no.uib.cipr.matrix.sparse.SparseVector;
import superclass.VSMThesis;
import weka.core.Stopwords;

/**
 * This is a Utility class for the Project Vector Space Modelling
 * 
 * @author sameerkhurana10
 *
 */

public class CommonUtil extends VSMThesis {

	private static ArrayList<File> filePaths = new ArrayList<File>();
	private static int fileNum;
	private static int index;
	private static int id;
	private static int treeCount;

	// private static final Logger LOGGER;

	/**
	 * This method needs to be called when extracting the outside features for
	 * any particular node in the tree. This will be called from inside the loop
	 * while looping over all the tree nodes. Becaise each node will have its
	 * own path. The method is written by Dr Shay Cohen
	 * 
	 * @param foottoroot
	 * @param subroot
	 * @param insideTree
	 * @return
	 */
	public static Stack<Tree<String>> updateFoottorootPath(Stack<Tree<String>> foottoroot, Tree<String> subroot,
			Tree<String> insideTree, Map<Tree<String>, Constituent<String>> constituentsMap) {
		foottoroot.push(subroot);

		Tree<String> footTree = insideTree;
		Constituent<String> footconstituent = constituentsMap.get(footTree);

		if (subroot.equals(footTree)) {
			return foottoroot;
		} else {
			List<Tree<String>> children = subroot.getChildren();
			for (int i = 0; i < children.size(); i++) {
				Tree<String> childTree = children.get(i);
				Constituent<String> childConstituent = constituentsMap.get(childTree);
				if ((footconstituent.getStart() >= childConstituent.getStart())
						&& (footconstituent.getEnd() <= childConstituent.getEnd())) {
					updateFoottorootPath(foottoroot, childTree, insideTree, constituentsMap);
					break;
				}
			}
		}
		return foottoroot;
	}

	public static List<String> getInsideFeatures(Tree<String> insideTree, boolean isPreterminal) {
		List<String> features = Arrays.asList(new InsideBinFull().getFeature(insideTree, isPreterminal),
				new InsideBinLeft().getFeature(insideTree, isPreterminal),
				new InsideBinLeftPlus().getFeature(insideTree, isPreterminal),
				new InsideBinRight().getFeature(insideTree, isPreterminal),
				new InsideBinRightPlus().getFeature(insideTree, isPreterminal),
				new InsideNtHeadPos().getFeature(insideTree, isPreterminal),
				new InsideNtNumOfWords().getFeature(insideTree, isPreterminal),
				new InsideUnary().getFeature(insideTree, isPreterminal));

		return features;
	}

	public static List<String> getOutsideFeatures(Stack<Tree<String>> footToRoot) {

		String outsideFootNumwordsleft = new OutsideFootNumwordsleft().getFeature(footToRoot);
		String outsideFootNumwordsright = new OutsideFootNumwordsright().getFeature(footToRoot);
		String outsideFootParentGrandParent = new OutsideFootParentGrandParent().getFeature(footToRoot);
		String outsideOtherheadposAbove = new OutsideOtherheadposAbove().getFeature(footToRoot);
		String outsideFootParent = new OutsideFootParent().getFeature(footToRoot);
		String outsideTreeabove1 = new OutsideTreeabove1().getFeature(footToRoot);
		// String outsideTreeAbove2 = new
		// OutsideTreeAbove2().getFeature(footToRoot);
		String outsideTreeAbove3 = new OutsideTreeAbove3().getFeature(footToRoot);

		List<String> features = new LinkedList<String>();

		if (isValid(outsideFootNumwordsleft))
			features.add(outsideFootNumwordsleft);
		if (isValid(outsideFootNumwordsright))
			features.add(outsideFootNumwordsright);
		if (isValid(outsideFootParentGrandParent))
			features.add(outsideFootParentGrandParent);
		if (isValid(outsideOtherheadposAbove))
			features.add(outsideOtherheadposAbove);
		if (isValid(outsideFootParent))
			features.add(outsideFootParent);
		if (isValid(outsideTreeabove1))
			features.add(outsideTreeabove1);
		// if (isValid(outsideTreeAbove2))
		// features.add(outsideTreeAbove2);
		if (isValid(outsideTreeAbove3))
			features.add(outsideTreeAbove3);

		return features;
	}

	public static boolean isValid(String feature) {
		boolean validity = true;
		if (feature.equalsIgnoreCase("NOTVALID"))
			validity = false;

		return validity;
	}

	/**
	 * 
	 * @param tree
	 * @return
	 */
	public static String getTreeString(Tree<String> tree) {
		if (tree.isPreTerminal()) {
			return (tree.getLabel() + "->" + tree.getChildren().get(0).getLabel().toLowerCase());
		} else {
			List<Tree<String>> children = tree.getChildren();
			if (children.size() > 1) {
				return (tree.getLabel() + "->" + children.get(0).getLabel() + "," + children.get(1).getLabel());
			} else {
				return null;
			}
		}
	}

	public static int getFeatureId(Alphabet source, String feature) {

		int featureid = source.lookupIndex(feature);

		/*
		 * TODO Commented out for now as we do not have NOTFREQUENT yet, because
		 * we have not filtered our features yet. Once we filter our features
		 * then we will have a NOTFREQUENT feature for each Alphabet
		 */

		if (featureid == -1) {

			featureid = source.lookupIndex("NOTFREQUENT");
		}

		return featureid;
	}

	public static int getVocabIndex(Alphabet source, String feature) {

		int featureid = source.lookupIndex(feature);

		/*
		 * TODO Commented out for now as we do not have NOTFREQUENT yet, because
		 * we have not filtered our features yet. Once we filter our features
		 * then we will have a <OOV> feature for each Alphabet
		 */

		if (featureid == -1) {

			featureid = source.lookupIndex("<OOV>");
		}

		return featureid;
	}

	/**
	 * TODO
	 * 
	 * @param URI
	 * @return
	 * @throws Exception
	 */
	public static PennTreeReader getTreeReader(String URI) throws Exception {
		System.out.println("++++URI+++" + URI);
		InputStreamReader inputData = new InputStreamReader(new FileInputStream(URI), "UTF-8");

		return new PennTreeReader(inputData);
	}

	public static PennTreeReader getTreeReaderBz(String URI) throws Exception {

		return new PennTreeReader(BLLIPCorpusReader.getBufferedReaderForBZ2File(URI));
	}

	/**
	 * This method should be called from inside the loop while iterating the
	 * nodes of a particular tree, so that the length variable can be updated
	 * for each node, as each node will have a different constituent length
	 * 
	 * @param constituent
	 */
	public static void setConstituentLength(Constituent<String> constituent) {
		/*
		 * Just setting the static variable
		 */

		InsideNtNumOfWords.length = constituent.getLength();

	}

	/**
	 * The method that returns the inside feature vector dimensions when passed
	 * the object store
	 * 
	 * @return
	 */
	public static int getInsideFeatureVectorDimensions(ArrayList<Alphabet> updatedFilteredDictionary) {

		int vectorDimension = 0;

		/*
		 * Getting the vector dimensitons of the inside feature vector phi, just
		 * by adding all the dictionary sizes together
		 */
		for (Alphabet dictionary : updatedFilteredDictionary) {
			vectorDimension += dictionary.size();
		}

		return vectorDimension;

	}

	/**
	 * 
	 * @param parentTree
	 * @param footTree
	 * @return
	 */
	public static String getStringFromParent(Tree<String> parentTree, Tree<String> footTree) {
		String feature = null;
		List<Tree<String>> children = parentTree.getChildren();
		if (children.size() > 1) {
			if (children.get(0).equals(footTree)) {
				// Left foot
				feature = parentTree.getLabel() + "->" + children.get(0).getLabel() + "*," + children.get(1).getLabel();
			} else {
				// right foot
				feature = parentTree.getLabel() + "->" + children.get(0).getLabel() + "," + children.get(1).getLabel()
						+ "*";
			}
		} else {
			feature = "NOTVALID";
		}
		return feature;
	}

	/**
	 * 
	 * @param grandparentTree
	 * @param parentTree
	 * @param footTree
	 * @return
	 */
	public static String getStringFromGrandparent(Tree<String> grandparentTree, Tree<String> parentTree,
			Tree<String> footTree) {
		String feature = null;
		List<Tree<String>> parents = grandparentTree.getChildren();
		if (parents.size() > 1) {
			if (parents.get(0).equals(parentTree)) {
				feature = grandparentTree.getLabel() + "->(" + getStringFromParent(parentTree, footTree) + "),"
						+ parents.get(1).getLabel();
			} else {
				feature = grandparentTree.getLabel() + "->" + parents.get(0).getLabel() + ",("
						+ getStringFromParent(parentTree, footTree) + ")";
			}
		} else {
			feature = "NOTVALID";
		}
		return feature;
	}

	/**
	 * 
	 * @param greatgrandparentTree
	 * @param grandparentTree
	 * @param parentTree
	 * @param footTree
	 * @return
	 */
	public static String getStringFromGreatgrandparent(Tree<String> greatgrandparentTree, Tree<String> grandparentTree,
			Tree<String> parentTree, Tree<String> footTree) {

		String feature = null;
		List<Tree<String>> grandparents = greatgrandparentTree.getChildren();
		if (grandparents.size() > 1) {
			if (grandparents.get(0).equals(grandparentTree)) {
				feature = greatgrandparentTree.getLabel() + "->("
						+ getStringFromGrandparent(grandparentTree, parentTree, footTree) + "),"
						+ grandparents.get(1).getLabel();
			} else {
				feature = greatgrandparentTree.getLabel() + "->" + grandparents.get(0).getLabel() + ",("
						+ getStringFromGrandparent(grandparentTree, parentTree, footTree) + ")";
			}
		} else {
			feature = "NOTVALID";
		}
		return feature;
	}

	/**
	 * Have to call this method from inside the while loop that iterates over
	 * the nodes in a tree
	 * 
	 * @param insideTree
	 * @param constituentsMap
	 * @param root
	 * @return
	 */
	public static void getNumberOfOutsideWordsRight(Tree<String> insideTree,
			Map<Tree<String>, Constituent<String>> constituentsMap, Tree<String> root) {

		int numOfWords = 0;
		/*
		 * Getting the end of the inside tree
		 */
		int footconstituent_end = constituentsMap.get(insideTree).getEnd();
		/*
		 * Getting the end of the sentence
		 */
		int rootconstituent_end = constituentsMap.get(root).getEnd();
		/*
		 * Number of words
		 */
		numOfWords = rootconstituent_end - footconstituent_end;
		/*
		 * Setting the static variable in the Feature Object class, So now
		 * everytime we call this method, the variable outsideWordsRight is
		 * changes in the class and any object accessing this variable will feel
		 * the change, Object Independent variable
		 */

		OutsideFootNumwordsright.outsideWordsRight = numOfWords;

	}

	/**
	 * Have to call this method from inside the while loop that iterates over
	 * the nodes in the tree, because we will get a different value for each
	 * node
	 * 
	 * @param insideTree
	 * @param constituentsMap
	 * @param root
	 */
	public static void getNumberOfOutsideWordsLeft(Tree<String> insideTree,
			Map<Tree<String>, Constituent<String>> constituentsMap, Tree<String> root) {

		int numOfWords = 0;
		/*
		 * Getting the start of the inside tree
		 */
		int footconstituent_start = constituentsMap.get(insideTree).getStart();
		/*
		 * Getting the end of the sentence
		 */
		int rootconstituent_start = constituentsMap.get(root).getStart();
		/*
		 * Number of words
		 */
		numOfWords = footconstituent_start - rootconstituent_start;
		/*
		 * Setting the static variable in the Feature Object class, So now every
		 * time we call this method, the variable outsideWordsRight is changes
		 * in the class and any object accessing this variable will feel the
		 * change, Object Independent variable
		 */
		OutsideFootNumwordsleft.outsideWordsLeft = numOfWords;

	}

	/**
	 * The method to get the outside feature vector dimensions
	 * 
	 * @return
	 */
	public static int getOutsideFeatureVectorDimensions(ArrayList<Alphabet> updatedFilteredDictionary) {

		int vectorDimension = 0;

		/*
		 * Getting the vector dimension for the inside feature vector phi. The
		 * dimensionality is just equal to the sum of all the inside features
		 * that have been extracted from the corpus
		 */
		for (Alphabet dictionary : updatedFilteredDictionary) {
			vectorDimension += dictionary.size();
		}

		return vectorDimension;

	}

	/**
	 * 
	 * @param updatedFilteredDictionary
	 * @return
	 */
	public static int getWordDictionarySize(Alphabet wordDictionary) {

		return wordDictionary.size();

	}

	/**
	 * Return an array containing all the files in a directory. Needed to
	 * iterate over the files in the BLLIP Corpus
	 * 
	 * @param directoryRoot
	 * @return
	 */
	public static File[] getFiles(String directoryRoot) {
		File[] files = new File(directoryRoot).listFiles();
		return files;
	}

	/**
	 * 
	 * @param root
	 * @return
	 */
	public static ArrayList<String> getBLLIPCorpusFilePaths(String root) {

		ArrayList<String> corpusFilePaths = new ArrayList<String>();
		File[] files = new File(root).listFiles(new FileFilter() {

			@Override
			public boolean accept(File pathname) {
				return !pathname.isHidden();
			}
		});

		/*
		 * We know that all the BLLLIP corpus files are one level down
		 */
		for (File file : files) {
			File[] corpusFiles = null;
			if (file.isDirectory()) {
				corpusFiles = file.listFiles(new FileFilter() {
					@Override
					public boolean accept(File pathname) {
						return !pathname.isHidden();
					}
				});
			}

			for (File corpusFile : corpusFiles) {
				corpusFilePaths.add(corpusFile.getAbsolutePath());
			}
		}

		return corpusFilePaths;

	}

	/**
	 * TODO
	 * 
	 * @param bllipCorpusPaths
	 */
	public static void makeTreeCorpusDirectoryStructure(ArrayList<String> bllipCorpusPaths) {
		String directoryPath = null;
		for (String path : bllipCorpusPaths) {
			int index = path.lastIndexOf("/");
			if (index != -1) {
				directoryPath = path.substring(0, index);
			}

			if (directoryPath != null && !directoryPath.isEmpty()) {
				File dir = new File(directoryPath.replaceAll("data", "data_extracted_trees"));
				if (!dir.exists()) {
					dir.mkdirs();
				}

			}
		}

	}

	/**
	 * The method extracts the tree from the given file path as argument and
	 * adds it to the file tree.txt
	 * 
	 * @param URI
	 * @throws IOException
	 * @throws CompressorException
	 */
	public static void extractAndAddTrees(String URI) {
		fileNum = fileNum + 1;
		// System.out.println(treeCount);
		// 4 million trees
		if (treeCount == 4000000) {
			System.out.println("4 million tree extracted and now exiting");
			System.exit(0);
		}

		File file = null;

		/*
		 * Forming the appropriate directory structure
		 */

		// if (fileNum <= 5000) {
		// file = new File(VSMContant.PARSED_TREE_CORPUS + "/tress_" + index
		// + "/trees_" + fileNum + ".txt");
		// } else {
		//
		// index++;
		// fileNum = 0;
		// fileNum = fileNum + 1;
		// file = new File(VSMContant.PARSED_TREE_CORPUS + "/tress_" + index
		// + "/trees_" + fileNum + ".txt");
		// }

		// file = new File(VSMContant.PARSED_TREE_CORPUS_ALL + "/corpus.txt");

		BufferedReader brZ = null;
		try {
			brZ = BLLIPCorpusReader.getBufferedReaderForBZ2File(URI);
		} catch (FileNotFoundException | CompressorException e) {
			System.out.println("++++Exception while decompressing the BLLIP CORPUS FILE+++");
			e.printStackTrace();
		}

		if (!file.exists()) {
			file.getParentFile().mkdirs();
		}

		FileWriter fw = null;
		try {
			fw = new FileWriter(file.getAbsoluteFile(), true);
		} catch (IOException e) {
			e.printStackTrace();
		}

		BufferedWriter bw = new BufferedWriter(fw);

		String line = null;
		boolean flag = false;
		int count = 0;
		int count1 = 0;

		try {
			while ((line = brZ.readLine()) != null) {

				String beginning = line.substring(0, 2);
				if (beginning.equalsIgnoreCase("50") && count1 == 0 && !line.contains("-")) {
					flag = true;
					count1 = 1;
				}

				if ((flag == true) && (line.charAt(0) == '(')) {
					// System.out.println(treeCount);
					treeCount++;
					bw.write(line);
					bw.newLine();
					bw.newLine();
					count++;
				}

				if (count == 1) {
					flag = false;
					count = 0;
					count1 = 0;
				}
			}
		} catch (IOException e) {
			System.out.println("+++EXCEPTION WHILE WRITING TO THE FILE+++");
			e.printStackTrace();
		} finally {
			try {
				bw.flush();
				bw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}

		}

	}

	public static ArrayList<String> extractTrees(String URI) {
		fileNum = fileNum + 1;

		BufferedReader brZ = null;

		String line = null;
		boolean flag = false;
		int count = 0;
		int count1 = 0;
		ArrayList<String> trees = new ArrayList<String>();

		try {
			brZ = BLLIPCorpusReader.getBufferedReaderForBZ2File(URI);
			while ((line = brZ.readLine()) != null) {

				String beginning = line.substring(0, 2);
				if (beginning.equalsIgnoreCase("50") && count1 == 0 && !line.contains("-")) {
					flag = true;
					count1 = 1;
				}

				if ((flag == true) && (line.charAt(0) == '(')) {
					trees.add(line);
					count++;
				}

				if (count == 1) {
					flag = false;
					count = 0;
					count1 = 0;
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} catch (CompressorException e) {
			e.printStackTrace();
		} finally {
			try {
				if (brZ != null)
					brZ.close();
			} catch (IOException e) {

				e.printStackTrace();
			}
		}

		return trees;

	}

	public static void extractAndAddTreesCorpusSpecific(String URI, String userPref) {
		fileNum = fileNum + 1;
		// System.out.println(treeCount);
		// 4 million trees
		if (treeCount == 4000000) {
			System.out.println("4 million tree extracted and now exiting");
			System.exit(0);
		}

		File file = null;

		/*
		 * Forming the appropriate directory structure
		 */

		// if (fileNum <= 5000) {
		// file = new File(VSMContant.PARSED_TREE_CORPUS + "/tress_" + index
		// + "/trees_" + fileNum + ".txt");
		// } else {
		//
		// index++;
		// fileNum = 0;
		// fileNum = fileNum + 1;
		// file = new File(VSMContant.PARSED_TREE_CORPUS + "/tress_" + index
		// + "/trees_" + fileNum + ".txt");
		// }

		// file = new File(VSMContant.PARSED_TREE_CORPUS_ALL + "/corpus" +
		// userPref + ".txt");

		BufferedReader brZ = null;
		try {
			brZ = BLLIPCorpusReader.getBufferedReaderForBZ2File(URI);
		} catch (FileNotFoundException | CompressorException e) {
			System.out.println("++++Exception while decompressing the BLLIP CORPUS FILE+++");
			e.printStackTrace();
		}

		if (!file.exists()) {
			file.getParentFile().mkdirs();
		}

		FileWriter fw = null;
		try {
			fw = new FileWriter(file.getAbsoluteFile(), true);
		} catch (IOException e) {
			e.printStackTrace();
		}

		BufferedWriter bw = new BufferedWriter(fw);

		String line = null;
		boolean flag = false;
		int count = 0;
		int count1 = 0;

		try {
			while ((line = brZ.readLine()) != null) {

				String beginning = line.substring(0, 2);
				if (beginning.equalsIgnoreCase("50") && count1 == 0 && !line.contains("-")) {
					flag = true;
					count1 = 1;
				}

				if ((flag == true) && (line.charAt(0) == '(')) {
					// System.out.println(treeCount);
					treeCount++;
					bw.write(line);
					bw.newLine();
					bw.newLine();
					count++;
				}

				if (count == 1) {
					flag = false;
					count = 0;
					count1 = 0;
				}
			}
		} catch (IOException e) {
			System.out.println("+++EXCEPTION WHILE WRITING TO THE FILE+++");
			e.printStackTrace();
		} finally {
			try {
				bw.flush();
				bw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}

		}

	}

	/**
	 * The method returns the file paths corresponding to all the directories in
	 * the corpus. We can iterate over these paths to extract syntax trees
	 * 
	 * @param files
	 * @return
	 */
	public static ArrayList<File> getFilePaths(File[] files) {

		for (File file : files) {
			filePaths.addAll(sort(file));
		}
		return filePaths;
	}

	public static ArrayList<File> sort(File file1) {

		MultiValueMap map = new MultiValueMap();
		File[] files = file1.listFiles(new FileFilter() {
			@Override
			public boolean accept(File pathname) {
				return !pathname.isHidden();
			}
		});

		ArrayList<File> sortedFiles = new ArrayList<File>();

		for (File file : files) {
			String[] tokens = file.getName().split("-");

			for (int i = 0; i < tokens.length - 1; i++) {
				map.put(tokens[0], tokens[i + 1]);

			}
		}

		List<String> list1 = new ArrayList<String>(map.keySet());
		Collections.sort(list1, new Comparator<String>() {
			public int compare(String string1, String string2) {
				String a = string1.replaceFirst("^0+(?!$)", "");
				String b = string2.replaceFirst("^0+(?!$)", "");
				return Integer.parseInt(a) - Integer.parseInt(b);
			}
		});

		for (Object i : list1) {
			String s2 = i.toString();
			for (Object s : map.getCollection(i)) {
				String s1 = s.toString();
				s2 = s2 + "-" + s1;
			}

			sortedFiles.add(new File(file1.getAbsolutePath() + "/" + s2));
		}

		return sortedFiles;
	}

	public static ArrayList<File> getFilePathsCorpusSpecific(File[] files, String userPref) {

		for (File file : files) {
			filePaths.addAll(sortCorpusSpec(file, userPref));
		}
		return filePaths;
	}

	public static ArrayList<File> sortCorpusSpec(File file1, String userPref) {

		MultiValueMap map = new MultiValueMap();
		File[] files = file1.listFiles(new FileFilter() {
			@Override
			public boolean accept(File pathname) {
				return !pathname.isHidden();
			}
		});

		ArrayList<File> sortedFiles = new ArrayList<File>();

		for (File file : files) {
			String[] tokens = file.getName().split("-");
			if (tokens[1].startsWith(userPref)) {
				for (int i = 0; i < tokens.length - 1; i++) {
					map.put(tokens[0], tokens[i + 1]);

				}
			}
		}

		List<String> list1 = new ArrayList<String>(map.keySet());
		Collections.sort(list1, new Comparator<String>() {
			public int compare(String string1, String string2) {
				String a = string1.replaceFirst("^0+(?!$)", "");
				String b = string2.replaceFirst("^0+(?!$)", "");
				return Integer.parseInt(a) - Integer.parseInt(b);
			}
		});

		for (Object i : list1) {
			String s2 = i.toString();
			for (Object s : map.getCollection(i)) {
				String s1 = s.toString();
				s2 = s2 + "-" + s1;
			}

			sortedFiles.add(new File(file1.getAbsolutePath() + "/" + s2));
		}

		return sortedFiles;
	}

	/**
	 * The method that returns the dictionary size given a dictionary
	 * 
	 * @param insideFeatureDictionary
	 * @return
	 */
	public static long getDictionarySize(ArrayList<Alphabet> insideFeatureDictionary) {
		long size = 0;
		for (Alphabet dictionary : insideFeatureDictionary) {
			size = size + dictionary.size();
		}
		return size;
	}

	/**
	 * Utility method to covert sparse matrix to dense matrix
	 * 
	 * @param featureMatrix
	 * @return
	 */
	public static Matrix createDenseMatrixJAMA(SparseMatrixLil featureMatrix) {
		DenseMatrix matrix = featureMatrix.toDense();
		Matrix x = new Matrix(featureMatrix.rows, featureMatrix.cols);
		for (int i = 0; i < featureMatrix.rows; i++) {
			for (int j = 0; j < featureMatrix.cols; j++) {
				x.set(i, j, matrix.get(i, j));
			}
		}

		return x;
	}

	/**
	 * The method to serialize the inside and outside matrices
	 * 
	 * @param opt
	 * @return
	 * @throws ClassNotFoundException
	 */
	public static Object[] deserializeCCAVariantsRun(String directoryName) throws ClassNotFoundException {

		Object[] matrixObj = new Object[3];

		String fileDirPath = "/afs/inf.ed.ac.uk/group/project/vsm.restored/syntacticprojectionserobjects/"
				+ directoryName;
		File fileDir = new File(fileDirPath);
		if (fileDir.exists()) {
			String fileName = fileDir.getAbsolutePath() + "/projectionInside.ser";
			String fileName1 = fileDir.getAbsolutePath() + "/projectionOutside.ser";

			Matrix Y = null, Z = null;

			try {

				ObjectInput y = new ObjectInputStream(new FileInputStream(fileName));
				ObjectInput z = new ObjectInputStream(new FileInputStream(fileName1));

				Y = (Matrix) y.readObject();
				Z = (Matrix) z.readObject();

				System.out.println("=======De-serialized the CCA Variant Run=======");
			} catch (IOException ioe) {
				System.out.println(ioe.getMessage());
			}
			matrixObj[0] = (Object) Y;
			matrixObj[1] = (Object) Z;
			matrixObj[2] = null;

			return matrixObj;
		} else {

			System.out.println(
					"***There is no such non-terminal for which we can get the projections***" + directoryName);
			return null;
		}

	}

	public static Object[] deserializeCCAVariantsRunSem(String directoryName) throws ClassNotFoundException {

		Object[] matrixObj = new Object[3];

		String fileDirPath = "/afs/inf.ed.ac.uk/group/project/vsm.restored/semanticprojectionserobjects/"
				+ directoryName;
		File fileDir = new File(fileDirPath);
		if (fileDir.exists()) {
			String fileName = fileDir.getAbsolutePath() + "/projectionInside.ser";
			String fileName1 = fileDir.getAbsolutePath() + "/projectionOutside.ser";

			Matrix Y = null, Z = null;

			try {

				ObjectInput y = new ObjectInputStream(new FileInputStream(fileName));
				ObjectInput z = new ObjectInputStream(new FileInputStream(fileName1));

				Y = (Matrix) y.readObject();
				Z = (Matrix) z.readObject();

				System.out.println("=======De-serialized the CCA Variant Run=======");
			} catch (IOException ioe) {
				System.out.println(ioe.getMessage());
			}
			matrixObj[0] = (Object) Y;
			matrixObj[1] = (Object) Z;
			matrixObj[2] = null;

			return matrixObj;
		} else {

			System.out.println(
					"***There is no such non-terminal for which we can get the projections***" + directoryName);
			return null;
		}

	}

	/**
	 * Method that writes the projection matrix Z(Inside) in a file
	 * 
	 * @param matrices
	 * @param name
	 * @param count
	 * @throws IOException
	 */
	public static void writeEigenDictInside(Object[] matrices, String name) {
		DenseDoubleMatrix2D dictLMatrix = createDenseMatrixCOLT((Matrix) matrices[1]);
		double[][] dictL = dictLMatrix.toArray();
		BufferedWriter writer = null;
		String eigenDict = "/afs/inf.ed.ac.uk/group/project/vsm.restored/syntacticprojectionstxt/" + name + "_YSyn.txt";
		try {
			writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(eigenDict), "UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		for (int i = 0; i < dictLMatrix.columns(); i++) {

			try {

				for (int j = 0; j < dictLMatrix.rows(); j++) {

					if (j != dictLMatrix.columns() - 1) {
						writer.write(Double.toString(dictL[i][j]));
						writer.write(' ');
					} else {
						writer.write(Double.toString(dictL[i][j]));
						writer.write('\n');
					}
				}

				writer.close();
			} catch (IOException e) {

				e.printStackTrace();
			}
		}

	}

	/**
	 * Method that writes the projection matrix Z in a file
	 * 
	 * @param matrices
	 * @param name
	 * @param count
	 * @throws IOException
	 */
	public static void writeEigenDictInsideSemantic(Object[] matrices, String name) {
		DenseDoubleMatrix2D dictLMatrix = createDenseMatrixCOLT((Matrix) matrices[1]);
		double[][] dictL = dictLMatrix.toArray();
		BufferedWriter writer = null;
		String eigenDict = "/afs/inf.ed.ac.uk/group/project/vsm-afs/projectionMatrices/" + name + "/" + name
				+ "_Ysem.txt";
		try {
			writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(eigenDict), "UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		for (int i = 0; i < dictLMatrix.rows(); i++) {
			try {

				for (int j = 0; j < dictLMatrix.columns(); j++) {

					if (j != dictLMatrix.columns() - 1) {
						writer.write(Double.toString(dictL[i][j]));
						writer.write(' ');
					} else {
						writer.write(Double.toString(dictL[i][j]));
						writer.write('\n');
					}
				}

				writer.close();
			} catch (IOException e) {

				e.printStackTrace();
			}
		}

	}

	/**
	 * 
	 * @param matrices
	 * @param name
	 * @param count
	 * @throws IOException
	 */
	public static void writeEigenDictOutside(Object[] matrices, String name) {
		DenseDoubleMatrix2D dictLMatrix = createDenseMatrixCOLT((Matrix) matrices[0]);
		double[][] dictL = dictLMatrix.toArray();
		BufferedWriter writer = null;
		String eigenDict = "/afs/inf.ed.ac.uk/group/project/vsm-afs/projectionMatrices/" + name + "/" + name
				+ "_Zsyn.txt";
		try {
			writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(eigenDict, false), "UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		for (int i = 0; i < dictLMatrix.rows(); i++) {
			try {

				for (int j = 0; j < dictLMatrix.columns(); j++) {

					if (j != dictLMatrix.columns() - 1) {
						writer.write(Double.toString(dictL[i][j]));
						writer.write(' ');
					} else {
						writer.write(Double.toString(dictL[i][j]));
						writer.write('\n');
					}
				}

				writer.close();
			} catch (IOException e) {

				e.printStackTrace();
			}
		}

	}

	public static void writeProjectionMatrix(Object[] matrices, String nonTerminal, String fileName, int index) {
		DenseDoubleMatrix2D dictLMatrix = createDenseMatrixCOLT((Matrix) matrices[index]);
		double[][] dictL = dictLMatrix.toArray();
		BufferedWriter writer = null;
		String eigenDict = "/afs/inf.ed.ac.uk/group/project/vsm-afs/projectionMatrices/" + nonTerminal + "/"
				+ nonTerminal + fileName + ".txt";
		try {
			writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(eigenDict, false), "UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		for (int i = 0; i < dictLMatrix.rows(); i++) {
			try {
				for (int j = 0; j < dictLMatrix.columns(); j++) {

					if (j != dictLMatrix.columns() - 1) {
						writer.write(Double.toString(dictL[i][j]));
						writer.write(' ');
					} else {
						writer.write(Double.toString(dictL[i][j]));
						writer.write('\n');
					}
				}

			} catch (IOException e) {

				e.printStackTrace();
			}
		}

		try {
			writer.close();
		} catch (IOException e) {

			e.printStackTrace();
		}

	}

	public static DenseDoubleMatrix2D createDenseMatrixCOLT(Matrix xJama) {
		DenseDoubleMatrix2D x_omega = new DenseDoubleMatrix2D(xJama.getRowDimension(), xJama.getColumnDimension());
		for (int i = 0; i < xJama.getRowDimension(); i++) {
			for (int j = 0; j < xJama.getColumnDimension(); j++) {
				x_omega.set(i, j, xJama.get(i, j));
			}
		}
		return x_omega;
	}

	/*
	 * Some text pre-processing being done here, nothing fancy!
	 */
	@SuppressWarnings("unchecked")
	public static ArrayList<String> normalize(List<String> wordList) {
		ArrayList<String> norm = new ArrayList<String>();
		Iterator<String> itNorm = wordList.iterator();
		while (itNorm.hasNext()) {
			String s1 = itNorm.next();
			if (s1.matches("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+"))
				norm.add("<num>");
			else
				norm.add(s1);
		}

		return (ArrayList<String>) norm.clone();
	}

	/**
	 * Method to add the words to the dictionary
	 * 
	 * @param wordList
	 */
	public static void addToDictionary(List<String> wordList, Alphabet wordDictionary) {

		/*
		 * Adding words to the dictionary
		 */
		for (String word : wordList) {
			wordDictionary.lookupIndex(word);
		}
	}

	@SuppressWarnings("unchecked")
	public static ArrayList<String> lowercase(List<String> wordList) {
		ArrayList<String> lower = new ArrayList<String>();
		Iterator<String> itlower = wordList.iterator();
		while (itlower.hasNext())
			lower.add(itlower.next().toLowerCase().trim());

		return (ArrayList<String>) lower.clone();
	}

	/**
	 * Removes duplicates from the array if we do not want counts
	 * 
	 * @param arr
	 * @return
	 */
	public static int[] removeDuplicates(int[] arr) {
		Set<Integer> alreadyPresent = new HashSet<>();
		int[] whitelist = new int[arr.length];
		int i = 0;

		for (int element : arr) {
			if (alreadyPresent.add(element)) {
				whitelist[i++] = element;
			}
		}

		return Arrays.copyOf(whitelist, i);
	}

	/**
	 * Create dense matrix MTJ from JAMA
	 * 
	 * @param xomega
	 * @return
	 */
	public static no.uib.cipr.matrix.DenseMatrix createDenseMatrixMTJ(Matrix xomega) {
		no.uib.cipr.matrix.DenseMatrix xMTJ = new no.uib.cipr.matrix.DenseMatrix(xomega.getRowDimension(),
				xomega.getColumnDimension());

		for (int i = 0; i < xomega.getRowDimension(); i++) {
			for (int j = 0; j < xomega.getColumnDimension(); j++) {
				xMTJ.set(i, j, xomega.get(i, j));
			}
		}
		return xMTJ;
	}

	/**
	 * Getting the norm of a vector used for cosine similarity
	 * 
	 * @param data
	 * @return
	 */
	public static double norm2(double[] data) {
		double norm = 0;
		for (int i = 0; i < data.length; ++i)
			norm += data[i] * data[i];

		return Math.sqrt(norm);
	}

	/**
	 * Getting the gold standard similarity scores
	 * 
	 * @return
	 */
	public static HashMap<Integer, Double> getGoldStandard() {
		String sickTrainingSet = "/group/project/vsm-afs/SICK_trial.txt";

		// String sickTrainingSet =
		// "/Users/sameerkhurana10/training_corpus/SICK_train.txt";

		HashMap<Integer, Double> goldStandard = new LinkedHashMap<Integer, Double>();

		Pattern regex = Pattern.compile("(\\d+(?:\\.\\d+)?)");

		BufferedReader br = null;

		try {
			String sCurrentLine;

			br = new BufferedReader(new FileReader(sickTrainingSet));

			int count = 0;
			while ((sCurrentLine = br.readLine()) != null) {

				// System.out.println("***Reading the file****");
				count++;
				/*
				 * Ignoring the first line
				 */
				if (count > 1) {
					// System.out.println(sCurrentLine);

					Matcher matcher = regex.matcher(new StringBuilder(sCurrentLine).reverse());

					// int i = 1;
					inner: while (matcher.find()) {

						// System.out.println(matcher.group());

						goldStandard.put((count - 1),
								Double.parseDouble(new StringBuffer(matcher.group()).reverse().toString()));

						break inner;

					}

				}
			}
		} catch (IOException e) {

			e.printStackTrace();

		} finally {

			try {

				if (br != null)
					br.close();

			} catch (IOException ex) {

				ex.printStackTrace();

			}
		}

		// System.out.println("****Gold Standard***" + goldStandard);

		return goldStandard;
	}

	/**
	 * Getting the gold standard similarity scores
	 * 
	 * @return
	 */
	public static HashMap<Integer, Double> getGoldStandardTrial() {
		String sickTrialSet = "/disk/gpfs/scohen/embeddings/datasets/dsm/SICK_trial.txt";

		// String sickTrainingSet =
		// "/Users/sameerkhurana10/training_corpus/SICK_train.txt";

		HashMap<Integer, Double> goldStandard = new LinkedHashMap<Integer, Double>();

		Pattern regex = Pattern.compile("(\\d+(?:\\.\\d+)?)");

		BufferedReader br = null;

		try {
			String sCurrentLine;

			br = new BufferedReader(new FileReader(sickTrialSet));

			int count = 0;
			while ((sCurrentLine = br.readLine()) != null) {

				// System.out.println("***Reading the file****");
				count++;
				/*
				 * Ignoring the first line
				 */
				if (count > 1) {
					// System.out.println(sCurrentLine);

					Matcher matcher = regex.matcher(new StringBuilder(sCurrentLine).reverse());

					// int i = 1;
					inner: while (matcher.find()) {

						// System.out.println(matcher.group());

						goldStandard.put((count - 1),
								Double.parseDouble(new StringBuffer(matcher.group()).reverse().toString()));

						break inner;

					}

				}
			}
		} catch (IOException e) {

			e.printStackTrace();

		} finally {

			try {

				if (br != null)
					br.close();

			} catch (IOException ex) {

				ex.printStackTrace();

			}
		}

		// System.out.println("****Gold Standard***" + goldStandard);

		return goldStandard;
	}

	// public static void writeFeatureDictionary(VSMDictionaryBean
	// dictionaryBean, String nonTerminal,
	// String dictionaryPath, String dictionaryType) {
	// File file = new File(dictionaryPath + "/" + nonTerminal);
	// if (!file.exists()) {
	// file.mkdirs();
	// }
	//
	// if (dictionaryType.equalsIgnoreCase("inside")) {
	// ArrayList<Alphabet> insideDictionary =
	// dictionaryBean.getInsideFeatureDictionary();
	// PrintWriter writerIn = null;
	// try {
	// writerIn = new PrintWriter(file.getAbsolutePath() + "/insidedict.txt");
	// /*
	// * INside dictionary
	// */
	// for (Alphabet dictionary : insideDictionary) {
	// Object[] features = dictionary.reverseMap.getValues();
	// for (Object feature : features) {
	// String insideFeature = (String) feature;
	// writerIn.println(insideFeature + " " + dictionary.getCount(feature));
	//
	// }
	// }
	// } catch (FileNotFoundException e) {
	// System.out.println("***Exception while writing the dictionary***" + e);
	// } finally {
	// System.out.println("****Written the dictionaries****");
	// writerIn.close();
	// // writerOut.close();
	// }
	// } else if (dictionaryType.equalsIgnoreCase("outside")) {
	// ArrayList<Alphabet> outsideDictionary =
	// dictionaryBean.getOutsideFeatureDictionary();
	// PrintWriter writerOut = null;
	// try {
	//
	// writerOut = new PrintWriter(file.getAbsolutePath() + "/outsidedict.txt");
	// /*
	// * Outside dictionary
	// */
	// for (Alphabet dictionary : outsideDictionary) {
	//
	// Object[] features = dictionary.reverseMap.getValues();
	// for (Object feature : features) {
	// String outsideFeature = (String) feature;
	// writerOut.println(outsideFeature + " " + dictionary.getCount(feature));
	//
	// }
	//
	// }
	//
	// } catch (FileNotFoundException e) {
	// System.out.println("***Exception while writing the dictionary***" + e);
	// } finally {
	// System.out.println("****Written the dictionaries****");
	// // writerIn.close();
	// writerOut.close();
	// }
	// }
	//
	// }

	/**
	 * A useful function here
	 * 
	 * @param xjeig
	 * @return
	 */

	public static SparseMatrixLil createJeigenMatrix(FlexCompRowMatrix xmtj) {
		SparseMatrixLil x = new SparseMatrixLil(xmtj.numRows(), xmtj.numColumns());

		for (MatrixEntry e : xmtj) {
			x.append(e.row(), e.column(), e.get());
		}

		System.out.println(
				"Size:" + " " + xmtj.numRows() + " " + xmtj.numColumns() + " " + xmtj.numRows() * xmtj.numColumns()
						+ " " + x.getSize() + " " + (x.getSize() * 1.0 / xmtj.numRows() / xmtj.numColumns()));

		System.out.println("+++Converted Matrix+++");

		return x;
	}

	public static void writeSparseMatrixToDisk(SparseMatrixLil sparseMatrix, String matrixTextFile, String nonTerminal,
			org.apache.log4j.Logger logger) {

		BufferedWriter bw = null;
		// BufferedWriter bwl = null;
		try {

			for (int i = 0; i < sparseMatrix.getSize(); i++) {

				bw = new BufferedWriter(new FileWriter(matrixTextFile, true));
				// bwl = new BufferedWriter(new FileWriter(matrixTextFile +
				// ".log", true));
				// +1 for matlab
				int rowidx = sparseMatrix.getRowIdx(i) + 1;
				int colidx = sparseMatrix.getColIdx(i) + 1;
				double value = sparseMatrix.getValue(i);
				bw.write(Integer.toString(rowidx) + "\t" + Integer.toString(colidx) + "\t" + Double.toString(value)
						+ "\n");
				// bwl.write(Integer.toString(rowidx) + "\t" +
				// Integer.toString(colidx) + "\t"
				// + Double.toString(Math.log(value)) + "\n");
				bw.flush();
				bw.close();
				// bwl.flush();
				// bwl.close();
			}

			// The last line is added to tell Octave about the matrix
			// dimensions: Very important this
			bw = new BufferedWriter(new FileWriter(matrixTextFile, true));
			bw.write(Integer.toString(sparseMatrix.rows) + "\t" + Integer.toString(sparseMatrix.cols) + "\t"
					+ Double.toString(0.0));
			//
			// bwl = new BufferedWriter(new FileWriter(matrixTextFile + ".log",
			// true));
			// bwl.write(Integer.toString(sparseMatrix.rows) + "\t" +
			// Integer.toString(sparseMatrix.cols) + "\t"
			// + Double.toString(0.0));

			bw.flush();
			bw.close();
			// bwl.flush();
			// bwl.close();

		} catch (IOException e) {
			logger.error("Error while writing the sparse matrix to disk: " + e);
		}

	}

	public static void writeCovarMatrixSem(SparseMatrixLil psiTPsi, String nonTerminal) {
		id++;
		String filePath = "/afs/inf.ed.ac.uk/group/project/vsm.restored/covarssem/" + nonTerminal + "/" + "covar_" + id
				+ ".txt";
		File file = new File(filePath);
		if (!file.exists()) {
			file.getParentFile().mkdirs();
		}

		PrintWriter writer = null;

		try {
			writer = new PrintWriter(file);
			for (int i = 0; i < psiTPsi.getSize(); i++) {
				String s = psiTPsi.getRowIdx(i) + " " + psiTPsi.getColIdx(i) + " " + psiTPsi.getValue(i);
				writer.println(s);

			}

		} catch (FileNotFoundException e) {
			System.out.println("***Exception while writing to the file***" + e);
		} finally {
			writer.close();
		}

	}

	/**
	 * Writing singular values to a file
	 * 
	 * @param s
	 * @param nonTerminal
	 */
	public static void writeSingularValues(double[] s, String nonTerminal, String fileName, int dimensions) {
		String filePath = "/afs/inf.ed.ac.uk/group/project/vsm-afs/projectionMatrices/" + nonTerminal + "/"
				+ nonTerminal + "_" + fileName + "_sigma.txt";

		File file = new File(filePath);
		if (!file.exists()) {
			file.getParentFile().mkdirs();
		}

		PrintWriter writer = null;
		try {
			writer = new PrintWriter(file);
			for (int i = 0; i < dimensions; i++) {

				/*
				 * Getting the singular values and writing to a file
				 */
				double sigma = s[i];
				writer.println(Double.toString(sigma));

			}

		} catch (FileNotFoundException e) {
			System.out.println("***Exception while writing Singular values to the file to the file***" + e);
		} finally {
			writer.close();
		}

	}

	/**
	 * Writing singular values to a file
	 * 
	 * @param s
	 * @param nonTerminal
	 */
	public static void writeSingularValuesSem(double[] s, String nonTerminal) {
		String filePath = "/afs/inf.ed.ac.uk/group/project/vsm.restored/covarssem/" + nonTerminal + "/sigma.txt";

		File file = new File(filePath);
		if (!file.exists()) {
			file.getParentFile().mkdirs();
		}
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(file);
			for (int i = 0; i < s.length; i++) {

				/*
				 * Getting the singular values and writing to a file
				 */
				double sigma = s[i];
				writer.println(Double.toString(sigma));

			}

		} catch (FileNotFoundException e) {
			System.out.println("***Exception while writing Singular values to the file to the file***" + e);
		} finally {
			writer.close();
		}

	}

	public static void createMatFileProjections(String nonTerminal) {

		/*
		 * Getting the inside and outside projection matrices
		 */
		Object[] matrices = null;
		try {
			matrices = CommonUtil.deserializeCCAVariantsRun(nonTerminal);
		} catch (ClassNotFoundException e1) {
			System.out.println("**Exception while deserializing***" + e1);
		}

		/*
		 * Getting the inside projection matrix
		 */
		DenseDoubleMatrix2D dictLMatrix = CommonUtil.createDenseMatrixCOLT((Matrix) matrices[0]);

		/*
		 * Getting the inside projection matrix
		 */
		DenseDoubleMatrix2D dictRMatrix = CommonUtil.createDenseMatrixCOLT((Matrix) matrices[1]);

		double[][] dictL = dictLMatrix.toArray();
		double[][] dictR = dictRMatrix.toArray();

		String matVarNameIn = "Z" + nonTerminal;
		String matVarNameOut = "Y" + nonTerminal;
		MLDouble dictInside = new MLDouble(matVarNameIn, dictL);
		MLDouble dictOutside = new MLDouble(matVarNameOut, dictR);

		ArrayList list = new ArrayList();
		list.add(dictInside);
		list.add(dictOutside);

		try {
			new MatFileWriter("/afs/inf.ed.ac.uk/group/project/vsm/matfiles/projections" + nonTerminal + ".mat", list);
		} catch (IOException e) {
			System.out.println("***Exception while making the mat file***");
		}

	}

	// public static Alphabet filterWordDictionary(Alphabet wordDictionary,
	// Alphabet filteredWordDictionary) {
	//
	// Object[] features = wordDictionary.map.keys();
	//
	// for (Object feature : features) {
	// String insideFeature = (String) feature;
	//
	// int featureFreq = wordDictionary.countMap.get(insideFeature);
	//
	// if (featureFreq >= VSMContant.WORD_THRESHOLD_FREQUENCY) {
	//
	// filteredWordDictionary.lookupIndex(insideFeature);
	//
	// filteredWordDictionary.countMap.put(insideFeature, featureFreq);
	// }
	//
	// }
	//
	// return filteredWordDictionary;
	// }

	/**
	 * Writing singular values to a file
	 * 
	 * @param s
	 * @param nonTerminal
	 */
	// public static void writeWordDictionary(Alphabet wordDictionary) {
	// String filePath = VSMContant.WORD_DICT_TXT;
	// File file = new File(filePath);
	// if (!file.exists()) {
	// try {
	// file.createNewFile();
	// } catch (IOException e) {
	// System.out.println("Error while writing the dictionary to the text file
	// at: " + filePath);
	// e.printStackTrace();
	// }
	// }
	//
	// PrintWriter writer = null;
	// Object[] features = wordDictionary.map.keys();
	//
	// try {
	//
	// writer = new PrintWriter(file);
	//
	// for (Object feature : features) {
	// String word = (String) feature;
	// writer.println(word);
	// }
	//
	// } catch (FileNotFoundException e) {
	// System.out.println("Excpetion while writing words to the file" + e);
	// } finally {
	// writer.close();
	// }
	//
	// }

	/**
	 * Returns a pseudo-random number between min and max, inclusive. The
	 * difference between min and max can be at most
	 * <code>Integer.MAX_VALUE - 1</code>.
	 *
	 * @param min
	 *            Minimum value
	 * @param max
	 *            Maximum value. Must be greater than min.
	 * @return Integer between min and max, inclusive.
	 * @see java.util.Random#nextInt(int)
	 */
	public static double randomDouble(Random r, int rangeMin, int rangeMax) {

		double randomValue = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
		return randomValue;
	}

	/**
	 * 
	 * @param docVector1
	 * @param docVector2
	 * @return
	 */
	public static double cosineSimilarity(double[] docVector1, double[] docVector2) {
		double dotProduct = 0.0;
		double magnitude1 = 0.0;
		double magnitude2 = 0.0;
		double cosineSimilarity = 0.0;

		for (int i = 0; i < docVector1.length; i++) // docVector1 and docVector2
													// must be of same length
		{
			dotProduct += docVector1[i] * docVector2[i]; // a.b
			magnitude1 += Math.pow(docVector1[i], 2); // (a^2)
			magnitude2 += Math.pow(docVector2[i], 2); // (b^2)
		}

		magnitude1 = Math.sqrt(magnitude1);// sqrt(a^2)
		magnitude2 = Math.sqrt(magnitude2);// sqrt(b^2)

		if (magnitude1 != 0.0 | magnitude2 != 0.0) {
			cosineSimilarity = dotProduct / (magnitude1 * magnitude2);
		} else {
			return 0.0;
		}
		return (cosineSimilarity * 5);
	}

	public static void createMatFileSentenceEmbedBOW(double[][] sentences, double[][] evaluation, int count) {

		double[][] dictL = sentences;
		double[][] dictR = evaluation;

		String matVarNameIn = "S";
		String matVarNameOut = "SCORE";
		MLDouble dictInside = new MLDouble(matVarNameIn, dictL);
		MLDouble dictOutside = new MLDouble(matVarNameOut, dictR);

		ArrayList list = new ArrayList();
		list.add(dictInside);
		list.add(dictOutside);

		try {
			new MatFileWriter("/disk/scratch/s1444025/SentenceEmbedBOW/matfile.mat", list);
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}

	}

	public static void createMatFileSentenceEmbedBOWTrial(double[][] sentences, double[][] evaluation, int count) {

		double[][] dictL = sentences;
		double[][] dictR = evaluation;

		String matVarNameIn = "S";
		String matVarNameOut = "SCORE";
		MLDouble dictInside = new MLDouble(matVarNameIn, dictL);
		MLDouble dictOutside = new MLDouble(matVarNameOut, dictR);

		ArrayList list = new ArrayList();
		list.add(dictInside);
		list.add(dictOutside);

		try {
			new MatFileWriter("/disk/scratch/s1444025/matfile.mat", list);
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}

	}

	public static void createMatFileSentenceEmbedBOWPhrases(double[][] sentences, double[][] evaluation, int count) {

		double[][] dictL = sentences;
		double[][] dictR = evaluation;

		String matVarNameIn = "S";
		String matVarNameOut = "SCORE";
		MLDouble dictInside = new MLDouble(matVarNameIn, dictL);
		MLDouble dictOutside = new MLDouble(matVarNameOut, dictR);

		ArrayList list = new ArrayList();
		list.add(dictInside);
		list.add(dictOutside);

		try {
			new MatFileWriter("/disk/scratch/s1444025/SentenceEmbedBOWPhrases/matfile.mat", list);
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}

	}

	public static void createMatFileSentenceEmbedBOWPhrasesIn(double[][] sentences, double[][] evaluation, int count) {

		double[][] dictL = sentences;
		double[][] dictR = evaluation;

		String matVarNameIn = "S";
		String matVarNameOut = "SCORE";
		MLDouble dictInside = new MLDouble(matVarNameIn, dictL);
		MLDouble dictOutside = new MLDouble(matVarNameOut, dictR);

		ArrayList list = new ArrayList();
		list.add(dictInside);
		list.add(dictOutside);

		try {
			new MatFileWriter("/disk/scratch/s1444025/SentenceEmbedBOWPhrasesIn/matfile.mat", list);
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}

	}

	public static void createMatFileSentenceEmbedSyn(double[][] sentences, double[][] evaluation, int count) {

		double[][] dictL = sentences;
		double[][] dictR = evaluation;

		String matVarNameIn = "S";
		String matVarNameOut = "SCORE";
		MLDouble dictInside = new MLDouble(matVarNameIn, dictL);
		MLDouble dictOutside = new MLDouble(matVarNameOut, dictR);

		ArrayList list = new ArrayList();
		list.add(dictInside);
		list.add(dictOutside);

		try {
			new MatFileWriter("/disk/scratch/s1444025/SentenceEmbedSyn/matfile.mat", list);
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}

	}

	/*
	 * The method that returns the normalized vector
	 */
	public static Vector normalizeVec(Vector embeddedVec) {

		double[] vecDouble = new double[embeddedVec.size()];
		for (int i = 0; i < embeddedVec.size(); i++) {
			double d = embeddedVec.get(i);
			vecDouble[i] = d;
		}

		double vectorNorm = CommonUtil.norm2(vecDouble);

		if (vectorNorm != 0.0)
			embeddedVec = embeddedVec.scale((double) 1 / (double) vectorNorm);

		return embeddedVec;

	}

	/**
	 * Reading the command line argument
	 * 
	 * @param args
	 * @return
	 */
	public static String getNonTerminal(String[] args) {
		String nonTerminal = null;
		if (args.length > 0) {
			nonTerminal = args[0];
		} else {
			System.out.println("PLEASE GIVE A NON TERMINAL FOR WHICH FEATURE VECTOR EMBEDDINGS ARE TO BE CALCULATED:");
			System.exit(-1);
		}

		return nonTerminal;

	}

	// public static Object[] getSyntacticFeatureDictionaries(String
	// nonTerminal) {
	//
	// Object[] featureDictionaries = new Object[2];
	//
	// String featureDictionarySyn = VSMContant.FEATURE_DICT_PATH +
	// nonTerminal.toLowerCase() + "/dictionary.ser";
	//
	// VSMDictionaryBean dictionaryBean =
	// readSerializedDictionary(featureDictionarySyn, null);
	//
	// /*
	// * Getting the inside and outside feature dictionaries, that are used
	// * for forming the feature vectors
	// */
	// featureDictionaries[0] = dictionaryBean.getOutsideFeatureDictionary();
	// featureDictionaries[1] = dictionaryBean.getInsideFeatureDictionary();
	//
	// return featureDictionaries;
	// }

	// public static Alphabet getWordDictionary() {
	//
	// String featureDictionarySem = VSMContant.WORD_DICT;
	// VSMWordDictionaryBean dictionaryBean =
	// VSMReadSerialWordDict.readSerializedDictionary(featureDictionarySem);
	//
	// Alphabet wordDictionary = dictionaryBean.getWordDictionary();
	//
	// return wordDictionary;
	// }

	// public static File[] getTreeCorpus() {
	//
	// String parsedTreeCorpus = VSMContant.PARSED_TREE_CORPUS;
	//
	// // top level directories that hold the trees.txt files
	// File[] trainingTreesFiles = new File(parsedTreeCorpus).listFiles(new
	// FileFilter() {
	//
	// @Override
	// public boolean accept(File file) {
	// // TODO Auto-generated method stub
	// return !file.isHidden();
	// }
	// });
	//
	// return trainingTreesFiles;
	// }

	public static Object[] getTransposedProjectionMatrices(String nonTerminal, Logger logger) {

		Object[] projectionMatricesT = new Object[4];

		Object[] matricesSyn = null;
		Object[] matricesSem = null;

		/*
		 * The syntactic projection matrices
		 */
		try {
			matricesSyn = CommonUtil.deserializeCCAVariantsRun(nonTerminal);
		} catch (ClassNotFoundException exception) {
			logger.log(Level.SEVERE, "Exception While deserializeCCAVariantsRun" + exception);
			exception.printStackTrace();
		}

		/*
		 * Semantic projection matrices
		 */
		try {
			matricesSem = CommonUtil.deserializeCCAVariantsRunSem(nonTerminal);
		} catch (ClassNotFoundException e) {
			logger.log(Level.SEVERE, "Exception While deserializeCCAVariantsRun" + e);
			e.printStackTrace();
		}

		Matrix Ysyn = (Matrix) matricesSyn[0];
		Matrix Ysem = (Matrix) matricesSem[0];

		/*
		 * Dense Matrix that holds YT
		 */
		no.uib.cipr.matrix.DenseMatrix YTSyn = new no.uib.cipr.matrix.DenseMatrix(Ysyn.getColumnDimension(),
				Ysyn.getRowDimension());

		no.uib.cipr.matrix.DenseMatrix YTSem = new no.uib.cipr.matrix.DenseMatrix(Ysem.getColumnDimension(),
				Ysem.getRowDimension());

		/*
		 * Getting the MTJ Matrix
		 */
		no.uib.cipr.matrix.DenseMatrix YMTJSyn = CommonUtil.createDenseMatrixMTJ(Ysyn);

		no.uib.cipr.matrix.DenseMatrix YMTJSem = CommonUtil.createDenseMatrixMTJ(Ysem);

		Ysyn = null;
		Ysem = null;

		/*
		 * Transform
		 */
		YMTJSyn.transpose(YTSyn);
		YMTJSem.transpose(YTSem);

		YMTJSyn = null;

		/*
		 * Inside Projection Matrix
		 */
		Matrix ZSyn = (Matrix) matricesSyn[1];
		Matrix ZSem = (Matrix) matricesSem[1];

		/*
		 * Dense Matrix that holds YT
		 */
		no.uib.cipr.matrix.DenseMatrix ZTSyn = new no.uib.cipr.matrix.DenseMatrix(ZSyn.getColumnDimension(),
				ZSyn.getRowDimension());
		no.uib.cipr.matrix.DenseMatrix ZTSem = new no.uib.cipr.matrix.DenseMatrix(ZSem.getColumnDimension(),
				ZSem.getRowDimension());

		/*
		 * Getting the MTJ Matrix
		 */
		no.uib.cipr.matrix.DenseMatrix ZMTJSyn = createDenseMatrixMTJ(ZSyn);
		no.uib.cipr.matrix.DenseMatrix ZMTJSem = createDenseMatrixMTJ(ZSem);

		ZSyn = null;
		ZSem = null;

		/*
		 * Transform
		 */
		ZMTJSyn.transpose(ZTSyn);
		ZMTJSem.transpose(ZTSem);
		ZMTJSyn = null;
		ZMTJSem = null;

		projectionMatricesT[0] = YTSyn;
		projectionMatricesT[1] = YTSem;
		projectionMatricesT[2] = ZTSyn;
		projectionMatricesT[3] = ZTSem;

		return projectionMatricesT;
	}

	public static Iterator<Tree<String>> getTreeIterator(Tree<String> syntaxTree) {

		return syntaxTree.iterator();
	}

	public static List<String> getProcessedWordList(Tree<String> insideTree) {

		List<String> wordList = insideTree.getTerminalYield();

		wordList = CommonUtil.lowercase(wordList);

		return wordList;
	}

	public static boolean inspectWord(String word) {

		System.out.println("testing new");
		if (StringUtils.isAlphanumeric(word)) {
			return true;
		}
		return false;
	}

	public static List<String> inspectWordList(List<String> wordList) {

		Iterator<String> wordListItr = wordList.iterator();
		while (wordListItr.hasNext()) {
			String word = wordListItr.next();
			if (Stopwords.isStopword(word)) {
				wordListItr.remove();
			}

			if (!StringUtils.isAlphanumeric(word)) {
				try {
					if (!Character.isLetterOrDigit(word.charAt(0))) {
						wordListItr.remove();
					}
				} catch (StringIndexOutOfBoundsException e) {
					System.out.println("**Catching the Exception and moving on***" + e);
				}
			}
		}
		return wordList;

	}

	// public static Object[] getBinaryFeatureVectors(Tree<String> insideTree,
	// Tree<String> syntaxTree,
	// Map<Tree<String>, Constituent<String>> constituentsMap,
	// ArrayList<Alphabet> outsideFeatureDictionary,
	// ArrayList<Alphabet> insideFeatureDictionary, Alphabet wordDicitonary) {
	//
	// Object[] binaryFeatureVecs = new Object[4];
	//
	// VSMUtil.setConstituentLength(constituentsMap.get(insideTree));
	// VSMUtil.getNumberOfOutsideWordsLeft(insideTree, constituentsMap,
	// syntaxTree);
	// VSMUtil.getNumberOfOutsideWordsRight(insideTree, constituentsMap,
	// syntaxTree);
	//
	// Stack<Tree<String>> foottoroot = new Stack<Tree<String>>();
	//
	// foottoroot = VSMUtil.updateFoottorootPath(foottoroot, syntaxTree,
	// insideTree, constituentsMap);
	//
	// VSMFeatureVectorBean vectorBean = new VSMFeatureVectorBean();
	//
	// VSMWordFeatureVectorBean vectorBeanWord = new VSMWordFeatureVectorBean();
	//
	// /*
	// * Binary Feature Vectors
	// */
	// SparseVector psiSyn = new
	// VSMOutsideFeatureVector().getOutsideFeatureVectorPsi(foottoroot,
	// outsideFeatureDictionary, vectorBean);
	// SparseVector phiSyn = new
	// VSMInsideFeatureVector().getInsideFeatureVectorPhi(insideTree,
	// insideFeatureDictionary, vectorBean);
	// SparseVector psiSem = new
	// VSMOutsideFeatureVectorWords().getOutsideFeatureVectorPsi(syntaxTree,
	// insideTree,
	// wordDicitonary, vectorBeanWord);
	// SparseVector phiSem = new
	// VSMInsideFeatureVectorWords().getInsideFeatureVectorPhi(insideTree,
	// wordDicitonary,
	// vectorBeanWord);
	//
	// binaryFeatureVecs[0] = psiSyn;
	// binaryFeatureVecs[1] = phiSyn;
	// binaryFeatureVecs[2] = psiSem;
	// binaryFeatureVecs[3] = phiSem;
	//
	// return binaryFeatureVecs;
	//
	// }

	public static Vector getPhiSynEmbedded(SparseVector phiSyn, no.uib.cipr.matrix.DenseMatrix YTSyn) {

		Vector phiDenseSyn = new DenseVector(phiSyn.size());

		java.util.Iterator<VectorEntry> sparseVecItr = phiSyn.iterator();

		while (sparseVecItr.hasNext()) {
			VectorEntry e = sparseVecItr.next();

			int idx = e.index();
			double val = e.get();

			phiDenseSyn.add(idx, val);
		}

		Vector phiSynEmbedded = new DenseVector(YTSyn.numRows());

		YTSyn.mult(phiDenseSyn, phiSynEmbedded);

		phiSynEmbedded = CommonUtil.normalizeVec(phiSynEmbedded);

		return phiSynEmbedded;
	}

	public static Vector getPhiSemEmbedded(SparseVector phiSem, no.uib.cipr.matrix.DenseMatrix YTSem) {

		Vector phiDenseSem = new DenseVector(phiSem.size());

		/*
		 * Iterator over the sparse vector MTJ
		 */

		/*
		 * Iterator over the sparse vector MTJ
		 */
		java.util.Iterator<VectorEntry> sparseVecSemItr = phiSem.iterator();

		/*
		 * Iterating over the sparse vector entries
		 */
		while (sparseVecSemItr.hasNext()) {
			VectorEntry e = sparseVecSemItr.next();

			/*
			 * Getting the sparse vector index and values
			 */
			int idx = e.index();
			double val = e.get();

			/*
			 * Forming the dense inside feature vector
			 */
			phiDenseSem.add(idx, val);
		}

		/*
		 * Multiply the matrix and the vector, to get the lower dimensional
		 * embedding
		 */

		Vector phiSemEmbedded = new DenseVector(YTSem.numRows());
		YTSem.mult(phiDenseSem, phiSemEmbedded);

		phiSemEmbedded = CommonUtil.normalizeVec(phiSemEmbedded);
		return phiSemEmbedded;
	}

	public static Vector getPsiSynEmbedded(SparseVector psiSyn, no.uib.cipr.matrix.DenseMatrix ZTSyn) {
		Vector psiDenseSyn = new DenseVector(psiSyn.size());

		/*
		 * Iterator over the sparse vector MTJ
		 */
		java.util.Iterator<VectorEntry> sparseVecItrOut = psiSyn.iterator();

		/*
		 * Iterating over the sparse vector entries
		 */
		while (sparseVecItrOut.hasNext()) {
			VectorEntry e = sparseVecItrOut.next();

			/*
			 * Getting the sparse vector index and values
			 */
			int idx = e.index();
			double val = e.get();

			/*
			 * Forming the dense inside feature vector
			 */
			psiDenseSyn.add(idx, val);

		}

		Vector psiSynEmbedded = new DenseVector(ZTSyn.numRows());
		ZTSyn.mult(psiDenseSyn, psiSynEmbedded);

		psiSynEmbedded = CommonUtil.normalizeVec(psiSynEmbedded);
		return psiSynEmbedded;
	}

	public static Vector getPsiSemEmbedded(SparseVector psiSem, no.uib.cipr.matrix.DenseMatrix ZTSem) {
		Vector psiDenseSem = new DenseVector(psiSem.size());
		/*
		 * Iterator over the sparse vector MTJ
		 */
		java.util.Iterator<VectorEntry> sparseVecItrSemOut = psiSem.iterator();

		/*
		 * Iterating over the sparse vector entries
		 */
		while (sparseVecItrSemOut.hasNext()) {
			VectorEntry e = sparseVecItrSemOut.next();

			/*
			 * Getting the sparse vector index and values
			 */
			int idx = e.index();
			double val = e.get();

			/*
			 * Forming the dense inside feature vector
			 */
			psiDenseSem.add(idx, val);
		}

		Vector psiSemEmbedded = new DenseVector(ZTSem.numRows());

		ZTSem.mult(psiDenseSem, psiSemEmbedded);

		psiSemEmbedded = CommonUtil.normalizeVec(psiSemEmbedded);
		return psiSemEmbedded;
	}

	// public static void serializeEmbeddedFeatureVecs(Vector phiSynEmbedded,
	// Vector phiSemEmbedded, Vector psiSynEmbedded,
	// Vector psiSemEmbedded, int wordCount, String word,
	// VSMSerializeFeatureVectorBeanTraining serializeBean,
	// String nonTerminal, Alphabet featureCount) {
	//
	// VSMFeatureVectorBeanEmbedded vectorBeanEmbedded = new
	// VSMFeatureVectorBeanEmbedded();
	//
	// vectorBeanEmbedded.setPhiSynEmbedded((DenseVector) phiSynEmbedded);
	// vectorBeanEmbedded.setPhiSemEmbedded((DenseVector) phiSemEmbedded);
	//
	// vectorBeanEmbedded.setPsiSynEmbedded((DenseVector) psiSynEmbedded);
	// vectorBeanEmbedded.setPsiSemEmbedded((DenseVector) psiSemEmbedded);
	//
	// vectorBeanEmbedded.setLabel(word);
	//
	// phiSynEmbedded = null;
	// phiSemEmbedded = null;
	// psiSemEmbedded = null;
	// psiSynEmbedded = null;
	//
	// /*
	// * Serializing the embedded vector bean for each word
	// */
	// if (wordCount == 30000) {
	// wordCount = 0;
	// }
	// serializeBean.serializeEmbeddedVectorBeanWords(vectorBeanEmbedded,
	// nonTerminal, word, wordCount,
	// featureCount.countMap.get(word));
	//
	// System.out.println("Serialized the feature vector***");
	//
	// }

	public static File[] getSICKSentenceDirectories(String directPath) {

		File[] directories = new File(directPath).listFiles(new FileFilter() {

			@Override
			public boolean accept(File file) {
				return !file.isHidden();
			}
		});

		return directories;
	}

	public static File[] getSentenceChunks(File sentenceDirectory, Logger logger) {

		File[] chunks = sentenceDirectory.listFiles(new FileFilter() {

			@Override
			public boolean accept(File file) {
				// TODO Auto-generated method stub
				return !file.isHidden();
			}
		});

		if (chunks != null) {
			return chunks;
		} else {
			logger.log(Level.SEVERE, "The Sentence Chunks cannot be retrieved for this particular sentence: "
					+ sentenceDirectory.getAbsolutePath());
		}
		return chunks;
	}

	public static File[] getWordChunks(File wordDirectory, Logger logger) {

		File[] chunks = wordDirectory.listFiles(new FileFilter() {

			@Override
			public boolean accept(File file) {
				// TODO Auto-generated method stub
				return !file.isHidden();
			}
		});

		if (chunks != null) {
			return chunks;
		} else {
			logger.log(Level.SEVERE, "The Sentence Chunks cannot be retrieved for this particular sentence: "
					+ wordDirectory.getAbsolutePath());
		}
		return chunks;
	}

	public static File[] getBinaryChunkVectors(File chunkDirectory, Logger logger) {
		File[] vectors = chunkDirectory.listFiles(new FileFilter() {
			@Override
			public boolean accept(File file) {
				return !file.isHidden();
			}
		});

		if (vectors == null) {
			logger.log(Level.SEVERE,
					"Exception Occured while reading the chunk directory: " + chunkDirectory.getAbsolutePath());
		}

		return vectors;
	}

	public static Object[] getSyntacticProjectionMatricesTransposed(String nonTerminal, Logger logger) {
		Object[] matricesSyn = null;
		Object[] projectionMatricesT = new Object[2];
		Matrix YSyn = null;
		Matrix ZSyn = null;
		no.uib.cipr.matrix.DenseMatrix YTSyn = null;
		no.uib.cipr.matrix.DenseMatrix ZTSyn;
		try {
			matricesSyn = CommonUtil.deserializeCCAVariantsRun(nonTerminal);
		} catch (ClassNotFoundException e) {
			logger.log(Level.SEVERE, "Exception While Reading the Syntactic Projection Matrices");
			System.exit(-1);
			e.printStackTrace();
		}

		if (matricesSyn[0] != null) {
			YSyn = (Matrix) matricesSyn[0];
		}

		if (matricesSyn[1] != null) {
			ZSyn = (Matrix) matricesSyn[1];
		}

		YTSyn = new no.uib.cipr.matrix.DenseMatrix(YSyn.getColumnDimension(), YSyn.getRowDimension());

		no.uib.cipr.matrix.DenseMatrix YMTJSyn = CommonUtil.createDenseMatrixMTJ(YSyn);

		YSyn = null;

		YMTJSyn.transpose(YTSyn);

		YMTJSyn = null;

		ZSyn = (Matrix) matricesSyn[1];

		ZTSyn = new no.uib.cipr.matrix.DenseMatrix(ZSyn.getColumnDimension(), ZSyn.getRowDimension());

		no.uib.cipr.matrix.DenseMatrix ZMTJSyn = CommonUtil.createDenseMatrixMTJ(ZSyn);

		ZSyn = null;

		ZMTJSyn.transpose(ZTSyn);

		ZMTJSyn = null;

		projectionMatricesT[0] = YTSyn;
		projectionMatricesT[1] = ZTSyn;

		return projectionMatricesT;

	}

	/**
	 * 
	 * @param args
	 * @param logger
	 * @return
	 */
	public static int getNumberOfNonTerminals(String[] args, Logger logger) {
		if (args.length > 0) {
			return Integer.parseInt(args[0]);
		}
		return -1;

	}

	public static File[] getEmbeddedChunkVecs(File chunkDirectory, Logger logger) {
		File[] vectors = chunkDirectory.listFiles(new FileFilter() {
			@Override
			public boolean accept(File file) {
				return !file.isHidden();
			}
		});

		if (vectors == null) {
			logger.log(Level.SEVERE,
					"Exception Occured while reading the chunk directory: " + chunkDirectory.getAbsolutePath());
			System.out.println(
					"***The Chunk Directory Could not be read for the Chunk**" + chunkDirectory.getAbsolutePath());
		}

		return vectors;

	}

	// public static ArrayList<String> getBLLIPCorpus(String bllipCorpus) {
	//
	// File[] files = VSMUtil.getFiles(bllipCorpus);
	//
	// return getFilePaths(files);
	//
	// }

	public static double[][] getDouble(DenseMatrix denseMatrix) {

		DenseDoubleMatrix2D x_omega = new DenseDoubleMatrix2D(denseMatrix.rows, denseMatrix.cols);
		for (int i = 0; i < denseMatrix.rows; i++) {
			for (int j = 0; j < denseMatrix.cols; j++) {

				x_omega.set(i, j, denseMatrix.get(i, j));
			}
		}
		return x_omega.toArray();

	}

	public static Matrix createDenseMatrixJAMA(DenseMatrix xJeigen) {
		Matrix x = new Matrix(xJeigen.rows, xJeigen.cols);
		for (int i = 0; i < xJeigen.rows; i++) {
			for (int j = 0; j < xJeigen.cols; j++) {
				x.set(i, j, xJeigen.get(i, j));
			}
		}

		return x;
	}

	public static DenseMatrix createDenseMatrixJEIGEN(Matrix xJama) {
		DenseMatrix x = new DenseMatrix(xJama.getRowDimension(), xJama.getColumnDimension());
		for (int i = 0; i < xJama.getRowDimension(); i++) {
			for (int j = 0; j < xJama.getColumnDimension(); j++) {
				x.set(i, j, xJama.get(i, j));
			}
		}

		return x;
	}

	public static String getUserPreference(String[] args) {
		String userPref = null;

		if (args.length > 1) {
			userPref = args[1];
			return userPref;
		} else {
			System.out.println("GIVE Sem or Syn AT THE COMMAND LINE SO THAT WE KNOW WHICH ALGORITHM TO RUN");
			System.exit(-1);
		}
		return null;
	}

	public static int getHiddenStates(String[] args) {
		int x = 0;
		if (args.length > 2) {
			x = Integer.parseInt(args[2]);
			return x;
		} else {
			System.out.println("GIVE HIDEEN STATTES AT THE COMMAND LINE SO THAT WE KNOW WHICH ALGORITHM TO RUN");
			System.exit(-1);
		}
		return 0;
	}

	public static Matrix normalize(Matrix x) {
		DenseMatrix temp = new DenseMatrix(x.getRowDimension(), x.getColumnDimension());
		DenseMatrix xJ = CommonUtil.createDenseMatrixJEIGEN(x);

		System.out.println("Normalizing the matrix+++++");
		for (int k = 0; k < xJ.rows; k++) {

			DenseMatrix rowVec = xJ.row(k);

			if (!(rowVec.nonZeroCols().rows == 0)) {
				double norm2 = CommonUtil.norm2(rowVec.getValues());
				rowVec = rowVec.div(norm2);
				// System.out.println(rowVec.cols);
				for (int l = 0; l < rowVec.cols; l++) {
					temp.set(k, l, rowVec.get(0, l));
				}

			}

		}

		return CommonUtil.createDenseMatrixJAMA(temp);

	}

	// public static VSMDictionaryBean readSerializedDictionary(String fileName,
	// Logger logger) {
	//
	// VSMDictionaryBean dictionaryBean = null;
	// FileInputStream fis = null;
	// ObjectInputStream in = null;
	//
	// try {
	// fis = new FileInputStream(fileName);
	// in = new ObjectInputStream(fis);
	// dictionaryBean = (VSMDictionaryBean) in.readObject();
	// in.close();
	// } catch (IOException ex) {
	// if (logger != null) {
	// logger.info("The dictionary does not already exists for the non
	// terminal");
	// }
	// return null;
	//
	// } catch (ClassNotFoundException cnfe) {
	// cnfe.printStackTrace();
	// }
	//
	// return dictionaryBean;
	// }

	// public static void serializeFeatureDictionary(VSMDictionaryBean
	// dictionaryBean, String nonTerminal,
	// String dictionaryType) {
	//
	// File file = new File(VSMContant.SYNTACTIC_FEATURE_DICTIONARY_FOLDER +
	// nonTerminal + "/");
	// if (!file.exists()) {
	// file.mkdirs();
	// }
	// String filename = file.getAbsolutePath() + "/dictionary" + dictionaryType
	// + ".ser";
	//
	// File ditionaryFile = new File(filename);
	// if (ditionaryFile.exists()) {
	// file.delete();
	// }
	//
	// FileOutputStream fos = null;
	// ObjectOutputStream out = null;
	//
	// try {
	// fos = new FileOutputStream(filename);
	//
	// out = new ObjectOutputStream(fos);
	//
	// out.writeObject(dictionaryBean);
	//
	// out.close();
	//
	// fos.close();
	// } catch (IOException ex) {
	// ex.printStackTrace();
	// }
	//
	// }

	/**
	 * Method to retrieve all the tree file paths and storing them in an
	 * ArrayList
	 * 
	 * @param treeCorpusRoot
	 * @param numOfTrees
	 * @return
	 */
	public static LinkedList<String> getTreeFilePaths(String treeCorpusRoot, String numOfTrees) {
		LinkedList<String> treeFilePaths = new LinkedList<String>();
		File[] files = new File(treeCorpusRoot).listFiles(new FileFilter() {
			@Override
			public boolean accept(File pathname) {
				return !pathname.isHidden();
			}
		});

		Arrays.sort(files);

		for (File file : files) {
			File[] treeFiles = file.listFiles();
			Arrays.sort(treeFiles);
			for (File treeFile : treeFiles) {
				if (!numOfTrees.equalsIgnoreCase("all") && treeFilePaths.size() > Integer.parseInt(numOfTrees))
					return treeFilePaths;
				else
					treeFilePaths.add(treeFile.getAbsolutePath());
			}
		}

		return treeFilePaths;
	}

	/**
	 * Method to count the number of lines in a file
	 * 
	 * @param filename
	 * @return
	 * @throws IOException
	 */
	public static int countLines(String filename) throws IOException {
		InputStream is = new BufferedInputStream(new FileInputStream(filename));
		try {
			byte[] c = new byte[1024];
			int count = 0;
			int readChars = 0;
			boolean empty = true;
			while ((readChars = is.read(c)) != -1) {
				empty = false;
				for (int i = 0; i < readChars; ++i) {
					if (c[i] == '\n') {
						++count;
					}
				}
			}
			return (count == 0 && !empty) ? 1 : count;
		} finally {
			is.close();
		}
	}

	public static String getDictionaryType(String[] args) {
		if (args.length > 1) {
			return args[1];
		} else {
			System.err.println("++++NOT ENOUGH INPUT ARGUMENTS+++");
			System.exit(-1);
		}
		return null;
	}

	public static LinkedList<Alphabet> getDictionaries(String featureDictionaries, String nonTerminal, String type)
			throws FileNotFoundException, IOException, ClassNotFoundException {
		LinkedList<Alphabet> dictionaries = new LinkedList<Alphabet>();
		File[] dictionaryFiles = new File(featureDictionaries + "/" + nonTerminal + "/ser/" + type)
				.listFiles(new FileFilter() {

					@Override
					public boolean accept(File pathname) {
						return !pathname.isHidden();
					}
				});

		Arrays.sort(dictionaryFiles);

		for (File f : dictionaryFiles) {
			if (!f.getName().contains(".map.ser")) {
				ObjectInputStream ois = new ObjectInputStream(new FileInputStream(f));
				Alphabet dictionary = (Alphabet) ois.readObject();
				dictionary.stopGrowth();
				ois.close();
				if (dictionary.size() > 1)
					dictionaries.add(dictionary);
			}
		}
		return dictionaries;
	}

	public static int getVectorDimensions(LinkedList<Alphabet> dictionaries) {
		int dimen = 0;
		for (Alphabet dictionary : dictionaries) {
			dimen = dimen + dictionary.size();
		}
		return dimen;
	}

	public static Alphabet combineDictionaries(LinkedList<Alphabet> dictionaries, int dim) {
		Alphabet dictionary = new Alphabet(dim);
		dictionary.allowGrowth();
		dictionary.lookupIndex("NOTFREQUENT");
		for (Alphabet a : dictionaries) {
			Object[] objs = a.map.keys();
			for (Object o : objs) {
				String feature = (String) o;
				if (feature.equals("NOTFREQUENT")) {
					dictionary.countMap.put("NOTFREQUENT",
							dictionary.countMap.get("NOTFREQUENT") + a.countMap.get("NOTFREQUENT"));
				} else {
					int count = a.countMap.get(feature);
					dictionary.lookupIndex(feature);
					dictionary.countMap.put(feature, count);
				}
			}
		}
		return dictionary;
	}

	public static void serializeVec(LinkedList<FeatureVector> featureVecs, String nonTerminal,
			String featureVectorsStoragePath, String type) {
		try {
			File file = new File(
					featureVectorsStoragePath + "/" + nonTerminal.replaceAll("-", "") + "/" + type + ".ser");
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

	public static LinkedList<OutsideFeature> getOutsideFeatureObjects() {
		LinkedList<OutsideFeature> outsideFeatureObjects = new LinkedList<OutsideFeature>(
				Arrays.asList(new OutsideFootNumwordsleft(), new OutsideFootNumwordsright(), new OutsideFootParent(),
						new OutsideFootParentGrandParent(), new OutsideOtherheadposAbove(), new OutsideTreeabove1(),
						new OutsideTreeAbove2(), new OutsideTreeAbove3()));
		return outsideFeatureObjects;

	}

	public static LinkedList<InsideFeature> getInsideFeatureObjects() {
		LinkedList<InsideFeature> insideFeatureObjects = new LinkedList<InsideFeature>(
				Arrays.asList(new InsideBinFull(), new InsideBinLeft(), new InsideBinLeftPlus(), new InsideBinRight(),
						new InsideBinRightPlus(), new InsideNtHeadPos(), new InsideNtNumOfWords(), new InsideUnary()));
		return insideFeatureObjects;
	}

	public static VSMSparseVector getVector(Alphabet alphabet, Alphabet dictionary, int dimension) {
		VSMSparseVector vec = new VSMSparseVector(dimension);
		for (Object obj : alphabet.map.keys()) {
			String feature = (String) obj;
			int dictionaryIndex = dictionary.lookupIndex(feature);
			int featureSampleCount = alphabet.countMap.get(feature);
			int featureCorpusCount = dictionary.countMap.get(feature);
			double tfidf = 0.0;
			// TODO calculate the tfidf
			vec.add(dictionaryIndex, tfidf);

		}
		return vec;
	}

	public static void serializeDictionary(Alphabet dictionary, String nonTerminal, String featureVectorsStoragePath,
			String type) {
		try {
			File file = new File(
					featureVectorsStoragePath + "/" + nonTerminal.replaceAll("-", "") + "/" + type + "dictionary.ser");
			if (!file.getParentFile().exists()) {
				file.getParentFile().mkdirs();
			}
			ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(file));
			os.writeObject(dictionary);
			os.flush();
			os.close();
		} catch (IOException e) {
			// TODO
		}
	}

	public static void writeStatisticsToDisk(LinkedList<FeatureVector> sparseVectors, String nonTerminal,
			String featureVectorsStoragePath, String type) {

		for (FeatureVector vecBean : sparseVectors) {

			try {
				BufferedWriter bw = new BufferedWriter(new FileWriter(
						featureVectorsStoragePath + "/" + nonTerminal.replaceAll("-", "") + "/" + type + "vectors.txt",
						true));
				VSMSparseVector vec = vecBean.getFeatureVec();
				int[] indexes = vec.getIndex();
				String t = "";
				for (int i : indexes) {
					double val = vec.get(i);
					t = t + " " + i + ":" + val;
				}

				t.trim();
				String f = "";
				LinkedList<String> features = vecBean.getFeatureList();
				for (String feature : features) {
					f = f + " " + feature;
				}
				bw.write(t + "\t" + f.trim() + "\t" + vecBean.getInsideTree() + "\t"
						+ Long.toString(vecBean.getTreeIdx()) + ":" + vecBean.getSyntaxTree() + "\n");
				bw.flush();
				bw.close();

			} catch (IOException e) {
				e.printStackTrace();
			}

		}

	}

	public static void writeDictionaryToDisk(Alphabet dictionary, Alphabet sampleDictionary, String nonTerminal,
			String featureVectorsStoragePath, String type) {
		Object[] objs = dictionary.map.keys();
		for (Object obj : objs) {
			try {
				BufferedWriter bw = new BufferedWriter(new FileWriter(featureVectorsStoragePath + "/"
						+ nonTerminal.replaceAll("-", "") + "/" + type + "dictionary.txt", true));
				String feature = (String) obj;
				int index = dictionary.lookupIndex(feature);
				int sampleCount = sampleDictionary.countMap.get(feature);
				bw.write(Integer.toString(index) + "\t" + feature + "\t" + Integer.toString(sampleCount) + "\n");
				bw.flush();
				bw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}

		}

	}

	public static void scaleFeatures(LinkedList<FeatureVector> sparseVectors, Alphabet sampleDictionary,
			Alphabet dictionary, int M, int k) {

		for (FeatureVector bean : sparseVectors) {
			VSMSparseVector scaledPsi = new VSMSparseVector(bean.getFeatureVec().size());
			VSMSparseVector psi = bean.getFeatureVec();
			Iterator<VectorEntry> vecItr = psi.iterator();
			while (vecItr.hasNext()) {
				VectorEntry e = vecItr.next();
				String feature = (String) dictionary.reverseMap.get(e.index());
				int featureCountInM = sampleDictionary.countMap.get(feature);
				double scalingFactor = Math.sqrt((double) (M) / (double) (featureCountInM + k));
				scaledPsi.add(e.index(), (e.get() * scalingFactor));
			}

			bean.setFeatureVec(scaledPsi);
		}

	}

	public static LinkedList<FeatureVector> getVectors(String vectorsPath, org.apache.log4j.Logger logger) {
		logger.info("Deserealizing the vectors from the file: " + vectorsPath);
		LinkedList<FeatureVector> featureVectors = new LinkedList<FeatureVector>();
		try {
			// The check is to ensure that whether .bz2 file is present or the
			// not
			ObjectInputStream ois = null;
			if (!new File(vectorsPath).exists()) {
				ois = new ObjectInputStream(BLLIPCorpusReader.getInputStream(vectorsPath + ".bz2"));
			} else {
				ois = new ObjectInputStream(new FileInputStream(vectorsPath));
			}

			featureVectors = (LinkedList<FeatureVector>) ois.readObject();
			ois.close();
		} catch (IOException e) {
			logger.error("IO Exception while reading the Object file " + e);
		} catch (ClassNotFoundException e) {
			logger.error("CLASSNOTFOUND " + e);
		} catch (CompressorException e) {
			logger.error("CompressorException " + e);
		}
		return featureVectors;
	}

	public static void formFeatureMatrix(LinkedList<FeatureVector> outsideVectors, SparseMatrixLil Psi,
			org.apache.log4j.Logger logger) {
		logger.info("Forming the feature Matrix of dimensions " + Psi.rows + " x " + Psi.cols);
		int col = 0;
		for (FeatureVector bean : outsideVectors) {
			SparseVector vec = bean.getFeatureVec();
			Iterator<VectorEntry> itrs = vec.iterator();
			while (itrs.hasNext()) {
				VectorEntry e = itrs.next();
				Psi.append(e.index(), col, e.get());
			}
			col++;
		}

	}

	public static void serializeFeatureMatrix(SparseMatrixLil featureMatrix, String matrixStoragePath,
			org.apache.log4j.Logger logger) {
		logger.info("Serializing the feature matrix");
		logger.info("++Feature Matrix conversion++");
		SparseDoubleMatrix2D PsiCern = CommonUtil
				.createSparseMatrixCOLT(CommonUtil.createSparseMatrixMTJFromJeigen(featureMatrix));
		try {

			if (new File(matrixStoragePath).exists()) {
				logger.info("The serialized feature matrix already exists and hence doing nothing");
				return;
			}

			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(matrixStoragePath));
			oos.writeObject(PsiCern);
			oos.flush();
			oos.close();
		} catch (IOException e) {
			logger.error("IOException " + e);
		}

	}

	public static void writeFeatureMatrices(SparseMatrixLil matrixS, String matrixStoragePath,
			org.apache.log4j.Logger logger) {

		logger.info("Total memory being used: " + Runtime.getRuntime().totalMemory());
		logger.info("Writing sparse matrices to file");
		try {
			BufferedWriter bw = null;
			BufferedWriter bwl = null;
			int size = matrixS.getSize();
			for (int i = 0; i < size; i++) {
				bw = new BufferedWriter(new FileWriter(matrixStoragePath, true));
				bwl = new BufferedWriter(new FileWriter(matrixStoragePath + ".log", true));
				// +1 for matlab
				int rowidx = matrixS.getRowIdx(i) + 1;
				int colidx = matrixS.getColIdx(i) + 1;
				double value = matrixS.getValue(i);
				bw.write(Integer.toString(rowidx) + "\t" + Integer.toString(colidx) + "\t" + Double.toString(value)
						+ "\n");
				// logarithm of the basis values in the co-variance matrix, just
				// changing the scale. Also provides magnitude normalization (so
				// that no one basis value is very large)
				// TODO put a flag for log tranform here
				bwl.write(Integer.toString(rowidx) + "\t" + Integer.toString(colidx) + "\t"
						+ Double.toString(Math.log(value)) + "\n");
				bw.flush();
				bw.close();
				bwl.flush();
				bwl.close();
			}

			// The last line is added to tell Octave about the matrix
			// dimensions: Very important this
			bw = new BufferedWriter(new FileWriter(matrixStoragePath, true));
			bw.write(Integer.toString(matrixS.rows) + "\t" + Integer.toString(matrixS.cols) + "\t"
					+ Double.toString(0.0));
			bwl = new BufferedWriter(new FileWriter(matrixStoragePath + ".log", true));
			bwl.write(Integer.toString(matrixS.rows) + "\t" + Integer.toString(matrixS.cols) + "\t"
					+ Double.toString(0.0));
			bw.flush();
			bw.close();
			bwl.flush();
			bwl.close();

		} catch (IOException e) {
			logger.error("IOExcpeiton while wirting the feature matrices to the disk" + e);
		}
	}

	public static Object[] deserializeFeatureMatrices(String matrixStorePath, String nonTerminal,
			org.apache.log4j.Logger logger) {

		logger.info("Deserialize the feature matrices ");
		Object[] matrices = new Object[2];
		try {
			ObjectInputStream ois1 = new ObjectInputStream(
					new FileInputStream(matrixStorePath + "/" + nonTerminal + "/ifm.ser"));
			matrices[0] = ois1.readObject();
			ois1.close();
			ObjectInputStream ois2 = new ObjectInputStream(
					new FileInputStream(matrixStorePath + "/" + nonTerminal + "/ofm.ser"));
			matrices[1] = ois2.readObject();
			ois2.close();
		} catch (IOException e) {
			logger.error("IOException " + e);
		} catch (ClassNotFoundException e) {
			logger.error("ClassNotFoundException " + e);
		}

		return matrices;

	}

	public static SparseMatrixLil calculateCovariance(SparseMatrixLil Phi, SparseMatrixLil Psi,
			org.apache.log4j.Logger logger) {
		logger.info("Calculating Covariance matrix");
		SparseMatrixLil covR = Phi.mmul(Psi.t());
		logger.info("CovMatrix Dimensions are given by " + covR.rows + " x " + covR.cols);

		return covR;
	}

	public static void writeMatrixToDisk(SparseMatrixLil covMatrix, String matrixStorePath, String nonTerminal,
			org.apache.log4j.Logger logger) {

		BufferedWriter bw = null;
		int size = covMatrix.getSize();
		for (int i = 0; i < size; i++) {
			try {
				bw = new BufferedWriter(new FileWriter(matrixStorePath + "/" + nonTerminal + "/covM.txt", true));
				// +1 for matlab
				int rowidx = covMatrix.getRowIdx(i) + 1;
				int colidx = covMatrix.getColIdx(i) + 1;
				double val = covMatrix.getValue(i);
				bw.write(Integer.toString(rowidx) + "\t" + Integer.toString(colidx) + "\t" + Double.toString(val)
						+ "\n");
				bw.flush();
				bw.close();

			} catch (IOException e) {
				logger.info("IOException " + e);
			}

		}

		try {
			bw = new BufferedWriter(new FileWriter(matrixStorePath + "/" + nonTerminal + "/covM.txt", true));
			bw.write(Integer.toString(covMatrix.rows) + "\t" + Integer.toString(covMatrix.cols) + "\t"
					+ Double.toString(0.0));
			bw.close();
		} catch (IOException e) {
			logger.error("IOException " + e);
		}

	}

	public static void serializeCovMatrix(SparseMatrixLil covMatrix, String matrixStorePath, String nonTerminal,
			org.apache.log4j.Logger logger) {
		logger.info("Serializing the covariance matrix for the non-terminal " + nonTerminal);
		SparseDoubleMatrix2D coMatrix2d = CommonUtil
				.createSparseMatrixCOLT(CommonUtil.createSparseMatrixMTJFromJeigen(covMatrix));
		try {
			ObjectOutputStream os = new ObjectOutputStream(
					new FileOutputStream(matrixStorePath + "/" + nonTerminal.replaceAll("-", "") + "/covM.ser"));
			os.writeObject(coMatrix2d);
			os.flush();
			os.close();
		} catch (FileNotFoundException e) {
			logger.error("FileNotFoundException " + e);
		} catch (IOException e) {
			logger.error("IOException " + e);
		}

	}

	public static void writeSparseMatrixToFile(SparseMatrixLil psi) {
		// TODO Auto-generated method stub

	}

	public static FlexCompRowMatrix createSparseMatrixMTJFromJeigen(SparseMatrixLil xjeig) {
		FlexCompRowMatrix x = new FlexCompRowMatrix(xjeig.rows, xjeig.cols);

		int count = xjeig.getSize();
		for (int i = 0; i < count; i++) {
			int row = xjeig.getRowIdx(i);
			int col = xjeig.getColIdx(i);
			double value = xjeig.getValue(i);
			// if(value!=0)
			x.set(row, col, value);
		}

		return x;
	}

	public static SparseDoubleMatrix2D createSparseMatrixCOLT(FlexCompRowMatrix xmtj) {

		System.out.println(" Number Rows: " + xmtj.numRows());
		System.out.println(" Number Cols: " + xmtj.numColumns());

		xmtj.compact();

		SparseDoubleMatrix2D x_omega = new SparseDoubleMatrix2D(xmtj.numRows(), xmtj.numColumns(), 0, 0.70, 0.75);

		for (MatrixEntry e : xmtj) {
			x_omega.set(e.row(), e.column(), e.get());
		}

		System.out.println("==Created Sparse Matrix==");
		return x_omega;
	}

	public DenseDoubleMatrix2D getOmegaMatrix(SparseDoubleMatrix2D covM, int m) {
		Random r = new Random();
		DenseDoubleMatrix2D Omega = new DenseDoubleMatrix2D(covM.columns(), m + 20);
		for (int i = 0; i < covM.columns(); i++) {
			for (int j = 0; j < m + 20; j++)
				Omega.set(i, j, r.nextGaussian());
		}
		return Omega;
	}

	public static void dictionaryInit(Alphabet[] alphabets) {
		for (Alphabet a : alphabets) {
			if (a != null) {
				a.allowGrowth();
				a.turnOnCounts();
			} else {
				System.err.println("Alphabet is null");
				System.exit(-1);
			}
		}

	}

	public static void createDictionaryFromFile(String wordListFile, Alphabet wordL) {
		try {
			System.out.println(wordListFile);
			BufferedReader reader = new BufferedReader(new FileReader(wordListFile));
			String word = null;
			while ((word = reader.readLine()) != null) {
				wordL.lookupIndex(word.toLowerCase());
			}
			reader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public static void createWordDictionary(Alphabet word, String treesFile, org.apache.log4j.Logger logger) {
		PennTreeReader treeParser = null;
		try {
			treeParser = CommonUtil.getTreeReaderBz(treesFile);

		} catch (Exception e) {
			e.printStackTrace();
		}

		long k = 0;
		while (treeParser.hasNext()) {
			k++;
			Tree<String> tree = FeatureDictionary.getNormalizedTree(treeParser.next());
			List<String> words = tree.getTerminalYield();
			for (String wordS : words) {
				if (wordS.matches("[0-9]")) {
					wordS = "<num>";
				}
				word.lookupIndex(wordS.toLowerCase().trim());
			}

			if (k == 1000000) {
				logger.info("A million trees parsed");
				k = 0;
			}
		}

	}

	public static void filterWordDictionary(Alphabet word, Alphabet wordF, String thresh) {
		Object[] words = word.map.keys();
		wordF.lookupIndex("NOTFREQUENT");
		for (Object wordO : words) {
			String wordS = (String) wordO;
			int wordCount = word.countMap.get(wordS);
			if (wordCount > Integer.parseInt(thresh)) {
				wordF.lookupIndex(wordS);
				wordF.countMap.put(wordS, wordCount);
			} else {
				wordF.countMap.put("NOTFREQUENT", wordF.getCount("NOTFREQUENT") + wordCount);
			}
		}
	}

	public static void stopDictionaryGrowth(Alphabet[] alphabets) {
		for (Alphabet a : alphabets) {
			a.stopGrowth();
		}

	}

	public static void createGarbageWordDictionary(Alphabet wordF, Alphabet wordG, Alphabet wordL) {
		Object[] wordsF = wordF.map.keys();
		for (Object o : wordsF) {
			String word = (String) o;
			if (wordL.lookupIndex(word) == -1) {
				wordG.lookupIndex(word);
				wordG.countMap.put(word, wordF.getCount(word));
			}
		}

	}

	public static void serializeDictionaries(Alphabet[] alphabets, String[] names, String storePath) {
		for (int i = 0; i < alphabets.length; i++) {
			Alphabet a = alphabets[i];
			String name = names[i];
			try {
				ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(storePath + "/" + name + ".ser"));
				os.writeObject(a);
				os.flush();
				os.close();
			} catch (IOException e) {
				e.printStackTrace();
			}

		}

	}

	public static void writeDictionariesToDisk(Alphabet[] alphabets, String[] names, String storePath) {

		for (int i = 0; i < alphabets.length; i++) {
			Alphabet a = alphabets[i];
			String name = names[i];
			try {
				BufferedWriter bw = null;
				for (Object o : a.map.keys()) {
					bw = new BufferedWriter(new FileWriter(storePath + "/" + name + ".txt", true));
					bw.write((String) o + "\t" + a.countMap.get(o) + "\n");
					bw.flush();
					bw.close();
				}

			} catch (IOException e) {
				e.printStackTrace();
			}

		}

	}

	public static void createAlphanumericDictionary(Alphabet word, Alphabet alNumeric) {
		Object[] objs = word.map.keys();
		for (Object o : objs) {
			String wordS = (String) o;
			if (!StringUtils.isAlphanumeric(wordS)) {
				alNumeric.lookupIndex(wordS);
				alNumeric.countMap.put(wordS, word.countMap.get(wordS));
			}
		}

	}

	public static void writeMatrixToDisk(String objectFile, String textFileStoragePath, String name, String nonTerminal,
			org.apache.log4j.Logger logger) {

		try {
			ObjectInputStream ois = new ObjectInputStream(BLLIPCorpusReader.getInputStream(objectFile));
			SparseDoubleMatrix2D covCern = (SparseDoubleMatrix2D) ois.readObject();
			ois.close();
			logger.info("Converting from colt to jeigen for " + nonTerminal);
			SparseMatrixLil cov = new SparseMatrixLil(covCern.rows(), covCern.columns());
			for (int i = 0; i < covCern.rows(); i++) {
				for (int k = 0; k < covCern.columns(); k++) {
					double val = covCern.get(i, k);
					// logrithm of the values of the covariance matrix? the
					// values are too small for some matrices
					if (val > 0.0) {
						cov.append(i, k, val);
					}
				}
			}
			String matrixStoragePath = textFileStoragePath + "/" + nonTerminal + "/" + name + ".txt";
			File file = new File(matrixStoragePath);
			if (!file.getParentFile().exists()) {
				file.getParentFile().mkdirs();
			} else {
				logger.info("Covariance matrices for the non-terminal: " + nonTerminal
						+ " already exists and hence exiting");
				System.exit(0);
			}
			CommonUtil.writeFeatureMatrices(cov, matrixStoragePath, logger);

		} catch (IOException e) {
			logger.error("IOException: " + e + " non-terminal " + nonTerminal);
		} catch (ClassNotFoundException e) {
			logger.error("ClassNotFoundException " + e + " non-term " + nonTerminal);
		} catch (CompressorException e) {
			logger.error("CompressorException " + e + " non-term " + nonTerminal);
		}

	}

	public static void writeDoubleArrayToFile(double[][] matrix, int rowDimension, int columnDimension,
			String storePath) {

		BufferedWriter bw = null;
		try {
			for (int i = 0; i < rowDimension; i++) {
				bw = new BufferedWriter(new FileWriter(storePath));
				for (int k = 0; k < columnDimension; k++) {
					double value = matrix[i][k];
					bw.write(Double.toString(value) + "\t");
				}
				bw.write("\n");
				bw.flush();
				bw.close();
			}
		} catch (IOException e) {

		}

	}

}
