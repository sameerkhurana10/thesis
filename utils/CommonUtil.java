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

import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
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
import jeigen.DenseMatrix;
import jeigen.SparseMatrixLil;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.VectorEntry;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import no.uib.cipr.matrix.sparse.SparseVector;
import weka.core.Stopwords;

/**
 * This is a Utility class for the Project Vector Space Modelling
 * 
 * @author sameerkhurana10
 *
 */

public class CommonUtil {

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

		List<String> features = Arrays.asList(new OutsideFootNumwordsleft().getFeature(footToRoot),
				new OutsideFootNumwordsright().getFeature(footToRoot), new OutsideFootParent().getFeature(footToRoot),
				new OutsideFootParentGrandParent().getFeature(footToRoot),
				new OutsideOtherheadposAbove().getFeature(footToRoot), new OutsideTreeabove1().getFeature(footToRoot),
				new OutsideTreeAbove2().getFeature(footToRoot), new OutsideTreeAbove3().getFeature(footToRoot));

		return features;
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

	public static void writeCovarMatrix(SparseMatrixLil psiTPsi, String nonTerminal) {
		id++;
		String filePath = "/afs/inf.ed.ac.uk/group/project/vsm.restored/covars/" + nonTerminal + "/" + "covar_" + id
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
	public static ArrayList<String> getTreeFilePaths(String treeCorpusRoot, String numOfTrees) {
		ArrayList<String> treeFilePaths = new ArrayList<String>();
		File[] files = new File(treeCorpusRoot).listFiles(new FileFilter() {
			@Override
			public boolean accept(File pathname) {
				return !pathname.isHidden();
			}
		});

		for (File file : files) {
			File[] treeFiles = file.listFiles();
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

}
