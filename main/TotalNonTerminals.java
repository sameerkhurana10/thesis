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

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.log4j.Logger;

import dictionary.Alphabet;
import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.syntax.Trees.PennTreeReader;
import utils.CommonUtil;

public class TotalNonTerminals {

	private static Alphabet nonTerminals;

	private static String corpusRootDir;

	private static String writeAtPath;

	final static Logger logger = Logger.getLogger(TotalNonTerminals.class);

	private static Options options;

	static {
		options = new Options();
		options.addOption("r", true, "give the trees directory root path");
		options.addOption("p", true,
				"path where to store the dictionary containing the non-temrinals and their counts in the corpus");

		nonTerminals = new Alphabet();
		nonTerminals.allowGrowth();
		nonTerminals.turnOnCounts();
	}

	public static void main(String[] args) {

		parse(args);
		logger.info("++EXTRACTING NON TERMINALS++");
		List<String> trees = CommonUtil.getTreeFilePaths(corpusRootDir, "all");
		for (String tree : trees) {
			PennTreeReader treeParser = null;
			try {
				treeParser = CommonUtil.getTreeReader(tree);

			} catch (Exception e) {
				e.printStackTrace();
			}

			while (treeParser.hasNext()) {
				Tree<String> sTree = FeatureDictionary.getNormalizedTree(treeParser.next());
				Iterator<Tree<String>> treeNodeItr = null;

				if (tree != null) {
					treeNodeItr = getTreeNodeIterator(sTree);
				}

				while (treeNodeItr != null && treeNodeItr.hasNext()) {
					Tree<String> insideTree = treeNodeItr.next();
					if (!insideTree.isLeaf())
						nonTerminals.lookupIndex(insideTree.getLabel());
				}
			}
		}

		nonTerminals.stopGrowth();

		logger.info("writing some stats about the corpus");
		String nonT = "";
		for (Object obj : nonTerminals.map.keys()) {
			nonT = nonT + "," + (String) obj;
		}
		logger.info("total non terminals in the corpus are: " + nonTerminals.size() + " ***given by**** " + nonT);

		writeDictionaryToDisk();

	}

	private static void writeDictionaryToDisk() {
		logger.info("writing dictionary to the disk");
		BufferedWriter bw = null;
		ObjectOutputStream os = null;
		FileOutputStream fos = null;

		try {
			fos = new FileOutputStream(new File(writeAtPath + "/nonterminals.ser"));
			os = new ObjectOutputStream(fos);
			os.writeObject(nonTerminals);
			os.flush();
			os.close();
		} catch (FileNotFoundException e1) {
			logger.error("File not found exception while wrtiting the non terminals dictionary object: " + e1);
		} catch (IOException e) {
			logger.error("Exception while writing the non terminals object to the disk: " + e);
		}

		try {
			for (Object obj : nonTerminals.map.keys()) {
				bw = new BufferedWriter(new FileWriter(writeAtPath + "/nonterminals.txt", true));
				bw.write((String) obj + "\t" + nonTerminals.countMap.get((String) obj) + "\n");
				bw.flush();
				bw.close();
			}
		} catch (IOException e) {
			logger.error("Exception while writing the dictionary to a file: " + e);
		}

		logger.info("DONE");

	}

	static Iterator<Tree<String>> getTreeNodeIterator(Tree<String> tree) {
		return tree.iterator();
	}

	private static void parse(String[] args) {

		CommandLineParser parser = new BasicParser();
		CommandLine cmd = null;

		try {
			cmd = parser.parse(options, args);
			if (cmd.hasOption("r") && cmd.hasOption("p")) {
				corpusRootDir = cmd.getOptionValue("r");
				writeAtPath = cmd.getOptionValue("p");

			} else {
				help();
			}

		} catch (ParseException e) {

		}

	}

	private static void help() {
		HelpFormatter formater = new HelpFormatter();
		formater.printHelp("TotalNonTerminals", options);
		System.exit(0);
	}

}
