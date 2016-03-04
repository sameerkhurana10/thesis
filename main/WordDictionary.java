package main;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.log4j.Logger;

import dictionary.Alphabet;
import utils.CommonUtil;

public class WordDictionary {

	private static Alphabet word;
	private static Alphabet wordF;
	private static Alphabet wordG;
	private static Alphabet alNumeric;
	private static String treesFile;
	private static String wordListFile;
	private static Alphabet wordL;
	private static Options options;
	private static String thresh;
	private static String storePath;

	final static Logger logger = Logger.getLogger(WordDictionary.class);

	static {
		word = new Alphabet();
		wordF = new Alphabet();
		wordG = new Alphabet();
		wordL = new Alphabet();
		alNumeric = new Alphabet();
		options = new Options();
		options.addOption("w", true, "path of the word list file");
		options.addOption("p", true, "storage path for the word dictionaries");
		options.addOption("t", true,
				"word cut off frequency. Words that occur in the corpus less than the threshold will be ignored");
		options.addOption("trees", true, "path to the file that contains all the trees");
	}

	public static void main(String[] args) {
		parse(args);
		CommonUtil.dictionaryInit(new Alphabet[] { word, wordF, wordG, wordL, alNumeric });
		CommonUtil.createDictionaryFromFile(wordListFile, wordL);
		logger.info("Creating word dictionary");
		CommonUtil.createWordDictionary(word, treesFile, logger);
		logger.info("Creating the alphanumeric dictionary");
		CommonUtil.createAlphanumericDictionary(word, alNumeric);
		logger.info("Filtering the dictionary");
		CommonUtil.filterWordDictionary(word, wordF, thresh);
		CommonUtil.stopDictionaryGrowth(new Alphabet[] { word, wordF, wordL, alNumeric });
		logger.info("Creating garbage dictionary for analysis");
		CommonUtil.createGarbageWordDictionary(wordF, wordG, wordL);
		logger.info("Serializing the word dictionaries");
		CommonUtil.serializeDictionaries(new Alphabet[] { wordG, wordF, word, alNumeric },
				new String[] { "gwords", "filwords", "allwords", "alnum" }, storePath);
		logger.info("Writing the dictionaries to the disk for analysis in Excel");
		CommonUtil.writeDictionariesToDisk(new Alphabet[] { wordG, wordF, word, alNumeric },
				new String[] { "gwords", "filwords", "allwords", "alnum" }, storePath);

	}

	private static void parse(String[] args) {
		CommandLineParser parser = new BasicParser();

		CommandLine cmd = null;

		try {
			cmd = parser.parse(options, args);
			if (cmd.hasOption("w") && cmd.hasOption("p") && cmd.hasOption("t") && cmd.hasOption("trees")) {
				wordListFile = cmd.getOptionValue("w");
				treesFile = cmd.getOptionValue("trees");
				thresh = cmd.getOptionValue("t");
				storePath = cmd.getOptionValue("p");
			} else {
				help();
			}

		} catch (ParseException e) {
			logger.error(e);
		}

	}

	private static void help() {
		HelpFormatter formater = new HelpFormatter();
		formater.printHelp("WordDictionary", options);
		System.exit(0);
	}

}
