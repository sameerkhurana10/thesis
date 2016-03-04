package main;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.log4j.Logger;

import runnables.InsideFeatureMatrix;
import runnables.OutsideFeatureMatrix;

public class CreateFeatureMatricesText {

	private static String nonTerminal;
	private static String matrixStoragePath;
	private static String vectorsPath;
	private static String serialize;
	private static String writeToDisk;
	private static String inside;
	private static String outside;
	final static Logger logger = Logger.getLogger(CreateFeatureMatricesText.class);
	private static Options options;

	static {
		options = new Options();
		options.addOption("nonTerminal", true, "non terminal for which text file needs to be generated");
		options.addOption("matrixStoragePath", true, "where you want to store your feature matrix");
		options.addOption("vectorsPath", true, "root directory where all the vectors are stored");
		options.addOption("serialize", true, "Yes/No do you want to serialize the feature matrix");
		options.addOption("writeToDisk", true, "Yes/No do you want to write the matrix to the disk");
		options.addOption("inside", true, "Yes/No: Whether to form the inside feature matrix or not");
		options.addOption("outside", true, "Yes/No whether to form the outside feature matrix or not");
	}

	public static void main(String[] args) {

		parse(args);

		if (outside.equalsIgnoreCase("yes")) {
			Thread outsideMatrix = new Thread(new OutsideFeatureMatrix(nonTerminal, matrixStoragePath, vectorsPath,
					serialize, writeToDisk, logger));
			outsideMatrix.start();
		}

		if (inside.equalsIgnoreCase("yes")) {
			Thread insideMatrix = new Thread(new InsideFeatureMatrix(nonTerminal, matrixStoragePath, vectorsPath,
					serialize, writeToDisk, logger));
			insideMatrix.start();
		}
	}

	private static void parse(String[] args) {
		CommandLineParser parser = new BasicParser();

		CommandLine cmd = null;

		try {
			cmd = parser.parse(options, args);
			if (cmd.hasOption("nonTerminal") && cmd.hasOption("matrixStoragePath") && cmd.hasOption("vectorsPath")
					&& cmd.hasOption("serialize") && cmd.hasOption("writeToDisk") && cmd.hasOption("inside")
					&& cmd.hasOption("outside")) {
				nonTerminal = cmd.getOptionValue("nonTerminal");
				matrixStoragePath = cmd.getOptionValue("matrixStoragePath");
				vectorsPath = cmd.getOptionValue("vectorsPath");
				serialize = cmd.getOptionValue("serialize");
				writeToDisk = cmd.getOptionValue("writeToDisk");
				inside = cmd.getOptionValue("inside");
				outside = cmd.getOptionValue("outside");
			} else {
				help();
			}
		} catch (ParseException e) {
			logger.error(e);
		}

	}

	private static void help() {
		HelpFormatter formater = new HelpFormatter();
		formater.printHelp("CreateFeatureMatricesText", options);
		System.exit(0);
	}

}
