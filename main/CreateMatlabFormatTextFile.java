package main;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.log4j.Logger;

import utils.CommonUtil;

public class CreateMatlabFormatTextFile {

	private static String objectFile;

	private static String textFileStoragePath;
	private static String nonTerminal;
	private static String name;

	private static Options options;

	final static Logger logger = Logger.getLogger(CreateMatlabFormatTextFile.class);

	static {
		options = new Options();
		options.addOption("o", true, "Object file with extension .ser");
		options.addOption("s", true, "text file storage path that will be loaded into matlab");
		options.addOption("nt", true, "give the name of the non-terminal");
		options.addOption("name", true, "the name of the text file with which you want to store the file");
	}

	public static void main(String[] args) {
		parse(args);
		CommonUtil.writeMatrixToDisk(objectFile, textFileStoragePath, name, nonTerminal, logger);
	}

	private static void parse(String[] args) {
		CommandLineParser parser = new BasicParser();

		CommandLine cmd = null;

		try {
			cmd = parser.parse(options, args);
			if (cmd.hasOption("o") && cmd.hasOption("s") && cmd.hasOption("nt") && cmd.hasOption("name")) {
				objectFile = cmd.getOptionValue("o");
				textFileStoragePath = cmd.getOptionValue("s");
				nonTerminal = cmd.getOptionValue("nt");
				name = cmd.getOptionValue("name");
			} else {
				help();

			}
		} catch (Exception e) {
			logger.error("Exception while parsing command line: " + e);
		}
	}

	private static void help() {
		HelpFormatter formater = new HelpFormatter();
		formater.printHelp("CreateMatlabFormatTextFile", options);
		System.exit(0);
	}

}
