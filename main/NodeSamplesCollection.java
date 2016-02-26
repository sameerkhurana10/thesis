package main;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.LinkedList;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.log4j.Logger;

import interfaces.InsideFeature;
import interfaces.OutsideFeature;
import jeigen.SparseMatrixLil;
import runnables.InsideFeatureMatrix;
import runnables.InsideFeatureVectors;
import runnables.OutsideFeatureMatrix;
import runnables.OutsideFeatureVectors;
import utils.CommonUtil;

/**
 * 
 * @author Sameer Khurana (University of Edinburgh, sameerkhurana10@gmail.com)
 *
 */

public class NodeSamplesCollection {

	private static Options options;
	private static String treeFile;
	private static String featureDictionaries;
	private static String featureVectorsStoragePath;
	private static String nonTerminal;
	private static int M;
	private static LinkedList<InsideFeature> insideFeatures;
	private static LinkedList<OutsideFeature> outsideFeatures;
	private static int k;
	private static String matrixStorePath;

	final static Logger logger = Logger.getLogger(NodeSamplesCollection.class);

	static {
		options = new Options();
		options.addOption("nonT", true, "non terminal for which vectors need to be extracted");
		options.addOption("trees", true, "location of the tree file that contains all the trees");
		options.addOption("dictionaries", true, "location of the folder where all the dictionaries are kept");
		options.addOption("M", true, "Total number of vectors to be extracted for the nonTerminal");
		options.addOption("vecpath", true, "the path where the feature vectors are to be stored on the disk");
		options.addOption("k", true, "the smotthness coefficient to be used in the scaling factor formula");
		options.addOption("matrix", true, "path where matrix needs to be stored");

		insideFeatures = CommonUtil.getInsideFeatureObjects();
		outsideFeatures = CommonUtil.getOutsideFeatureObjects();

	}

	public static void main(String[] args) {

		parse(args);

		Thread outsideFeatureVec = new Thread(new OutsideFeatureVectors(treeFile, featureVectorsStoragePath, logger,
				nonTerminal, M, featureDictionaries, outsideFeatures, k));

		Thread insideFeatureVec = new Thread(new InsideFeatureVectors(treeFile, featureVectorsStoragePath, logger,
				nonTerminal, M, featureDictionaries, insideFeatures, k));

		outsideFeatureVec.start();
		insideFeatureVec.start();
		try {
			outsideFeatureVec.join();
			insideFeatureVec.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		OutsideFeatureMatrix outsideMatrixThread = new OutsideFeatureMatrix(nonTerminal, matrixStorePath,
				featureVectorsStoragePath, M, logger);
		InsideFeatureMatrix insideMatrixThread = new InsideFeatureMatrix(nonTerminal, matrixStorePath,
				featureVectorsStoragePath, M, logger);

		Thread outsideFeatureMatrix = new Thread(outsideMatrixThread);
		Thread insideFeatureMatrix = new Thread(insideMatrixThread);

		outsideFeatureMatrix.start();
		insideFeatureMatrix.start();

		try {
			outsideFeatureMatrix.join();
			insideFeatureMatrix.join();
		} catch (InterruptedException e) {
			logger.info("Exception while joining the matrix threads " + e);
		}

		SparseMatrixLil Psi = outsideMatrixThread.getPsi();
		SparseMatrixLil Phi = insideMatrixThread.getPhi();

		SparseMatrixLil covR = CommonUtil.calculateCovariance(Phi, Psi, logger).div(M);
		CommonUtil.serializeCovMatrix(covR, matrixStorePath, nonTerminal, logger);

	}

	private static void parse(String[] args) {
		CommandLineParser parser = new BasicParser();

		CommandLine cmd = null;

		try {
			cmd = parser.parse(options, args);
			if (cmd.hasOption("nonT") && cmd.hasOption("trees") && cmd.hasOption("dictionaries") && cmd.hasOption("M")
					&& cmd.hasOption("vecpath") && cmd.hasOption("k") && cmd.hasOption("matrix")) {
				nonTerminal = cmd.getOptionValue("nonT");
				treeFile = cmd.getOptionValue("trees");
				featureDictionaries = cmd.getOptionValue("dictionaries");
				M = Integer.parseInt(cmd.getOptionValue("M"));
				featureVectorsStoragePath = cmd.getOptionValue("vecpath");
				k = Integer.parseInt(cmd.getOptionValue("k"));
				matrixStorePath = cmd.getOptionValue("matrix");
			} else {
				help();
			}

		} catch (ParseException e) {
			logger.error(e);
		}

	}

	private static void help() {
		HelpFormatter formater = new HelpFormatter();
		formater.printHelp("FeatureVectors", options);
		System.exit(0);
	}

}
