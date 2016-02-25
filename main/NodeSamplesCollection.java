package main;

import java.util.LinkedList;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.log4j.Logger;

import Jama.Matrix;
import beans.FeatureVector;
import interfaces.InsideFeature;
import interfaces.OutsideFeature;
import jeigen.DenseMatrix;
import jeigen.SparseMatrixLil;
import no.uib.cipr.matrix.DenseVector;
import runnables.InsideFeatureMatrix;
import runnables.InsideFeatureVectors;
import runnables.OutsideFeatureMatrix;
import runnables.OutsideFeatureVectors;
import utils.CommonUtil;
import utils.VSMDenseVector;
import utils.VSMSparseMatrixLil;
import utils.VSMSparseVector;
import weka.classifiers.rules.DecisionTable.Link;

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
		//
		// Thread outsideFeatureVec = new Thread(new
		// OutsideFeatureVectors(treeFile, featureVectorsStoragePath, logger,
		// nonTerminal, M, featureDictionaries, outsideFeatures, k));
		//
		// Thread insideFeatureVec = new Thread(new
		// InsideFeatureVectors(treeFile, featureVectorsStoragePath, logger,
		// nonTerminal, M, featureDictionaries, insideFeatures, k));
		//
		// outsideFeatureVec.start();
		// insideFeatureVec.start();
		// try {
		// outsideFeatureVec.join();
		// insideFeatureVec.join();
		// } catch (InterruptedException e) {
		// e.printStackTrace();
		// }

		logger.info("Get the inside and the outside sparse feature vectors for the non-terminal " + nonTerminal);

		LinkedList<FeatureVector> outsideFeatureVectors = CommonUtil
				.getVectors(featureVectorsStoragePath + "/" + nonTerminal + "/outside.ser.bz2", logger);
		LinkedList<FeatureVector> insideFeatureVectors = CommonUtil
				.getVectors(featureVectorsStoragePath + "/" + nonTerminal + "/inside.ser.bz2", logger);

		DenseMatrix covMatFinal = new DenseMatrix(insideFeatureVectors.get(0).getFeatureVec().size(),
				outsideFeatureVectors.get(0).getFeatureVec().size());

		logger.info("Covariance matrix dimensions " + covMatFinal.shape());

		// Getting the covariance Matrix for each sample
		for (int i = 0; i < M; i++) {

			System.out.println("++Covariance for sample: " + i);
			// Don't forget to center the vectors
			DenseVector ov = CommonUtil.normalizeVector(outsideFeatureVectors.get(i).getFeatureVec(), "no", logger);
			double[] ovvalues = ov.getData();
			double[][] ovMatrixD = new double[ovvalues.length][1];
			DenseMatrix ovMat = new DenseMatrix(ovMatrixD);

			DenseVector iv = CommonUtil.normalizeVector(insideFeatureVectors.get(i).getFeatureVec(), "no", logger);
			double[] ivvalues = iv.getData();
			double[][] ivMatrixD = new double[ivvalues.length][1];
			DenseMatrix ivMat = new DenseMatrix(ivMatrixD);

			DenseMatrix covMat = CommonUtil.calculateCovariance(ivMat, ovMat, logger);

			covMatFinal = covMat.add(covMat);

		}

		// matrix conversion so that it can be serialized
		Matrix covJama = CommonUtil.createDenseMatrixJAMA(covMatFinal);
		CommonUtil.serializeCovMatrix(covJama, matrixStorePath, nonTerminal, logger);

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
