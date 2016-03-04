package main;

import java.io.File;
import java.io.IOException;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.log4j.Logger;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;
import com.jujutsu.tsne.FastTSne;
import com.jujutsu.tsne.TSne;

import Jama.Matrix;
import utils.CommonUtil;

public class TsnePreparation {

	private static String nonTerminal;
	private static String featureVectorsFile;
	private static String matFile;
	private static String storeTsneFilePath;
	private static MatFileReader matFileReader;
	private static String matrixName;

	final static Logger logger = Logger.getLogger(TsnePreparation.class);

	private static Options options;

	static {
		options = new Options();
		options.addOption("nonTerminal", true, "Non terminal name");
		options.addOption("featureVectorsFileDir", true, "feature vectors file");
		options.addOption("matFile", true, "mat file");
		options.addOption("storeTsneFilePath", true, "directory where tsne file is to be stored");
		options.addOption("matName", true, "name of the matrix inside the mat file");
	}

	public static void main(String[] args) {
		parse(args);
		try {
			System.out.println(matFile);
			matFileReader = new MatFileReader(new File(matFile));
			Matrix matrix = new Matrix(((MLDouble) matFileReader.getMLArray(matrixName)).getArray());
			matrix = matrix.transpose();
			double[][] matrixD = matrix.getArray();
			TSne tsne = new FastTSne();
			double[][] tsneMatrix = tsne.tsne(matrixD, 2, matrix.getColumnDimension(), 20.0);

			// Tsne projection of matrixD
			CommonUtil.writeDoubleArrayToFile(tsneMatrix, matrix.getRowDimension(), matrix.getColumnDimension(),
					storeTsneFilePath + "/" + nonTerminal + "/tsne.txt");

		} catch (IOException e) {
			logger.error("Error while Reading the mat file" + e);
		}
	}

	private static void parse(String[] args) {
		CommandLineParser parser = new BasicParser();

		CommandLine cmd = null;

		try {
			cmd = parser.parse(options, args);
			if (cmd.hasOption("nonTerminal") && cmd.hasOption("matFile") && cmd.hasOption("storeTsneFilePath")
					&& cmd.hasOption("matName")) {
				nonTerminal = cmd.getOptionValue("nonTerminal");
				matFile = cmd.getOptionValue("matFile");
				storeTsneFilePath = cmd.getOptionValue("storeTsneFilePath");
				matrixName = cmd.getOptionValue("matName");
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
