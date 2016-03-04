package runnables;

import java.io.File;
import java.util.LinkedList;

import org.apache.log4j.Logger;

import beans.FeatureVector;
import jeigen.SparseMatrixLil;
import utils.CommonUtil;

public class InsideFeatureMatrix implements Runnable {

	String vectorsPath;
	String matrixStoragePath;
	LinkedList<FeatureVector> insideVectors;
	SparseMatrixLil Phi;
	Logger logger;
	int d;

	String nonTerminal;
	String serialize;
	String writeToDisk;

	public InsideFeatureMatrix(String nonTerminal, String matrixStoragePath, String vectorsPath, String serialize,
			String writeToDisk, Logger logger) {
		this.nonTerminal = nonTerminal;
		this.matrixStoragePath = matrixStoragePath;
		this.logger = logger;

		this.matrixStoragePath = matrixStoragePath + "/" + nonTerminal.replaceAll("-", "");
		File matrixDirec = new File(this.matrixStoragePath);
		if (!matrixDirec.exists())
			matrixDirec.mkdirs();
		this.vectorsPath = vectorsPath + "/" + nonTerminal.replaceAll("-", "") + "/inside.ser";
		this.serialize = serialize;
		this.writeToDisk = writeToDisk;
	}

	@Override
	public void run() {
		logger.info("Forming the Inside feature Matrices");

		logger.info("Desearilizing the feature vectors");
		insideVectors = CommonUtil.getVectors(vectorsPath, logger);

		d = insideVectors.get(0).getFeatureVec().size();
		int M = insideVectors.size();
		Phi = new SparseMatrixLil(d, M);

		logger.info("Forming the Matrix (d x M): " + Phi.rows + " x " + Phi.cols);
		CommonUtil.formFeatureMatrix(insideVectors, Phi, logger);

		if (serialize.equalsIgnoreCase("yes")) {
			logger.info("Serializing the Inside Feature Matrix at: " + matrixStoragePath);
			CommonUtil.serializeFeatureMatrix(Phi, matrixStoragePath + "/ifm.ser", logger);
		}

		if (writeToDisk.equalsIgnoreCase("yes")) {
			logger.info("Writing the feature matrix to a text file");
			CommonUtil.writeSparseMatrixToDisk(Phi, matrixStoragePath + "/ifm.txt", nonTerminal, logger);
		}
	}

	public SparseMatrixLil getPhi() {
		return Phi;
	}

}
