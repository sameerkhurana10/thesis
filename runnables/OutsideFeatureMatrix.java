package runnables;

import java.io.File;
import java.util.LinkedList;

import org.antlr.runtime.SerializedGrammar;
import org.apache.log4j.Logger;

import beans.FeatureVector;
import jeigen.SparseMatrixLil;
import utils.CommonUtil;

public class OutsideFeatureMatrix implements Runnable {

	String vectorsPath;
	String matrixStoragePath;
	LinkedList<FeatureVector> outsideVectors;
	SparseMatrixLil Psi;
	Logger logger;
	int dprime;

	String nonTerminal;
	String serialize;
	String writeToDisk;

	public OutsideFeatureMatrix(String nonTerminal, String matrixStoragePath, String vectorsPath, String serialize,
			String writeToDisk, Logger logger) {
		this.nonTerminal = nonTerminal;
		this.matrixStoragePath = matrixStoragePath;
		this.logger = logger;

		this.matrixStoragePath = matrixStoragePath + "/" + nonTerminal.replaceAll("-", "");
		File matrixDirec = new File(this.matrixStoragePath);
		if (!matrixDirec.exists())
			matrixDirec.mkdirs();
		this.vectorsPath = vectorsPath + "/" + nonTerminal.replaceAll("-", "") + "/outside.ser";
		this.serialize = serialize;
		this.writeToDisk = writeToDisk;
	}

	@Override
	public void run() {
		logger.info("Forming the feature Matrices");
		logger.info("Desearilizing the feature vectors");
		outsideVectors = CommonUtil.getVectors(vectorsPath, logger);

		dprime = outsideVectors.get(0).getFeatureVec().size();
		int M = outsideVectors.size();
		Psi = new SparseMatrixLil(dprime, M);

		// Later change it to only colt
		logger.info("Forming the Matrix (dprime x M): " + Psi.rows + " x " + Psi.cols);
		CommonUtil.formFeatureMatrix(outsideVectors, Psi, logger);

		if (serialize.equalsIgnoreCase("yes")) {
			logger.info("Serializing the Outside Feature Matrix at: " + matrixStoragePath);
			CommonUtil.serializeFeatureMatrix(Psi, matrixStoragePath + "/ofm.ser", logger);
		}

		if (writeToDisk.equalsIgnoreCase("yes")) {
			CommonUtil.writeSparseMatrixToDisk(Psi, matrixStoragePath + "/ofm.txt", nonTerminal, logger);
		}

	}

	public SparseMatrixLil getPsi() {
		return Psi;
	}

}
