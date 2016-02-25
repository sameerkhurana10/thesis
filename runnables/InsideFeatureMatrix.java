package runnables;

import java.io.File;
import java.util.LinkedList;

import org.apache.log4j.Logger;

import beans.FeatureVector;
import utils.CommonUtil;
import utils.VSMSparseMatrixLil;

public class InsideFeatureMatrix implements Runnable {

	String vectorsPath;
	String matrixStoragePath;
	LinkedList<FeatureVector> insideVectors;
	VSMSparseMatrixLil Phi;
	Logger logger;
	int d;
	int M;
	String nonTerminal;

	public InsideFeatureMatrix(String nonTerminal, String matrixStoragePath, String vectorsPath, int M, Logger logger) {
		this.nonTerminal = nonTerminal;
		this.matrixStoragePath = matrixStoragePath;
		this.logger = logger;
		this.M = M;
		this.matrixStoragePath = matrixStoragePath + "/" + nonTerminal;
		File matrixDirec = new File(this.matrixStoragePath);
		if (!matrixDirec.exists())
			matrixDirec.mkdirs();
		this.vectorsPath = vectorsPath + "/" + nonTerminal + "/inside.ser";
	}

	@Override
	public void run() {
		logger.info("Forming the Inside feature Matrices");

		logger.info("Desearilizing the feature vectors");
		insideVectors = CommonUtil.getVectors(vectorsPath, logger);

		d = insideVectors.get(0).getFeatureVec().size();
		Phi = new VSMSparseMatrixLil(d, M);

		logger.info("Forming the Matrix (d x M): " + Phi.rows + " x " + Phi.cols);
		CommonUtil.formFeatureMatrix(insideVectors, Phi, logger);

		logger.info("Serializing the Inside Feature Matrix at: " + matrixStoragePath);
		// CommonUtil.serializeFeatureMatrix(Phi, matrixStoragePath +
		// "/ifm.ser", logger);
		CommonUtil.writeFeatureMatrices(Phi, matrixStoragePath + "/ifm.txt", logger);

	}

	public VSMSparseMatrixLil getPhi() {
		return Phi;
	}

}
