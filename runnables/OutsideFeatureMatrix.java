package runnables;

import java.io.File;
import java.util.LinkedList;

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
	int M;
	String nonTerminal;

	public OutsideFeatureMatrix(String nonTerminal, String matrixStoragePath, String vectorsPath, int M,
			Logger logger) {
		this.nonTerminal = nonTerminal;
		this.matrixStoragePath = matrixStoragePath;
		this.logger = logger;
		this.M = M;
		this.matrixStoragePath = matrixStoragePath + "/" + nonTerminal.replaceAll("-", "");
		File matrixDirec = new File(this.matrixStoragePath);
		if (!matrixDirec.exists())
			matrixDirec.mkdirs();
		this.vectorsPath = vectorsPath + "/" + nonTerminal.replaceAll("-", "") + "/outside.ser";
	}

	@Override
	public void run() {
		logger.info("Forming the feature Matrices");
		logger.info("Desearilizing the feature vectors");
		outsideVectors = CommonUtil.getVectors(vectorsPath, logger);

		dprime = outsideVectors.get(0).getFeatureVec().size();
		Psi = new SparseMatrixLil(dprime, M);

		// Later change it to only colt
		logger.info("Forming the Matrix (dprime x M): " + Psi.rows + " x " + Psi.cols);
		CommonUtil.formFeatureMatrix(outsideVectors, Psi, logger);

		logger.info("Serializing the Outside Feature Matrix at: " + matrixStoragePath);
		CommonUtil.serializeFeatureMatrix(Psi, matrixStoragePath + "/ofm.ser", logger);

	}

	public SparseMatrixLil getPsi() {
		return Psi;
	}

}
