package runnables;

import java.io.File;
import java.util.LinkedList;

import org.apache.log4j.Logger;

import beans.FeatureVector;
import jeigen.SparseMatrixLil;
import no.uib.cipr.matrix.DenseVector;
import utils.CommonUtil;
import utils.VSMSparseMatrixLil;

public class OutsideFeatureMatrix implements Runnable {

	String vectorsPath;
	String matrixStoragePath;
	LinkedList<FeatureVector> outsideVectors;
	VSMSparseMatrixLil Psi;
	Logger logger;
	int dprime;
	int M;
	String nonTerminal;
	String isL2Norm;

	public OutsideFeatureMatrix(String nonTerminal, String matrixStoragePath, String vectorsPath, int M, Logger logger,
			String isL2Norm) {
		this.nonTerminal = nonTerminal;
		this.matrixStoragePath = matrixStoragePath;
		this.logger = logger;
		this.M = M;
		this.matrixStoragePath = matrixStoragePath + "/" + nonTerminal;
		File matrixDirec = new File(this.matrixStoragePath);
		if (!matrixDirec.exists())
			matrixDirec.mkdirs();
		this.vectorsPath = vectorsPath + "/" + nonTerminal + "/outside.ser";
		this.isL2Norm = isL2Norm;
	}

	@Override
	public void run() {
		logger.info("Forming the feature Matrices");
		logger.info("Desearilizing the feature vectors");
		outsideVectors = CommonUtil.getVectors(vectorsPath, logger);

		dprime = outsideVectors.get(0).getFeatureVec().size();
		Psi = new VSMSparseMatrixLil(dprime, M);

		logger.info("Forming the Matrix (dprime x M): " + Psi.rows + " x " + Psi.cols);
		CommonUtil.formFeatureMatrix(outsideVectors, Psi, logger);

		CommonUtil.writeFeatureMatrices(Psi, matrixStoragePath + "/ofm.txt", logger);

	}

	public VSMSparseMatrixLil getPsi() {
		return Psi;
	}

}
