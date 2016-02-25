package utils;

import java.io.Serializable;

import no.uib.cipr.matrix.DenseVector;

public class VSMDenseVector extends DenseVector implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public VSMDenseVector(double[] x) {
		super(x);
	}

}
