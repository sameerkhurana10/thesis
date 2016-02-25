package utils;

import java.io.Serializable;

import jeigen.SparseMatrixLil;

public class VSMSparseMatrixLil extends SparseMatrixLil implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public VSMSparseMatrixLil(int rows, int cols) {
		super(rows, cols);
	}

}
