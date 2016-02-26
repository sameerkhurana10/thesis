package tests;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.stat.descriptive.moment.Mean;

import Jama.Matrix;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import ch.akuhn.edu.mit.tedlab.SMat;
import ch.akuhn.edu.mit.tedlab.Svdlib;
import dictionary.Alphabet;
import jeigen.DenseMatrix;
import jeigen.SparseMatrixLil;
import junit.framework.TestCase;
import no.uib.cipr.matrix.VectorEntry;
import no.uib.cipr.matrix.sparse.SparseVector;
import utils.CommonUtil;
import utils.MatrixHelper;
import utils.VSMSparseVector;

public class TestCode extends TestCase {
	protected int value1, value2;

	protected void setUp() {
		value1 = 3;
		value2 = 3;
	}

	public void testAdd() {
		// System.out.println("inside test method");
		double result = value1 + value2;
		assertTrue(result == 6);
	}

	public void testSparseVectors() {
		SparseVector vec = new SparseVector(10);
		// System.out.println(vec.get(0));
	}

	public void testSparseMatrixLil() {
		SparseMatrixLil xjeig = SparseMatrixLil.sprand(2, 3);
		System.out.println(xjeig);
		for (int i = 0; i < xjeig.getSize(); i++) {
			System.out.println((xjeig.getRowIdx(i)) + " " + (xjeig.getColIdx(i)) + " " + xjeig.getValue(i));
		}
	}

	public void testDictionary() {
		Alphabet a = new Alphabet();
		a.allowGrowth();
		a.turnOnCounts();
		a.lookupIndex("test1");
		a.lookupIndex("test1");
		a.map.put("test1", 20);

		a.lookupIndex("test1");
		a.lookupIndex("test1");

		a.lookupIndex("test2");
		a.map.put("test2", 30);

		a.stopGrowth();
		// System.out.println(a.countMap.get("test2"));
		// System.out.println(a.lookupIndex("test2"));
	}

	public void testSVD() {
		Svdlib s = new Svdlib();
		SMat x = new SMat(2, 3, 3);
		x.pointr[0] = 0;
		x.rowind[0] = 1;
		x.rowind[1] = 9;

	}

	public void testDense() throws ClassNotFoundException {
		double[][] n = new double[2][1];
		n[0][0] = 1.0;
		n[1][0] = 2.0;
		DenseMatrix x = new DenseMatrix(n);
		Matrix m = CommonUtil.createDenseMatrixJAMA(x);
		try {
			ObjectOutputStream os = new ObjectOutputStream(
					new FileOutputStream(System.getProperty("user.dir") + "/test.ser"));
			os.writeObject(m);
			os.close();
			ObjectInputStream ois = new ObjectInputStream(
					new FileInputStream(System.getProperty("user.dir") + "/test.ser"));
			Matrix m1 = (Matrix) ois.readObject();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void testSparse() {
		VSMSparseVector v = new VSMSparseVector(3);
		v.add(0, 1.0);
		v.add(1, 2.0);
		v.add(2, 0.0);

		VSMSparseVector v1 = new VSMSparseVector(3);
		v1.add(0, 3.0);
		v1.add(1, 7.0);

		double[] d1 = new double[3];
		Iterator<VectorEntry> itr = v.iterator();
		int i = 0;
		while (itr.hasNext()) {
			VectorEntry e = itr.next();
			d1[i] = e.get();
			i++;
		}

		System.out.println(d1.length);

		double[] d2 = new double[3];
		Iterator<VectorEntry> itr1 = v1.iterator();
		int k = 0;
		while (itr1.hasNext()) {
			VectorEntry e = itr1.next();
			d2[k] = e.get();
			k++;
		}

	}

	public void testCernLA() {
		DenseDoubleMatrix1D x = new DenseDoubleMatrix1D(new double[] { 1, 2, 3 });
		DenseDoubleMatrix1D y = new DenseDoubleMatrix1D(new double[] { 5, 7, 0, 11, 1, 9 });
		DenseDoubleMatrix2D c = new DenseDoubleMatrix2D(x.size(), y.size());

		System.out.println(x + "\t" + y);
		Algebra a = new Algebra();
		Mean m = new Mean();
		double[] mean = new double[x.size()];
		Arrays.fill(mean, (-m.evaluate(x.toArray())));
		DenseDoubleMatrix1D ux = new DenseDoubleMatrix1D(mean);
		x.assign(MatrixHelper.addVectors(x, ux));

		double[] meany = new double[y.size()];
		Arrays.fill(meany, (-m.evaluate(y.toArray())));
		DenseDoubleMatrix1D uy = new DenseDoubleMatrix1D(meany);
		y.assign(MatrixHelper.addVectors(y, uy));

		System.out.println(x + "\t" + y);

		a.multOuter(x, y, c);
		System.out.println(c);
	}

	public void testSer() throws ClassNotFoundException {
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(
					"/Users/alt-sameerk/Documents/edinburgh/exp_2016_1/featurematrices/CC/covM.ser"));
			cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D mat = (cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D) ois
					.readObject();
			System.out.println("CERN++ " + mat.get(0, 0));
			ois.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
