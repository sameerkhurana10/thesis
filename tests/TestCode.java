package tests;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Iterator;
import java.util.LinkedList;

import org.j_paine.formatter.CJFormat;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import Jama.Matrix;
import beans.FeatureVector;
import ch.akuhn.edu.mit.tedlab.SMat;
import ch.akuhn.edu.mit.tedlab.Svdlib;
import dictionary.Alphabet;
import jeigen.DenseMatrix;
import jeigen.SparseMatrixLil;
import junit.framework.*;
import no.uib.cipr.matrix.VectorEntry;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import no.uib.cipr.matrix.sparse.SparseVector;
import utils.CommonUtil;

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

	// public void testVecs() {
	// try {
	// ObjectInputStream ois1 = new ObjectInputStream(
	// new
	// FileInputStream("/Users/alt-sameerk/Documents/edinburgh/exp_2016_1/featurevecs/CC/inside.ser"));
	// LinkedList<FeatureVector> beanList = (LinkedList<FeatureVector>)
	// ois1.readObject();
	// for (FeatureVector bean : beanList) {
	// BufferedWriter bw = new BufferedWriter(
	// new FileWriter(System.getProperty("user.dir") + "/testvecs", true));
	// Iterator<VectorEntry> itr = bean.getFeatureVec().iterator();
	// String t = "";
	// while (itr.hasNext()) {
	// VectorEntry e = itr.next();
	// t = t + " " + Integer.toString(e.index()) + ":" +
	// Double.toString(e.get());
	// }
	//
	// t.trim();
	// bw.write(t + "\t" + bean.getVectorDimensions() + "\t" + bean.getTreeIdx()
	// + "\t" + bean.getInsideTree()
	// + "\n");
	// bw.flush();
	// bw.close();
	// }
	//
	// } catch (IOException e) {
	// // TODO Auto-generated catch block
	// e.printStackTrace();
	// } catch (ClassNotFoundException e) {
	// // TODO Auto-generated catch block
	// e.printStackTrace();
	// }
	// }

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

}
