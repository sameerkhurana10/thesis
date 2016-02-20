package tests;

import junit.framework.*;
import no.uib.cipr.matrix.sparse.SparseVector;

public class TestCode extends TestCase {
	protected int value1, value2;

	protected void setUp() {
		value1 = 3;
		value2 = 3;
	}

	public void testAdd() {
		System.out.println("inside test method");
		double result = value1 + value2;
		assertTrue(result == 6);
	}

	public void testSparseVectors() {
		SparseVector vec = new SparseVector(10);
		System.out.println(vec.get(0));
	}
}