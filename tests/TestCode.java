package tests;

import junit.framework.*;

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
}