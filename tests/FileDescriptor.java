package tests;

import java.io.*;

import java.util.*;

public class FileDescriptor {
	private static List<InputStream> streams = new ArrayList<InputStream>();

	public static void main(String[] args) {
		for (int i = 0; true; i++) {
			FileInputStream f = null;
			try {
				f = new FileInputStream("/dev/null");
			} catch (Throwable e) {
				System.err.println(e.getMessage());
				e.printStackTrace();
				System.exit(1);
			}
			streams.add(f);
			System.out.println("We have " + (i + 1) + " InputStream's for /dev/null");
		}
	}

}
