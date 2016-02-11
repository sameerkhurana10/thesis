package dictionary;

/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.  For further
 information, see the file `LICENSE' included with this distribution. */

/** 
 @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 Modifications made by Shay Cohen, for reverse look-up of string.
 Changes made by Shashi Narayan
 */

import java.io.Serializable;

public class Alphabet implements Serializable {

	private static final long serialVersionUID = -5342264414535295402L;

	public gnu.trove.TObjectIntHashMap map;
	public gnu.trove.TIntObjectHashMap reverseMap;
	public gnu.trove.TObjectIntHashMap countMap;

	int capacity = 0;
	int numEntries;

	boolean growthStopped = false;

	public boolean doCount;
	public boolean lockCounts;
	public int minCount;

	public static String NewEntry = "NEWENTRY";

	public Alphabet(int capacity) {
		this.map = new gnu.trove.TObjectIntHashMap(capacity);
		this.reverseMap = new gnu.trove.TIntObjectHashMap(capacity);
		numEntries = 0;
		this.countMap = new gnu.trove.TObjectIntHashMap(capacity);
		this.capacity = capacity;
	}

	public Alphabet() {
		this(10000);
	}

	public void turnOnCounts() {
		countMap = new gnu.trove.TObjectIntHashMap(capacity);
		doCount = true;
		lockCounts = false;
	}

	public void lockCount(int m) {
		lookupIndex("$$absorb$$");
		lockCounts = true;
		minCount = m;
	}

	public void print() {
		int[] keys = reverseMap.keys();
		for (int i = 0; i < keys.length; i++) {
			System.err.println(keys[i] + " = " + reverseMap.get(keys[i]));
		}
	}

	/** Return -1 if entry isn't present. */
	public int lookupIndex(Object entry) {
		if (entry == null) {
			throw new IllegalArgumentException(
					"Can't lookup \"null\" in an Alphabet.");
		}

		int ret = map.get(entry);
	

		if ((doCount) && (lockCounts) && (countMap.get(entry) < minCount)) {
			return map.get("$$absorb$$");
		}

		if (!growthStopped) {
			if (ret == -1 && !growthStopped) {
				ret = numEntries;
				map.put(entry, ret);
				reverseMap.put(ret, entry);
				numEntries++;

				if ((doCount) && (!lockCounts)) {
					countMap.put(entry, 1);
				}
			} else {
				if ((doCount) && (!lockCounts)) {
					countMap.put(entry, countMap.get(entry) + 1);
				}
			}
		}
		return ret;
	}

	public int getCount(Object entry) {
		if (!doCount) {
			System.err
					.println("Warning in Alphabet object: asking for count even though not counting: "
							+ entry);
			return 0;
		}

		return countMap.get(entry);
	}

	public Object reverseLookup(int index) {
		return reverseMap.get(index);
	}

	public Object[] toArray() {
		return map.keys();
	}

	public boolean contains(Object entry) {
		return map.contains(entry);
	}

	public int size() {
		return numEntries;
	}

	public void stopGrowth() {
		growthStopped = true;
		map.compact();
	}

	public void allowGrowth() {
		growthStopped = false;
	}

	public boolean growthStopped() {
		return growthStopped;
	}
	/*
	 * TODO
	 */
	// public double getScaleFactor(int objectId){
	// double k = 5.0;
	// int count = getCount(reverseLookup(objectId));
	// return Math.sqrt(((double) GlobalParameters.hugeFileCount)/(count+k));
	// }

	// // Serialization
	// private void writeObject(ObjectOutputStream out) throws IOException {
	// out.writeInt(numEntries);
	// out.writeObject(map);
	// out.writeObject(reverseMap);
	// out.writeBoolean(growthStopped);
	// }
	//
	// private void readObject(ObjectInputStream in) throws IOException,
	// ClassNotFoundException {
	// numEntries = in.readInt();
	// map = (gnu.trove.TObjectIntHashMap) in.readObject();
	// reverseMap = (gnu.trove.TIntObjectHashMap) in.readObject();
	// growthStopped = in.readBoolean();
	// }
}
