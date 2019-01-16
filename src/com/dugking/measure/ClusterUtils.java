package com.dugking.measure;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * DATE: 16-6-18 TIME: ÉÏÎç11:07
 */
public class ClusterUtils {
	public static int combination(int n, int k) {
		if (k > n) {
			return 0;
		}
		int[] data = new int[n + 1];
		data[0] = 1;
		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j >= 1; j--) {
				data[j] += data[j - 1];
			}
		}
		return data[k];
	}

	public static int computeTPAndFP(int[] clusters) {
		int result = 0;
		for (int cluster : clusters) {
			result += combination(cluster, 2);
		}
		return result;
	}

	public static int computeFP(List<Map<Integer, Integer>> mapList) {
		int FP = 0;
		for (Map<Integer, Integer> map : mapList) {
			for (Integer integer : map.values()) {
				if (integer >= 2) {
					FP += combination(integer, 2);
				}
			}
		}
		return FP;
	}

	public static int computeOneClass(List<Integer> list) {
		int n = list.size();
		if (n == 0) {
			return 0;
		}
		int result = 0;
		for (int i = 0; i < n - 1; i++) {
			for (int j = i + 1; j < n; j++) {
				result += list.get(i) * list.get(j);
			}
		}
		return result;
	}

	public static int computeFN(List<List<Integer>> lists) {
		int result = 0;
		for (List<Integer> list : lists) {
			result += computeOneClass(list);
		}
		return result;
	}

	public static double computeFValue(double P, double R, double beta) {
		return (beta * beta + 1) * P * R / (beta * beta * P + R);
	}

	public static void main(String[] args) {
		List<Integer> list = Arrays.asList(1, 4, 0);
		System.out.println("computeOneClass(list) = " + computeOneClass(list));
	}
}
