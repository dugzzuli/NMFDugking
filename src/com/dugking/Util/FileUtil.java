package com.dugking.Util;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLDouble;

import Jama.Matrix;

public class FileUtil {

	public static void Matrix2CSV(Matrix V, String path) throws IOException, FileNotFoundException {
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path), "UTF-8"));
		int m = V.getRowDimension();
		int n = V.getColumnDimension();
		double[][] arr = V.getArray();
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {

				out.write(String.valueOf(arr[i][j]));
				out.write(",");
			}
			out.newLine();
		}
		out.flush();
		out.close();
	}

	public static void Arr2CSV(int[] arr, String path) throws IOException, FileNotFoundException {
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path), "UTF-8"));

		for (int k = 0; k < arr.length; k++) {
			int l = arr[k];
			out.write(String.valueOf(l));
			out.newLine();
		}
		out.flush();
		out.close();
	}

	// 导出到csv文件
	public static void Array2CSV(ArrayList<ArrayList<String>> data, String path) {
		try {
			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path), "UTF-8"));
			for (int i = 0; i < data.size(); i++) {
				ArrayList<String> onerow = data.get(i);
				for (int j = 0; j < onerow.size(); j++) {
					out.write(DelQuota(onerow.get(j)));
					out.write(",");
				}
				out.newLine();
			}
			out.flush();
			out.close();

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public static String DelQuota(String str) {
		String result = str;
		String[] strQuota = { "~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "`", ";", "'", ",", ".", "/", ":",
				"/,", "<", ">", "?" };
		for (int i = 0; i < strQuota.length; i++) {
			if (result.indexOf(strQuota[i]) > -1)
				result = result.replace(strQuota[i], "");
		}
		return result;
	}

	public static ArrayList<ArrayList<String>> CSV2Array(String path) {
		try {
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(path), "UTF-8"));
			ArrayList<ArrayList<String>> alldata = new ArrayList<ArrayList<String>>();
			String line;
			String[] onerow;
			while ((line = in.readLine()) != null) {
				onerow = line.split(","); // 默认分割符为逗号，可以不使用逗号
				List<String> onerowlist = Arrays.asList(onerow);
				ArrayList<String> onerowaArrayList = new ArrayList<String>(onerowlist);
				alldata.add(onerowaArrayList);
			}
			in.close();
			return alldata;
		} catch (Exception e) {
			return null;
		}

	}

	public static int[] CSV2intArr(String path, int m,boolean head) {
		try {
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(path), "UTF-8"));
			int[] arr = new int[m];
			String line;
			int i = 0;
			while ((line = in.readLine()) != null) {
				if(head==true)
				{
					head=false;
					continue;
					
				}
				arr[i] = Integer.valueOf(line.trim());
				i = i + 1;
			}
			in.close();
			return arr;
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}

	}

	public static Matrix CSV2Matrix(String path, int m, int n) {
		try {
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(path), "UTF-8"));
			double[][] arr = null;
			String line;
			String[] onerow;
			int i = 0;
			while ((line = in.readLine()) != null) {
				onerow = line.split(","); // 默认分割符为逗号，可以不使用逗号
				if (i == 0) {
					n = onerow.length;
					arr = new double[m][n];
				}
				for (int j = 0; j < onerow.length; j++) {
					String d = onerow[j];
					arr[i][j] = Double.valueOf(d);
				}
				i = i + 1;
			}
			in.close();
			return new Matrix(arr);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}

	}
	
	public static Matrix loadDataSetTrain(String path, int m, int n,boolean head) {
		try {
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(path), "UTF-8"));
			double[][] arr = null;
			String line;
			String[] onerow;
			int i = 0;
			while ((line = in.readLine()) != null) {
				onerow = line.split(","); // 默认分割符为逗号，可以不使用逗号
				if (i == 0) {
					n = onerow.length;
					arr = new double[m][n];
				}
				if(head==true)
				{
					head=false;
					continue;
				}
				for (int j = 0; j < onerow.length; j++) {
					String d = onerow[j];
					arr[i][j] = Double.valueOf(d);
				}
				i = i + 1;
			}
			in.close();
			return new Matrix(arr,m,n);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}

	}
	
	
	
	/**
	 * @param path
	 * @param key
	 * @param label
	 * @return  获取训练集和测试集合标签
	 */
	@SuppressWarnings("finally")
	public static Map<String, ArrayList<double[][]>> getMatCell2ArrayList(String path, String key, String label) {
		Map<String, ArrayList<double[][]>> dict = new HashMap<String, ArrayList<double[][]>>();
		try {
			MatFileReader read = new MatFileReader(path);
			MLCell ml = (MLCell) read.getContent().get(key);
			ArrayList<MLArray> list = ml.cells();
			ArrayList<double[][]> listData = new ArrayList<double[][]>();
			for (MLArray mlArray : list) {
				MLDouble dd = (MLDouble) mlArray;
				listData.add(dd.getArray());
			}
			dict.put(key, listData);

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		try {
			MatFileReader read = new MatFileReader(path);
			MLCell ml = (MLCell) read.getContent().get(label);
			ArrayList<MLArray> list = ml.cells();
			ArrayList<double[][]> listData = new ArrayList<double[][]>();
			for (MLArray mlArray : list) {
				MLDouble dd = (MLDouble) mlArray;
				listData.add(dd.getArray());
			}
			dict.put(label, listData);

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return dict;

	}

	/**
	 * @param path 读取文件路径
	 * @param key 
	 * @return 获取cell 转为List
	 */
	public static Map<String, ArrayList<double[][]>> getMatCell2ArrayList(String path, String key) {
		Map<String, ArrayList<double[][]>> dict = new HashMap<String, ArrayList<double[][]>>();
		try {
			MatFileReader read = new MatFileReader(path);
			MLCell ml = (MLCell) read.getContent().get(key);
			ArrayList<MLArray> list = ml.cells();
			ArrayList<double[][]> listData = new ArrayList<double[][]>();
			for (MLArray mlArray : list) {
				MLDouble dd = (MLDouble) mlArray;
				listData.add(dd.getArray());
			}
			dict.put(key, listData);

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return dict;
	}

	/**
	 * @param path  文件路径
	 * @param key 获取数组的名称
	 * @return 返回获取的当前的二维数组
	 */
	public static double[][] getMatrix2ArrayList(String path, String key) {
		double[][] dict = null;
		try {
			MatFileReader read = new MatFileReader(path);
			MLArray ml = (MLArray) read.getContent().get(key);

			MLDouble dd = (MLDouble) ml;
			dict = dd.getArray();

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return dict;
	}

}
