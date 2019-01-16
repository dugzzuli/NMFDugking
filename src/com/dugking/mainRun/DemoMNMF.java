package com.dugking.mainRun;

import java.util.ArrayList;
import java.util.List;

import com.dugking.algorithmMNMF.MNMF;

import Jama.Matrix;

public class DemoMNMF {
public static void main(String[] args) {
	List<Matrix> listV=new ArrayList<Matrix>();
	for (int i = 0; i < 3; i++) {
		listV.add(Matrix.random(100, 100).times(100));
	}
	MNMF model=new MNMF(listV, 1000, 3, Math.pow(0.1, 10), Math.pow(0.1, 10));
	model.update();
}
}
