package com.dugking.Util;

import java.io.Serializable;

import smile.math.distance.Metric;

public class CosineDistance implements Metric<double[]>, Serializable  {

	@Override
	public double d(double[] x, double[] y) {
		// TODO Auto-generated method stub
		double upSum=0;
		for (int i = 0; i < y.length; i++) {
			double x1 = x[i];
			double y1= y[i];
			upSum=upSum+x1*y1;
		}
		double downLeft=0;
		double downRight=0;
		for (int i = 0; i < y.length; i++) {
			double x1 = x[i];
			double y1= y[i];
			downLeft=downLeft+x1*x1;
			downRight=downRight+y1*y1;
			
		}
		downLeft=Math.sqrt(downLeft);
		downRight=Math.sqrt(downRight);
		double cos=upSum/(downLeft*downRight);
		return cos;
	}

}
