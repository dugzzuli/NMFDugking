package com.dugking.mainRun;

import java.io.FileNotFoundException;
import java.io.IOException;

import com.dugking.Util.FileUtil;
import com.dugking.Util.GraphType;
import com.dugking.algorithmNMF.Single_GNMF;
import com.dugking.manifold.LaplanceDug;
import com.dugking.measure.ClusterEvaluation;

import Jama.Matrix;
import smile.clustering.KMeans;

public class RunMain {
	public static void main(String[] args) throws FileNotFoundException, IOException {
		Matrix W = FileUtil.loadDataSetTrain("src/com/dugking/resource/iris.csv", 150, 4, true);
		int[] label=FileUtil.CSV2intArr("src/com/dugking/resource/label.csv", 150,true);
		LaplanceDug eigenmap = new LaplanceDug(W.getArray(), 5, 0.5,GraphType.Binary);
		Matrix d=eigenmap.getdDug();
		Matrix w=eigenmap.getwDug();
		Single_GNMF model = new Single_GNMF(W.transpose(), 1000,3,Math.pow(0.1, 5),Math.pow(0.1, 5),10,d,w);
		model.update();

		model.getClusterLabel(model.getH());
		double RandIndex=new smile.validation.RandIndex().measure(model.getLabelCluster(), label);
		double NMI2=ClusterEvaluation.NMI(model.getLabelCluster(), label);
		System.out.println("∂‘”¶RandIndex:"+RandIndex+","+NMI2);
		FileUtil.Arr2CSV(model.getLabelCluster(), "src/com/dugking/resource/pre.csv");
		FileUtil.Matrix2CSV(model.getH(), "src/com/dugking/resource/H.csv");
	}
}
