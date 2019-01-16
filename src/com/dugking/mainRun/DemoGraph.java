package com.dugking.mainRun;

import com.dugking.Util.FileUtil;
import com.dugking.Util.GraphType;
import com.dugking.manifold.ConstrucGraph;

import Jama.Matrix;

public class DemoGraph {
	public static void main(String[] args) {
		Matrix randData = FileUtil.CSV2Matrix("src/com/dugking/resource/graph.csv", 300, 2);
		// Matrix randData=Matrix.random(20, 20).times(100);
		ConstrucGraph modelGraph = new ConstrucGraph(randData.getArray(), 7, 7, GraphType.HeartKernel);
		modelGraph.getGraphKnn().print(4, 2);
		System.out.println(modelGraph.getGraphKnn().normF());
		
	}
}
