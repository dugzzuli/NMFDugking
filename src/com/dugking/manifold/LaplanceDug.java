package com.dugking.manifold;

/*******************************************************************************
 * Copyright (c) 2010 Haifeng Li
 *   
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *  
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
import java.util.Collection;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.dugking.Util.CosineDistance;
import com.dugking.Util.GraphType;

import smile.data.SparseDataset;
import smile.graph.AdjacencyList;
import smile.graph.Graph;
import smile.graph.Graph.Edge;
import smile.math.Math;
import smile.math.SparseArray;
import smile.math.distance.EuclideanDistance;
import smile.math.matrix.DenseMatrix;
import smile.math.matrix.EVD;
import smile.math.matrix.Matrix;
import smile.math.matrix.SparseMatrix;
import smile.neighbor.CoverTree;
import smile.neighbor.KDTree;
import smile.neighbor.KNNSearch;
import smile.neighbor.Neighbor;


public class LaplanceDug {
	private static final Logger logger = LoggerFactory.getLogger(LaplanceDug.class);

	private double[][] wDug;
	private double[][] dDug;
	private double[][] lDug;

	public Jama.Matrix getwDug() {
		return new Jama.Matrix(wDug);
	}

	public void setwDug(double[][] wDug) {
		this.wDug = wDug;
	}

	public Jama.Matrix getdDug() {
		return new Jama.Matrix(dDug);
	}

	public void setdDug(double[][] dDug) {
		this.dDug = dDug;
	}

	/**
	 * The width of heat kernel.
	 */
	private double t;
	/**
	 * The original sample index.
	 */
	private int[] index;
	/**
	 * Coordinate matrix.
	 */
	private double[][] coordinates;
	/**
	 * Nearest neighbor graph.
	 */
	private Graph graph;

	/**
	 * Constructor. Learn Laplacian Eigenmaps with discrete weights.
	 * 
	 * @param data
	 *            the dataset.
	 * @param d
	 *            the dimension of the manifold.
	 * @param k
	 *            k-nearest neighbor.
	 */
	public LaplanceDug(double[][] data, int d, int k) {
		this(data, d, k, -1);
	}

	/**
	 * Constructor. Learn Laplacian Eigenmap with Gaussian kernel.
	 * 
	 * @param data
	 *            the dataset.
	 * @param d
	 *            the dimension of the manifold.
	 * @param k
	 *            k-nearest neighbor.
	 * @param t
	 *            the smooth/width parameter of heat kernel e<sup>-||x-y||
	 *            <sup>2</sup> / t</sup>. Non-positive value means discrete
	 *            weights.
	 */
	public LaplanceDug(double[][] data, int d, int k, double t) {
		this.t = t;

		int n = data.length;
		KNNSearch<double[], double[]> knn = null;
		if (data[0].length < 10) {
			knn = new KDTree<>(data, data);
		} else {
			knn = new CoverTree<>(data, new EuclideanDistance());
		}

		graph = new AdjacencyList(n);
		for (int i = 0; i < n; i++) {
			Neighbor<double[], double[]>[] neighbors = knn.knn(data[i], k);
			for (int j = 0; j < k; j++) {
				graph.setWeight(i, neighbors[j].index, neighbors[j].distance);
			}
		}
		SparseDataset W = new SparseDataset(n);
		double[] D = new double[n];
		double gamma = -1.0 / t;

		for (int i = 0; i < n; i++) {
			Collection<Edge> edges = graph.getEdges(i);
			for (Edge edge : edges) {
				int j = edge.v2;
				if (i == j) {
					j = edge.v1;
				}

				double w = t <= 0 ? 1.0 : Math.exp(gamma * Math.sqr(edge.weight));
				W.set(i, j, w);

				D[i] += w;
			}

			D[i] = 1 / Math.sqrt(D[i]);
		}
		this.setdDug(SparseMatrix.diag(D).array());
		this.setwDug(W.toArray());

	}

	/**
	 * Constructor. Learn Laplacian Eigenmap with Gaussian kernel.
	 * 
	 * @param data
	 *            the dataset.
	 * @param k
	 *            k-nearest neighbor.
	 * @param t
	 *            the smooth/width parameter of heat kernel e<sup>-||x-y||
	 *            <sup>2</sup> / t</sup>. Non-positive value means discrete
	 *            weights.
	 * @param type
	 *            表示使用哪一种权重
	 */
	public LaplanceDug(double[][] data, int k, double t, GraphType type) {
		this.t = t;

		int n = data.length;
		wDug=new double[n][n];
		dDug=new double[n][n];
		KNNSearch<double[], double[]> knn = null;
		if (data[0].length < 10) {
			knn = new KDTree<>(data, data);
		} else {
			if (type == GraphType.HeartKernel)
				knn = new CoverTree<>(data, new EuclideanDistance());
			else if (type == GraphType.Cosine) {
				knn = new CoverTree<>(data, new CosineDistance());
			} else if (type == GraphType.Binary) {
				knn = new CoverTree<>(data, new EuclideanDistance());
			}
		}

		graph = new AdjacencyList(n);
		for (int i = 0; i < n; i++) {
			Neighbor<double[], double[]>[] neighbors = knn.knn(data[i], k);
			for (int j = 0; j < k; j++) {
				graph.setWeight(i, neighbors[j].index, neighbors[j].distance);
			}
		}
		SparseDataset W = new SparseDataset(n);
		double[] D = new double[n];
		double[] DD = new double[n];
		double gamma = -1.0 / t;

		for (int i = 0; i < n; i++) {
			Collection<Edge> edges = graph.getEdges(i);
			for (Edge edge : edges) {
				int j = edge.v2;
				if (i == j) {
					j = edge.v1;
				}
				double w;
				switch (type) {
				case Binary:
					w = 1.0;
					W.set(i, j, w);
					D[i] += w;
					break;
				case HeartKernel:
					w = t <= 0 ? 1.0 : Math.exp(gamma * Math.sqr(edge.weight));
					W.set(i, j, w);
					D[i] += w;

					break;
				case Cosine:
					w = edge.weight;
					W.set(i, j, w);
					D[i] += w;

					break;
				default:
					break;
				}

			}
			DD[i]=D[i];
			D[i] = 1 / Math.sqrt(D[i]);
		}
		
		this.setdDug(SparseMatrix.diag(DD).array());
		this.setwDug(W.toArray());

	}

	/**
	 * Returns the original sample index. Because Laplacian Eigenmap is applied
	 * to the largest connected component of k-nearest neighbor graph, we record
	 * the the original indices of samples in the largest component.
	 */
	public int[] getIndex() {
		return index;
	}

	/**
	 * Returns the coordinates of projected data.
	 */
	public double[][] getCoordinates() {
		return coordinates;
	}

	/**
	 * Returns the nearest neighbor graph.
	 */
	public Graph getNearestNeighborGraph() {
		return graph;
	}

	/**
	 * Returns the width of heat kernel.
	 */
	public double getHeatKernelWidth() {
		return t;
	}
}
