package com.dugking.algorithmNMF;

import java.util.Random;

import Jama.Matrix;
import smile.clustering.KMeans;

/**
 * @author Administrator
 *
 */
public class Single_GNMF {
	private Matrix W, H, V;
	private int k = 0, maxIter = 0, m, n, finalIter = 0;
	private double absErrE = 0, relarErrE = 0;;
	/**
	 * 记录标签
	 */
	private int[] labelCluster;

	public Matrix getWG() {
		return WG;
	}

	public void setWG(Matrix wG) {
		WG = wG;
	}

	public Matrix getDG() {
		return DG;
	}

	public void setDG(Matrix dG) {
		DG = dG;
	}

	/**
	 * 记录所有误差
	 */
	private double[][] errAll;
	/**
	 * 图正则参数
	 */
	private double alpha;
	/**
	 * 图正则 W
	 */
	private Matrix WG;
	/**
	 * 图正则 D
	 */
	private Matrix DG;

	public double getAlpha() {
		return alpha;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	/**
	 * @param V
	 * @param maxIter
	 * @param clusterNum
	 * @param absErr
	 * @param relarErr
	 * @param alpha
	 * @param DG
	 *            图正则 D
	 * @param WG
	 *            图正则 W
	 */
	public Single_GNMF(Matrix V, int maxIter, int clusterNum, double absErr, double relarErr, double alpha, Matrix DG,
			Matrix WG) {
		if (checkIsHasNOn(V)) {
			System.err.println("分解矩陣不能為负数....V");
			return;
		}

		this.V = V;
		this.k = clusterNum;
		this.n = V.getColumnDimension();
		this.m = V.getRowDimension();
		System.out.println("初始化.....W......H");

		this.W = Matrix.random(this.m, this.k);
		this.H = Matrix.random(this.n, this.k);

		this.labelCluster = new int[this.n];

		if (checkIsHasNOn(this.W, this.m, this.k)) {
			System.err.println("分解矩陣不能為负数....E");
			return;
		}

		if (checkIsHasNOn(this.H, this.n, this.k)) {
			System.err.println("分解矩陣不能為负数....H");
			return;
		}
		this.maxIter = maxIter;

		this.absErrE = absErr;
		this.relarErrE = relarErr;

		errAll = new double[this.maxIter][1];
		this.finalIter = this.maxIter;
		this.alpha = alpha;
		this.DG = DG;
		this.WG = WG;

	}

	public void update() {
		double lastErr = Double.MAX_VALUE;
		double eps=Math.pow(0.1, 10);
		for (int i = 0; i < this.maxIter; i++) {
			Matrix up = V.times(H);
			Matrix down = W.times(H.transpose().times(H));

			Matrix WRight = up.arrayRightDivide(down);

			W = W.arrayTimes(WRight);
			
			for (int j = 0; j < this.m; j++) {
				for (int j2 = 0; j2 < this.k; j2++) {
					if(W.get(j, j2)<0)
					{
						W.set(j, j2, eps);
					}
				}

			}	
			
			Matrix upH = WG.times(H).times(alpha);
			Matrix Hup = V.transpose().times(W).plus(upH);
			
			Matrix downH = DG.times(H).times(alpha);
			Matrix Hdown = (H.times(W.transpose())).times(W).plus(downH);
			Matrix HRight = (Hup).arrayRightDivide(Hdown);
			H = H.arrayTimes(HRight);

			
			for (int j = 0; j < this.n; j++) {
				for (int j2 = 0; j2 < this.k; j2++) {
					if(H.get(j, j2)<0)
					{
						H.set(j, j2, eps);
					}
				}

			}
			
			System.out.println("正在更新..." + i);
			
			double graphErr=this.getGraphErr();
			double err = this.getNorm(V.minus(W.times(H.transpose())))+graphErr;
			System.out.println("绝对誤差:" + err);

			System.out.println("绝对誤差:" + Math.abs(lastErr - err));

			if (err < absErrE) {
				System.out.println("绝对误差值终止");
				finalIter = i;
				break;
			}
			if (Math.abs(lastErr - err) < relarErrE) {
				System.out.println("相对误差值终止");
				finalIter = i;
				break;
			}
			lastErr = err;
		}
	}

	public double getNorm(Matrix normV) {
		return normV.normF();

	}

	public boolean checkIsHasNOn(Matrix V) {
		double[][] VM = V.getArray();
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (VM[i][j] < 0) {
					return true;
				}
			}
		}
		return false;
	}

	public boolean checkIsHasNOn(Matrix V, int mm, int nn) {
		double[][] VM = V.getArray();
		for (int i = 0; i < mm; i++) {
			for (int j = 0; j < nn; j++) {
				if (VM[i][j] < 0) {
					return true;
				}
			}
		}
		return false;
	}

	public void getClusterLabel(Matrix clusterInc) {
		KMeans km=new smile.clustering.KMeans(clusterInc.getArray(),this.k);
		this.labelCluster=km.getClusterLabel();
	}
	
	/**
	 * @return  返回图正则错误
	 */
	public double getGraphErr(){
		Matrix L=this.getDG().minus(this.getWG());
		
		return L.normF();
	}
	/**
	 * @param m
	 */
	public void printMatrix(Matrix m) {
		m.print(4, 2);
	}

	public Matrix getW() {
		return W;
	}

	public void setW(Matrix w) {
		W = w;
	}

	public Matrix getH() {
		return H;
	}

	public void setH(Matrix h) {
		H = h;
	}

	public Matrix getV() {
		return V;
	}

	public void setV(Matrix v) {
		V = v;
	}

	public int getK() {
		return k;
	}

	public void setK(int k) {
		this.k = k;
	}

	public int getMaxIter() {
		return maxIter;
	}

	public void setMaxIter(int maxIter) {
		this.maxIter = maxIter;
	}

	public int getM() {
		return m;
	}

	public void setM(int m) {
		this.m = m;
	}

	public int getN() {
		return n;
	}

	public void setN(int n) {
		this.n = n;
	}

	public int getFinalIter() {
		return finalIter;
	}

	public void setFinalIter(int finalIter) {
		this.finalIter = finalIter;
	}

	public double getAbsErrE() {
		return absErrE;
	}

	public void setAbsErrE(double absErrE) {
		this.absErrE = absErrE;
	}

	public double getRelarErrE() {
		return relarErrE;
	}

	public void setRelarErrE(double relarErrE) {
		this.relarErrE = relarErrE;
	}

	public int[] getLabelCluster() {
		return labelCluster;
	}

	public void setLabelCluster(int[] labelCluster) {
		this.labelCluster = labelCluster;
	}

	public double[][] getErrAll() {
		return errAll;
	}

	public void setErrAll(double[][] errAll) {
		this.errAll = errAll;
	}
}
