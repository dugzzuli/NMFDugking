package com.dugking.algorithmMNMF;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import Jama.Matrix;
import smile.clustering.KMeans;

public abstract class MNMF_Base {
	private List<Matrix> listV;
	
	/**
	 * k 聚类个数
	 */
	private int k;
	public int getK() {
		return k;
	}

	public void setK(int k) {
		this.k = k;
	}

	public double getAbsuluteErr() {
		return absuluteErr;
	}

	public void setAbsuluteErr(double absuluteErr) {
		this.absuluteErr = absuluteErr;
	}

	public double getRelarErr() {
		return relarErr;
	}

	public void setRelarErr(double relarErr) {
		this.relarErr = relarErr;
	}

	public int getMaxIter() {
		return maxIter;
	}

	public void setMaxIter(int maxIter) {
		this.maxIter = maxIter;
	}

	public double[][] getErrAll() {
		return errAll;
	}

	public void setErrAll(double[][] errAll) {
		this.errAll = errAll;
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

	public int[] getLabelCluster() {
		return labelCluster;
	}

	public void setLabelCluster(int[] labelCluster) {
		this.labelCluster = labelCluster;
	}

	/**
	 * absuluteErr  绝对误差
	 */
	private double absuluteErr=Math.pow(0.1, 5);
	
	/**
	 * relarErr 相对误差
	 */
	private double relarErr=Math.pow(0.1, 5);
	/**
	 * 最大迭代次数
	 */
	private int maxIter=1000;
	
	/**
	 * 保存所有误差
	 */
	private double[][] errAll;
	/**
	 * m 特征数
	 */
	private int m;
	/**
	 * 样本书
	 */
	private int n;
	
	/**
	 * 最终迭代次数
	 */
	private int finalIter;
	
	
	/**
	 * 记录标签
	 */
	private int[] labelCluster;

	private Matrix W;
	
	
	private Matrix H;
	
	private int viewNum;
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

	public int getViewNum() {
		return viewNum;
	}

	public void setViewNum(int viewNum) {
		this.viewNum = viewNum;
	}

	public List<Matrix> getListH() {
		return listH;
	}

	public void setListH(List<Matrix> listH) {
		this.listH = listH;
	}

	/**
	 *多片NMF 的H 数组
	 */
	private List<Matrix> listH=new ArrayList<Matrix>();
	
	
	/**
	 * @param listV 输入数据
	 * @param maxIter 迭代次数
	 * @param clusterNum 聚类数目
	 * @param absErr 绝对误差
	 * @param relarErr 相对误差
	 */
	public MNMF_Base(List<Matrix> listV,int maxIter, int clusterNum, double absErr, double relarErr){
		this.setListV(listV);
		for (Matrix matrix : listV) {
			this.checkIsHasNOn(matrix);
		}
		this.setViewNum(listV.size());
		this.setMaxIter(maxIter);
		this.setK(clusterNum);
		this.setAbsuluteErr(absErr);
		this.setRelarErr(relarErr);
		this.setM(this.getListV().get(0).getRowDimension());
		this.setN(this.getListV().get(0).getColumnDimension());
		
		this.W = Matrix.random(this.m, this.k);
		this.H = Matrix.random(this.n, this.k);
		
		if (checkIsHasNOn(this.W, this.m, this.k)) {
			System.err.println("分解矩陣不能為负数....E");
			return;
		}
		
		if (checkIsHasNOn(this.H, this.n, this.k)) {
			System.err.println("分解矩陣不能為负数....H");
			return;
		}
		for (int i = 0; i < this.getViewNum(); i++) {
			this.getListH().add(H);
		}
		
		
		this.setErrAll(new double[this.getMaxIter()][1]);
		this.setFinalIter(maxIter);
	}
	
	public List<Matrix> getListV() {
		return listV;
	}

	public void setListV(List<Matrix> listV) {
		this.listV = listV;
	}
	
	/**
	 * @param updateW 替换矩阵
	 * @param eps 替换数值
	 * @return Matrix 替换后的矩阵
	 */
	 Matrix updateWreplaceEps(Matrix updateW, double eps) {
		// TODO Auto-generated method stub
		int mm=updateW.getRowDimension();
		int nn=updateW.getColumnDimension();
		for (int j = 0; j <mm; j++) {
			for (int j2 = 0; j2 < nn; j2++) {
				if(updateW.get(j, j2)<0)
				{
					updateW.set(j, j2, eps);
				}
			}

		}
		return updateW;
	}

	/**
	 * @param V 输入矩阵
	 * @param mm 矩阵行数
	 * @param nn 矩阵列数
	 * @return
	 */
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
	/**
	 * 迭代步骤
	 */
	public abstract void update();
	
	
	/**
	 * @param clusterInc 通过输入矩阵 ，每一行是一个样本
	 */
	public void getClusterLabel(Matrix clusterInc) {
		KMeans km=new smile.clustering.KMeans(clusterInc.getArray(),this.k);
		this.labelCluster=km.getClusterLabel();
	}
	
	/**
	 * @param normV 需要计算范数的和
	 * @return
	 */
	public double getNorm(Matrix normV) {
		return normV.normF();

	}
	/**
	 * @param V 检测一个矩阵是否含有负数
	 * @return
	 */
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
	
	/**
	 * @return 获取损失函数误差
	 */
	public abstract double getErrOBJ();
	
	
	
}
