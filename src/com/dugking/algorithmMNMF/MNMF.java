package com.dugking.algorithmMNMF;

import java.util.ArrayList;
import java.util.List;

import Jama.Matrix;

public class MNMF extends MNMF_Base {

	public MNMF(List<Matrix> listV, int maxIter, int clusterNum, double absErr, double relarErr) {
		super(listV, maxIter, clusterNum, absErr, relarErr);

	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
		double lastErr = Double.MAX_VALUE;
		double eps=Math.pow(0.1, 10);
		for (int i = 0; i < this.getMaxIter(); i++) {
			
			List<Matrix> upWAll=new ArrayList<Matrix>();
			for(int j=0;j<this.getViewNum();j++)
			{
				Matrix VTW=this.getListV().get(j).transpose().times(this.getW());
				upWAll.add(VTW);
			}
			
			List<Matrix> downWAll=new ArrayList<Matrix>();
			for(int j=0;j<this.getViewNum();j++)
			{
				Matrix MM=this.getW();
				Matrix downW=this.getListH().get(j).times(MM.transpose()).times(MM);
				downWAll.add(downW);
			}
			
			
			for(int j=0;j<this.getViewNum();j++)
			{
				Matrix updateW=upWAll.get(j).arrayRightDivide(downWAll.get(j));
				updateW=super.updateWreplaceEps(updateW,eps);
				Matrix HH=this.getListH().get(j);
				HH=HH.arrayTimes(updateW);
				this.getListH().set(j, HH);
			}
			
			
			
			Matrix upH=new Matrix(this.getM(), this.getK());
			
			for(int j=0;j<this.getViewNum();j++)
			{
				upH.plusEquals(this.getListV().get(j).times(this.getListH().get(j)));
			}
			Matrix downH=new Matrix(this.getM(), this.getK());
			for(int j=0;j<this.getViewNum();j++)
			{
				Matrix HH=this.getListH().get(j);
				downH.plusEquals(this.getW().times(HH.transpose()).times(HH));
			}
			
			Matrix updateH=upH.arrayRightDivide(downH);
			updateH=super.updateWreplaceEps(updateH,eps);
			Matrix WW=this.getW();
			WW.arrayTimesEquals(updateH);
			
			System.out.println("正在更新..." + i);
			double err = this.getErrOBJ();
			System.out.println("绝对誤差:" + err);

			System.out.println("绝对誤差:" + Math.abs(lastErr - err));
			
			this.getErrAll()[i][1]=err;
			
			if (err < this.getAbsuluteErr()) {
				System.out.println("绝对误差值终止");
				this.setFinalIter(i);
				break;
			}
			if (Math.abs(lastErr - err) < this.getRelarErr()) {
				System.out.println("相对误差值终止");
				this.setFinalIter(i);
				break;
			}
			lastErr = err;
		}
		
	}

	

	@Override
	public double getErrOBJ() {
		double errAllMNMF=0;
		for (int i = 0; i < this.getViewNum(); i++) {
			errAllMNMF=errAllMNMF+this.getNorm(this.getW().times(this.getListH().get(i).transpose()));
		}
		return errAllMNMF;
	}

}
