/*
 * NumeralCalculations.cpp
 *
 *  Created on: Aug 31, 2019
 *      Author: qzs
 */

#include "NumeralCalculations.h"

NumeralCalculations::NumeralCalculations() {
	// TODO Auto-generated constructor stub
}

NumeralCalculations::~NumeralCalculations() {
	// TODO Auto-generated destructor stub
}

void NumeralCalculations::SolvingNonlinearEquations(const lst &nleqs, const	lst &nxs,const	lst &nxs0, lst &soluts)
{
	size_t n = nleqs.nops();
	matrix dFx(n,n);
	matrix Fx(n,1,nleqs);
	for(size_t i=0; i<n; i++)
	{
		for(size_t j=0; j<n; j++)
		{
			dFx(i,j)=nleqs[i].diff(ex_to<symbol>(nxs[j]));
		}
	}
	dFx=dFx.inverse();
	matrix nxsmat(n,1,nxs0);
	matrix tM, nxsmat1;
	tM = dFx.mul(Fx);
	ex lambd=1,fnt0=0,fnt1=0;
	do{
		exmap mm;
		for(size_t i=0; i<n; i++)
		{
			mm[ex_to<symbol>(nxs[i])]=nxsmat(i,0);
		}
		nxsmat1=nxsmat.sub(ex_to<matrix>(tM.subs(mm)).mul_scalar(lambd));
		ex fn0=0,fn1=0;
		for(size_t i=0; i<n; i++)
		{
			fn0+=pow(nxsmat(i,0),2);
			fn1+=pow(nxsmat1(i,0),2);
		}
		fnt0=fn0,fnt1=fn1;
		lambd/=2;
		//cout<<"lambd:"<<lambd.evalf()<<endl;
	}while((lambd.evalf()> 9.7656E-04)&&(fnt0<fnt1));
	nxsmat=nxsmat1;
	int cnt=10000;
	do
	{
		exmap mm;
		for(size_t i=0; i<n; i++)
		{
			mm[ex_to<symbol>(nxs[i])]=nxsmat(i,0);
		}
	    matrix tmat;
	    tmat=nxsmat.sub(ex_to<matrix>(tM.subs(mm)));
		nxsmat1=tmat.sub(nxsmat),fnt1=0;
		for(size_t i=0; i<n; i++)fnt1+=pow(nxsmat1(i,0),2);
		fnt1 = sqrt(fnt1);
		nxsmat=tmat;
		//cout<<"e:"<<fnt1.evalf()<<endl;
	}while(cnt--<0||fnt1.evalf()>1E-03);

	for(size_t i=0; i<n; i++)
	{
		soluts.append(nxsmat(i,0));
	}
	if(cnt<0)cout<<"NonlinearEquations no solution..."<<endl;
}

void NumeralCalculations::UnconstrainedOptimization(const ex &Oleq, const	lst &nxs,const	lst &nxs0, lst &soluts)
{
	size_t n = nxs.nops();
	lst Grads;
	for(size_t i=0; i<n; i++)
	{
		ex grd=Oleq.diff(ex_to<symbol>(nxs[i]));
		Grads.append(grd);
	}
	SolvingNonlinearEquations(Grads, nxs,nxs0,soluts);
	Grads.remove_all();
}
void NumeralCalculations::ConstrainedOptimizationExternal(const ex &Oleq,const lst Equaleqs,const lst Gretleqs ,const lst &nxs, const	lst &nxs0, lst &soluts)
{
	ex P1=0,P2=0,P=0,M=1,C=10,F;
	for(size_t i=0;i<Equaleqs.nops();i++)P1+=pow(Equaleqs[i],2);
	lst xSoluts, nxs00=nxs0;
	int cnt=1000;
	while(cnt>0)
	{
		P2=0;
		for(size_t i=0;i< Gretleqs.nops();i++)Gretleqs[i].subs(nxs,nxs00)>0?P2+=0:P2+=pow(Gretleqs[i],2);
		P=P1+P2;
		F=Oleq+M*P;
		UnconstrainedOptimization(F, nxs, nxs00,xSoluts);
		ex tp=M*(P.subs(nxs,xSoluts));
		if(tp.evalf()<=1E-04)
		{
			for(size_t i=0; i<nxs.nops(); i++)soluts.append(xSoluts[i]);
			break;
		}
		else
		{
			nxs00=xSoluts;
			M*=C;
			if(cnt!=1)xSoluts.remove_all();
		}
		cnt--;
	}
	cout<<"cnt:"<<(1000-cnt)<<endl;
	if(cnt<=0)
	{
		cout<<"Optimization no solution..."<<endl;
		for(size_t i=0; i<xSoluts.nops(); i++)soluts.append(xSoluts[i]);
	}
	xSoluts.remove_all(),nxs00.remove_all();
}
void NumeralCalculations::SolvingNonlinearEquationsBroyden(const lst &nleqs, const	lst &nxs,const	lst &nxs0, lst &soluts)
{

	size_t n = nleqs.nops();
	matrix dFx(n,n),Bk(n,n),Pk(n,1);
	matrix Fx(n,1,nleqs),Fxt,FxVal;
	ex fnt1=0;
	int cnt=100000;
	for(size_t i=0; i<n; i++)
	{
		for(size_t j=0; j<n; j++)
		{
			dFx(i,j)=nleqs[i].diff(ex_to<symbol>(nxs[j]));
		}
	}
	matrix nxsmat(n,1,nxs0);
	exmap mm;
	for(size_t i=0; i<n; i++)
	{
		mm[ex_to<symbol>(nxs[i])]=nxsmat(i,0);
	}
	Bk=ex_to<matrix>(dFx.subs(mm));
	if(Bk.rank()<n){if(cnt<0)cout<<"NonlinearEquations no solution..."<<endl;return ;}
    Bk=Bk.inverse();
    do
    {
    	exmap mm2;
    	for(size_t i=0; i<n; i++)
    	{
    	   mm2[ex_to<symbol>(nxs[i])]=nxsmat(i,0);
    	}
    	FxVal=ex_to<matrix>(Fx.subs(mm2));

    	Fxt=FxVal;
    	Pk=Bk.mul(FxVal).mul_scalar(-1);
    	nxsmat=nxsmat.add(Pk);

    	exmap mm3;
    	for(size_t i=0; i<n; i++)
    	{
    	   mm3[ex_to<symbol>(nxs[i])]=nxsmat(i,0);
    	}
    	FxVal=ex_to<matrix>(Fx.subs(mm3));

    	fnt1=0;
		for(size_t i=0; i<n; i++)fnt1+=pow(FxVal(i,0),2);
		fnt1 = sqrt(fnt1);
		//cout<<"ft:"<<fnt1<<endl;
		if(fnt1<1E-03)
		{
			for(size_t i=0; i<n; i++)
			{
				soluts.append(nxsmat(i,0));
			}
			break;
		}
		else
		{
			matrix Qk,Tk,Dk;
			Qk=FxVal.sub(Fxt);
			Tk=Pk.sub(Bk.mul(Qk));
			Tk=Tk.mul(Pk.transpose());
			Tk=Tk.mul(Bk);
			Dk=Pk.transpose().mul(Bk).mul(Qk);
			//Dk=Dk.inverse();
			//Tk=Tk.mul_scalar(ex_to<symbol>(Dk(0,0)));
			if(Dk(0,0)==0){cnt=-100;break;}
			Tk=Tk.mul_scalar(1/Dk(0,0));
			Bk=Bk.add(Tk);
		}
		cnt--;
		//cout<<cnt<<endl;
    }while(cnt>0);

    if(cnt<0)cout<<"NonlinearEquations no solution..."<<endl;
}
void NumeralCalculations::UnconstrainedOptimizationBroyden(const ex &Oleq, const	lst &nxs,const	lst &nxs0, lst &soluts)
{

	size_t n = nxs.nops();
	lst Grads;
	for(size_t i=0; i<n; i++)
	{
		ex grd=Oleq.diff(ex_to<symbol>(nxs[i]));
		Grads.append(grd);
	}
	SolvingNonlinearEquationsBroyden(Grads, nxs,nxs0,soluts);
	Grads.remove_all();

}
void NumeralCalculations::ConstrainedOptimizationExternalBroyden(const ex &Oleq,const lst Equaleqs,const lst Gretleqs ,const lst &nxs, const	lst &nxs0, lst &soluts)
{
	ex P1=0,P2=0,P=0,M=1,C=10,F;
	for(size_t i=0;i<Equaleqs.nops();i++)P1+=pow(Equaleqs[i],2);
	lst xSoluts, nxs00=nxs0;
	int cnt=1000;
	while(cnt>0)
	{
		P2=0;
		for(size_t i=0;i< Gretleqs.nops();i++)Gretleqs[i].subs(nxs,nxs00)>0?P2+=0:P2+=pow(Gretleqs[i],2);
		P=P1+P2;
		F=Oleq+M*P;
		UnconstrainedOptimizationBroyden(F, nxs, nxs00,xSoluts);
		ex tp=M*(P.subs(nxs,xSoluts));
		if(tp.evalf()<=1E-04)
		{
			for(size_t i=0; i<nxs.nops(); i++)soluts.append(xSoluts[i]);
			break;
		}
		else
		{
			nxs00=xSoluts;
			M*=C;
			if(cnt!=1)xSoluts.remove_all();
		}
		cnt--;
	}
	cout<<"cnt:"<<(1000-cnt)<<endl;
	if(cnt<=0)
	{
		cout<<"Optimization no solution..."<<endl;
		for(size_t i=0; i<xSoluts.nops(); i++)soluts.append(xSoluts[i]);
	}
	xSoluts.remove_all(),nxs00.remove_all();
}

void NumeralCalculations::SumOfSine(vector<float> &xpts,vector<float> &ypts,lst &nxs0,lst &soluts)
{
    ex Oleq=0;
    int nc=nxs0.nops();
    ex xSmat=symbolic_matrix(1,nc,"x");
    lst xS;
    for(size_t i=0;i<xSmat.nops();i++)xS.append(xSmat[i]);

    for(int i=0;i<int(ypts.size());i++)
     {   for(int j=0;j<nc;j+=3)
        {
            Oleq=Oleq+pow((xS[j]*sin(xS[j+1]*xpts[i]+xS[j+2])-ypts[i]),2);
        }
    }
    Oleq=Oleq/2;
    //cout<<Oleq<<endl;
    UnconstrainedOptimizationBroyden(Oleq, xS,nxs0, soluts);


}
void NumeralCalculations::Polynomail(vector<float> &xpts,vector<float> &ypts,lst &nxs0,lst &soluts)
{
    ex Oleq=0;
    int nc=nxs0.nops();
    ex xSmat=symbolic_matrix(1,nc,"x");
    lst xS;
    for(size_t i=0;i<xSmat.nops();i++)xS.append(xSmat[i]);

    for(int i=0;i<int(ypts.size());i++)
     {
        ex Oleqt=0;
        for(int j=0;j<nc;j++)
        {
            float tp=1.0;
            for(int k=0;k<j;k++)
                tp*=xpts[i];
            Oleqt=Oleqt+xS[j]*tp;
        }
        Oleq+=pow((Oleqt-ypts[i]),2);
    }
    Oleq=Oleq/2;
    UnconstrainedOptimization(Oleq, xS,nxs0, soluts);
    //UnconstrainedOptimizationBroyden(const ex &Oleq, const	lst &nxs,const	lst &nxs0, lst &soluts)
    xS.remove_all();
}

void NumeralCalculations::CubicSplineTrain(vector<float> &xpts, vector<float> &ypts, vector<float> &mcoefs)
{
    vector<float> dVec;dVec.resize(xpts.size());
    vector<float> uVec;uVec.resize(xpts.size());
    vector<float> ldVec;ldVec.resize(xpts.size());
    for(int j=1;j<int(xpts.size());j++)
    {
        float h0=0.0f,h1=0.0f;
        h1=xpts[j+1]-xpts[j];
        h0=xpts[j]-xpts[j-1];
        if(h1==0.0f||h0==0.0f)return;
        dVec[j]=6*((ypts[j+1]-ypts[j])/h1-(ypts[j]-ypts[j-1])/h0)/(h1+h0);
        uVec[j]=h0/(h1+h0),ldVec[j]=1-uVec[j];
    }
    cv::Mat A=cv::Mat::zeros((xpts.size()-2),(xpts.size()-2),CV_32FC1);
    cv::Mat dMat=cv::Mat::zeros((xpts.size()-2),1,CV_32FC1);
    A.at<float>(0,0)=2, A.at<float>(0,1)=ldVec[1];
    A.at<float>((xpts.size()-2)-1,((xpts.size()-2)-1-1))=uVec[(xpts.size()-2)], A.at<float>((xpts.size()-2)-1,(xpts.size()-2)-1)=2;
    dMat.at<float>(0,0)=dVec[1];
    dMat.at<float>(xpts.size()-2-1,0)=dVec[xpts.size()-2];
    for(int i=1;i<int(xpts.size()-2-1);i++)
    {
        A.at<float>(i,i)=2;
        A.at<float>(i,i-1)=uVec[i+1];
        A.at<float>(i,i+1)=ldVec[i+1];
        dMat.at<float>(i,0)=dVec[i+1];
    }
    cv::Mat M;
    M=A.inv()*dMat;
    mcoefs.clear();
    mcoefs.resize(xpts.size()-1);
    for(int i=0;i<M.rows;i++)mcoefs[i+1]=M.at<float>(i,0);

}

void NumeralCalculations::CubicSplineTest(vector<float> &xpts, vector<float> &ypts,vector<float> &mcoefs,float &xpt0,float &ypt0)
{
    int j=0;
    if(xpt0<xpts[0]||xpt0>xpts[xpts.size()-1]){std::cout<<"x0 is wrong"<<"\n"; return;}
    for(j=0;j<int(xpts.size()-1);j++)
    {
        if(xpt0>=xpts[j]&&xpt0<xpts[j+1])
        {
            float h=xpts[j+1]-xpts[j];
            ypt0=mcoefs[j]*gsl_pow_3(xpts[j+1]-xpt0)/(6*h)+
              mcoefs[j+1]*gsl_pow_3(xpt0-xpts[j])/(6*h)+
              (ypts[j]-mcoefs[j]*gsl_pow_2(h)/6)*(xpts[j+1]-xpt0)/h+
              (ypts[j+1]-mcoefs[j+1]*gsl_pow_2(h)/6)*(xpt0-xpts[j])/h;
            break;
        }
    }

}



