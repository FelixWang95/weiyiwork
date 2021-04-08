/*
 * NumeralCalculations.h
 *
 *  Created on: Aug 31, 2019
 *      Author: qzs
 */

#ifndef NUMERALCALCULATIONS_H_
#define NUMERALCALCULATIONS_H_
#include"headerall.h"
class NumeralCalculations {
public:
	NumeralCalculations();
	virtual ~NumeralCalculations();
	void SolvingNonlinearEquations(const lst &nleqs, const	lst &nxs,const	lst &nxs0, lst &soluts);
	void UnconstrainedOptimization(const ex &Oleq, const	lst &nxs,const	lst &nxs0, lst &soluts);
	void ConstrainedOptimizationExternal(const ex &Oleq,const lst Equaleqs,const lst Gretleqs ,const lst &nxs, const	lst &nxs0, lst &soluts);
	void SolvingNonlinearEquationsBroyden(const lst &nleqs, const	lst &nxs,const	lst &nxs0, lst &soluts);
	void UnconstrainedOptimizationBroyden(const ex &Oleq, const	lst &nxs,const	lst &nxs0, lst &soluts);
    void ConstrainedOptimizationExternalBroyden(const ex &Oleq,const lst Equaleqs,const lst Gretleqs ,const lst &nxs, const	lst &nxs0, lst &soluts);
    void SumOfSine(vector<float> &xpts,vector<float> &ypts,lst &nxs0,lst &soluts);
    void Polynomail(vector<float> &xpts,vector<float> &ypts,lst &nxs0,lst &soluts);
    void CubicSplineTrain(vector<float> &xpts,vector<float> &ypts,vector<float> &mcoefs);
    void CubicSplineTest(vector<float> &xpts, vector<float> &ypts,vector<float> &mcoefs,float &xpt0,float &ypt0);
private:

};

#endif /* NUMERALCALCULATIONS_H_ */
