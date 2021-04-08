#include "ResultEvaluation.h"
#include <QtAlgorithms>
#include <memory.h>

CalcResult ResultEvaluation::operator()(const std::vector<int> &realValues, const std::vector<int> &predictValues)
{
	CalcResult result;
	memset(&result, 0, sizeof result);
	if (realValues.size() != predictValues.size()) return result;
	
	for (int i = 0; i < realValues.size(); i++)
	{
		if (realValues[i] == 1 && predictValues[i] == 1) ++result.a;
		else if (realValues[i] == 1 && predictValues[i] == -1) ++result.b;
		else if (realValues[i] == -1 && predictValues[i] == 1) ++result.c;
		else if (realValues[i] == -1 && predictValues[i] == -1) ++result.d;
	}
	result.n1 = result.a + result.b;
	result.n2 = result.c + result.d;

	auto f = [](double r, double p)->double {
		if ((r == 0 && p == 0) || (r == -1 && p == -1)) return -1.;
		return (double)(2 * r * p) / (r + p);
	};

	auto nan = [](double& d) {
		if (qIsNaN(d))
			d = -1;
	};

	// positive
	result.R_p = (double)result.a / ((long long)result.a + result.b);
	result.P_p = (double)result.a / ((long long)result.a + result.c);

	nan(result.R_p), nan(result.P_p);
	result.F1_p = f(result.R_p, result.P_p);
	//negative
	result.R_n = (double)result.d / ((long long)result.c + result.d);
	result.P_n = (double)result.d / ((long long)result.b + result.d);

	nan(result.R_n), nan(result.P_n);
	result.F1_n = f(result.R_n, result.P_n);

	return result;
}
