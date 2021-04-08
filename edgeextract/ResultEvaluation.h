#pragma once
#include <vector>
struct CalcResult
{
	int a, b, c, d;
	int n1, n2;
	double R_p, P_p, F1_p;
	double R_n, P_n, F1_n;
};

class ResultEvaluation
{
public:
	ResultEvaluation() = default;
	~ResultEvaluation() = default;
	CalcResult operator()(const std::vector<int> &realValues, const std::vector<int> &predictValues);
};

