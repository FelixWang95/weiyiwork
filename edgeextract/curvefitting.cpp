#include "curvefitting.h"


curvefitting::curvefitting()
{
}


curvefitting::~curvefitting()
{
}
void curvefitting::operator()(std::vector<std::vector<cv::Point2d>> &ptVec,std::vector<std::vector<float>> &res)
{

	std::ofstream ofs("abc.txt");
    res.clear();
	for (size_t i = 0; i < ptVec.size(); i++)
	{
		cv::Mat A = cv::Mat::zeros(3, 3, CV_64FC1);
		cv::Mat B = cv::Mat::zeros(3, 1, CV_64FC1);
		cv::Mat V;
        std::vector<float> singleres(4,0);
		for (size_t j = 0; j < ptVec[i].size(); j++)
		{
			A.at<double>(0, 0) += pow((ptVec[i][j].x),4);
			A.at<double>(0, 1) += pow((ptVec[i][j].x), 3);
			A.at<double>(0, 2) += pow((ptVec[i][j].x), 2);
			A.at<double>(1, 0) += pow((ptVec[i][j].x), 3);
			A.at<double>(1, 1) += pow((ptVec[i][j].x), 2);
			A.at<double>(1, 2) += pow((ptVec[i][j].x), 1);
			A.at<double>(2, 0) += pow((ptVec[i][j].x), 2);
			A.at<double>(2, 1) += pow((ptVec[i][j].x), 1);
			A.at<double>(2, 2) += pow((ptVec[i][j].x), 0);
			B.at<double>(0, 0) += (ptVec[i][j].y*pow((ptVec[i][j].x), 2));
			B.at<double>(1, 0) += (ptVec[i][j].y*pow((ptVec[i][j].x), 1));
			B.at<double>(2, 0) += (ptVec[i][j].y*pow((ptVec[i][j].x), 0));
		}
		solve(A, B, V, CV_SVD);
		//std::cout << V.at<double>(0, 0) << " " << V.at<double>(1, 0) << " " << V.at<double>(2, 0) << std::endl;

        singleres[0]=V.at<double>(0, 0);
        singleres[1]=V.at<double>(1, 0);
        singleres[2]=V.at<double>(2, 0);
        double squareerror=0;
        for(int k=0;k<ptVec[i].size();++k){
            squareerror+=(ptVec[i][k].y-(singleres[0]*k*k+singleres[1]*k+singleres[2]))*(ptVec[i][k].y-(singleres[0]*k*k+singleres[1]*k+singleres[2]));
        }
        singleres[3]=squareerror;
        res.push_back(singleres);
        ofs << V.at<double>(0, 0) << " " << V.at<double>(1, 0) << " " << V.at<double>(2, 0) << " " << squareerror << std::endl;
    }
	ofs.close();
	//std::vector<cv::Point3d> rt;

	
}
