#ifndef FINDLINES_H
#define FINDLINES_H
#include <opencv2/opencv.hpp>
#include <cmath>
#define PI 3.1415926

using namespace std;

class FindLines
{
public:
    FindLines();
    int GetDouLines(cv::Mat srcimg, vector<cv::Point>& linepoints, int num);
    int GetLines(cv::Mat srcimg, vector<cv::Point>& linepoints, int num);
    void LineFitLeastSquares(vector<cv::Point> contours, vector<float> &vResult);

private:
    int LineFit(vector<cv::Point> contours,cv::Vec2f& linepara);
    int PointDense(cv::Mat srcimg, cv::Point center, int rad, float thresh);
    //void LineFitLeastSquares(vector<cv::Point> contours, vector<float> &vResult);
};

#endif // FINDLINES_H
