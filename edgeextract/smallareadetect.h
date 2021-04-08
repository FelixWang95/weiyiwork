#ifndef SMALLAREADETECT_H
#define SMALLAREADETECT_H

#include<opencv2/opencv.hpp>

using namespace std;

class SmallAreaDetect
{
public:
    SmallAreaDetect();
    int KernelSelect(cv::Mat srcimg, int unitlength, int deltathresh, vector<cv::Point>& position);
    int KernelSelectglcm(cv::Mat srcimg, int unitlength, float deltathresh, vector<cv::Point>& position);

private:
    bool KernelJudgement(cv::Mat kernelimg, int deltathresh);
    bool KernelJudgementglcm(cv::Mat kernelimg, float deltathresh);
    float DistCalc(vector<float> vec1, vector<float> vec2);
    float Cosdist(vector<float> vec1, vector<float> vec2);
};

#endif // SMALLAREADETECT_H
