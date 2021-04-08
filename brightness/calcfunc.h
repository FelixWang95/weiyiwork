#ifndef CALCFUNC_H
#define CALCFUNC_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;

class calcfunc
{
public:
    calcfunc();
    //输入点集，原图，输出点集的Lab中的L
    float BrightnessLevel(vector<cv::Point> area,cv::Mat srcimg);
    //遍历文件夹中txt的行数
    int ReadPointNum(std::string filename,vector<int>& pointnum);
    int DilateThresh(cv::Mat srcimg, vector<cv::Point> inputarea, vector<cv::Point>& outputarea);

private:
    //传统计算Lab方法
    const float param_13 = 1.0f / 3.0f;
    const float param_16116 = 16.0f / 116.0f;
    const float Xn = 0.950456f;
    const float Yn = 1.0f;
    const float Zn = 1.088754f;
    float gamma(float x);
    void RGB2XYZ(float R,float G,float B, float& X, float& Y, float& Z);
    void XYZ2Lab(float X,float Y,float Z, float& L);
    int OpenCVLab(int R,int G,int B);//Opencv计算Lab方法
};

#endif // CALCFUNC_H
