#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include <opencv2/opencv.hpp>

class interpolation
{
public:
    interpolation();
    int imageRotation(cv::Mat srcimg, cv::Mat& dstimg, float rad);
    int imageRotationbl(cv::Mat srcimg, cv::Mat& dstimg, float rad);
    int imageRotationbcl(cv::Mat srcimg, cv::Mat& dstimg, float rad);
    cv::Point nearestInterpolation(cv::Point2f srcpt);
    int bilinearInterpolation(cv::Point2f srcpt, cv::Mat srcimg, cv::Point rotateCenter);
    int bicubicInterpolation(cv::Point2f srcpt, cv::Mat srcimg, cv::Point rotateCenter);
    float bicubicWeight(double dist,float a);
    template<class T1,class T2>
    double calcEdistance(T1 x1,T1 y1,T2 x2,T2 y2);
};

#endif // INTERPOLATION_H
