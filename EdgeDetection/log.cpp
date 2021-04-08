#include "edgedetecction.h"
//得到一维的 （x^2 /(2*sigma^2)-1）Gauss(x,sigma) 和 一维的 Gauss(x,sigma)
void getSepLoGKernel(float sigma,int length,Mat & kernelX,Mat & kernelY)
{
    //分配内存
    kernelX.create(Size(length, 1), CV_32FC1);
    kernelY.create(Size(1, length), CV_32FC1);
    int center = (length - 1) / 2;
    double sigma2 = pow(sigma, 2.0);
    double cofficient = 1.0 / (sqrt(2 * CV_PI)*sigma);
    for (int c = 0; c < length; c++)
    {
        float norm2 = pow(c - center, 2.0);
        kernelY.at<float>(c,0) = cofficient*exp(-norm2 / (2 * sigma2));
        kernelX.at<float>(0, c) = (norm2 / sigma2 - 1.0)*kernelY.at<float>(c, 0);
    }
}

// LoG 卷积
Mat LoG(InputArray image,float sigma,int win)
{
    Mat kernelX, kernelY;
    //得到两个分离核
    getSepLoGKernel(sigma, win, kernelX, kernelY);
    //先水平卷积再垂直卷积
    Mat covXY;
    sepConv2D_X_Y(image, covXY, CV_32FC1, kernelX, kernelY);
    //卷积核转置
    Mat kernelX_T = kernelX.t();
    Mat kernelY_T = kernelY.t();
    //先垂直卷积再水平卷积
    Mat covYX;
    sepConv2D_Y_X(image,covYX,CV_32FC1,kernelX_T,kernelY_T);
    //计算两个卷积结果的和，得到 LoG 卷积
    Mat LoGCov;
    add(covXY, covYX, LoGCov);
    return LoGCov;
}
