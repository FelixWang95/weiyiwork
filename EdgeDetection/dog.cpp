#include "edgedetecction.h"

//高斯卷积
Mat gaussConv(Mat I,float sigma,int s)
{
    //创建水平方向上的非归一化的高斯核
    Mat xkernel = Mat::zeros(1, s, CV_32FC1);
    //中心位置
    int cs = (s - 1) / 2;
    //方差
    float sigma2 = pow(sigma, 2.0);
    for (int c = 0; c < s; c++)
    {
        float norm2 = pow(float(c - cs), 2.0);
        xkernel.at<float>(0, c) = exp(-norm2 / (2 * sigma2));
    }
    //将 xkernel 转置，得到垂直方向上的卷积核
    Mat ykernel = xkernel.t();
    //分离卷积核的卷积运算
    Mat gauConv;
    sepConv2D_X_Y(I, gauConv, CV_32F, xkernel, ykernel);
    gauConv.convertTo(gauConv, CV_32F, 1.0 / sigma2);
    return gauConv;
}
//高斯差分
Mat DoG(Mat I, float sigma, int s, float k)
{
    //与标准差为 sigma 的非归一化的高斯核卷积
    Mat Ig = gaussConv(I, sigma, s);
    //与标准差为 k*sigma 的非归一化的高斯核卷积
    Mat Igk = gaussConv(I, k*sigma, s);
    //两个高斯卷积结果做差
    Mat doG = Igk - Ig;
    return doG;
}
