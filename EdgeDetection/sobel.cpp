#include "edgedetecction.h"
//计算阶乘
int factorial(int n)
{
    int fac = 1;
    // 0 的阶乘等于 1
    if (n == 0)
        return fac;
    for (int i = 1; i <= n; i++)
        fac *= i;
    return fac;
}
//计算平滑系数
Mat getPascalSmooth(int n)
{
    Mat pascalSmooth = Mat::zeros(Size(n, 1), CV_32FC1);
    for (int i = 0; i < n; i++)
        pascalSmooth.at<float>(0, i) = factorial(n - 1) / (factorial(i) * factorial(n - 1 - i));
    return pascalSmooth;
}
//计算差分
Mat getPascalDiff(int n)
{
    Mat pascalDiff = Mat::zeros(Size(n, 1), CV_32FC1);
    Mat pascalSmooth_previous = getPascalSmooth(n - 1);
    for (int i = 0; i<n; i++)
    {
        if (i == 0)
            pascalDiff.at<float>(0, i) = 1;
        else if (i == n - 1)
            pascalDiff.at<float>(0, i) = -1;
        else
            pascalDiff.at<float>(0, i) = pascalSmooth_previous.at<float>(0, i) - pascalSmooth_previous.at<float>(0, i - 1);
    }
    return pascalDiff;
}

// sobel 边缘检测
Mat sobel(Mat image, int x_flag, int y_flag, int winSize, int borderType)
{
    // sobel 卷积核的窗口大小为大于 3 的奇数
    CV_Assert(winSize >= 3 && winSize % 2 == 1);
    //平滑系数
    Mat pascalSmooth = getPascalSmooth(winSize);
    //差分系数
    Mat pascalDiff = getPascalDiff(winSize);
    Mat image_con_sobel;
    /* 当 x_falg != 0 时，返回图像与水平方向的 Sobel 核的卷积*/
    if (x_flag != 0)
    {
        //根据可分离卷积核的性质
        //先进行一维垂直方向上的平滑，再进行一维水平方向的差分
        sepConv2D_Y_X(image, image_con_sobel, CV_32FC1, pascalSmooth.t(), pascalDiff, Point(-1, -1), borderType);
    }
    /* 当 x_falg == 0 且 y_flag != 0 时，返回图像与垂直 方向的 Sobel 核的卷积*/
    if (x_flag == 0 && y_flag != 0)
    {
        //根据可分离卷积核的性质
        //先进行一维水平方向上的平滑，再进行一维垂直方向的差分
        sepConv2D_X_Y(image, image_con_sobel, CV_32FC1, pascalSmooth, pascalDiff.t(), Point(-1, -1), borderType);
    }
    return image_con_sobel;
}
