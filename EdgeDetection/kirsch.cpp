#include "edgedetecction.h"

/* Krisch 边缘检测算法*/
Mat krisch(InputArray src,int borderType)
{
    //存储八个卷积结果
    vector<Mat> eightEdge;
    eightEdge.clear();
    /*第1步：图像矩阵与8 个 卷积核卷积*/
    /*Krisch 的 8 个卷积核均不是可分离的*/
    //图像矩阵与 k1 卷积
    Mat k1 = (Mat_<float>(3, 3) << 5, 5, 5, -3, 0, -3, -3, -3, -3);
    Mat src_k1;
    conv2D(src, k1, src_k1, CV_32FC1);
    convertScaleAbs(src_k1, src_k1);
    eightEdge.push_back(src_k1);
    //图像矩阵与 k2 卷积
    Mat k2 = (Mat_<float>(3, 3) << -3, -3, -3, -3, 0, -3, 5, 5, 5);
    Mat src_k2;
    conv2D(src, k2, src_k2, CV_32FC1);
    convertScaleAbs(src_k2, src_k2);
    eightEdge.push_back(src_k2);
    //图像矩阵与 k3 卷积
    Mat k3 = (Mat_<float>(3, 3) << -3, 5, 5, -3, 0, 5, -3, -3, -3);
    Mat src_k3;
    conv2D(src, k3, src_k3, CV_32FC1);
    convertScaleAbs(src_k3, src_k3);
    eightEdge.push_back(src_k3);
    //图像矩阵与 k4 卷积
    Mat k4 = (Mat_<float>(3, 3) << -3, -3, -3, 5, 0, -3, 5, 5, -3);
    Mat src_k4;
    conv2D(src, k4, src_k4, CV_32FC1);
    convertScaleAbs(src_k4, src_k4);
    eightEdge.push_back(src_k4);
    //图像矩阵与 k5 卷积
    Mat k5 = (Mat_<float>(3, 3) << -3, -3, 5, -3, 0, 5, -3, -3, 5);
    Mat src_k5;
    conv2D(src, k5, src_k5, CV_32FC1);
    convertScaleAbs(src_k5, src_k5);
    eightEdge.push_back(src_k5);
    //图像矩阵与 k6 卷积
    Mat k6 = (Mat_<float>(3, 3) << 5, -3, -3, 5, 0, -3, 5, -3, -3);
    Mat src_k6;
    conv2D(src, k6, src_k6, CV_32FC1);
    convertScaleAbs(src_k6, src_k6);
    eightEdge.push_back(src_k6);
    //图像矩阵与 k7 卷积
    Mat k7 = (Mat_<float>(3, 3) << -3, -3, -3, -3, 0, 5, -3, 5, 5);
    Mat src_k7;
    conv2D(src, k7, src_k7, CV_32FC1);
    convertScaleAbs(src_k7, src_k7);
    eightEdge.push_back(src_k7);
    //图像矩阵与 k8 卷积
    Mat k8 = (Mat_<float>(3, 3) << 5, 5, -3, 5, 0, -3, -3, -3, -3);
    Mat src_k8;
    conv2D(src, k8, src_k8, CV_32FC1);
    convertScaleAbs(src_k8, src_k8);
    eightEdge.push_back(src_k8);
    /*第二步：将求得的八个卷积结果,取对应位置的最大值，作为最后的边缘输出*/
    Mat krischEdge = eightEdge[0].clone();
    for (int i = 0; i < 8; i++)
    {
        max(krischEdge, eightEdge[i], krischEdge);
    }
    return krischEdge;
}
