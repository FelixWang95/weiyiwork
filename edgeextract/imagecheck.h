#ifndef IMAGECHECK_H
#define IMAGECHECK_H

#include <opencv2/opencv.hpp>

#define PI 3.1415926

using namespace std;

class ImageCheck
{
public:
    ImageCheck();
    bool IsBlackAround(cv::Mat srcimg);
    bool IsBlackAreaSatisfied(cv::Mat srcimg, int thresh, float lower, float upper);
    float GetBlackArea(cv::Mat srcimg, int thresh);

    //获取图像外轮廓
    int GetContour(cv::Mat srcimg, vector<cv::Point>& contour);
    //轮廓转角度，step为两连线起始点的间隔，length为连线的长度，角度为连线的夹角
    int Contour2Angle(vector<cv::Point> contour,int step, int length, vector<float>& angles);
    //根据四点求两向量夹角
    float GetAngle(cv::Point A, cv::Point B, cv::Point C, cv::Point D);
    //计算灰度直方图
    int CalcHistogram(cv::Mat srcimg,int thresh,int bin,cv::Mat& hist);
    //PCA降维
    cv::PCA PCADecreaseDim(cv::Mat iuputMat,cv::Mat& outputMat,float percentage);
    //vector<float>转Mat 32FC1
    void Vecf2Mat32F(vector<float> vec, cv::Mat& outputMat);
    //把mat输出为txt文件
    int OutputMat(std::string filename,cv::Mat outputMat);
    //生成SVM模型
    cv::Ptr<cv::ml::SVM> OneClassSVMmodel(cv::Mat traindata,float gamma, float nu, cv::String path);
    //测试SVM模型
    void TestSVMmodel(cv::Ptr<cv::ml::SVM> svm,cv::Mat testdata,cv::Mat& resultlabel);
    //生成错误图片
    cv::Mat GenerateWrongImage(cv::Mat srcimg, float hor);
    //读取SVM模型
    cv::Ptr<cv::ml::SVM> ReadSVMFromXML(cv::String path);
    //读取PCA模型
    cv::PCA ReadPCAFromXML(cv::String path);
    //PCA降维
    cv::Mat PCAdecrease(cv::PCA pca, cv::Mat inputMat);
    //边缘点排序
    vector<cv::Point> SortContour(vector<cv::Point> contour);
    vector<cv::Point> SortContourCenter(vector<cv::Point> contour, cv::Point center);
    //找灰度直方图最优svm参数
    int FindHistSVM(cv::Mat histMat, float& fp, float& fn);
    //找边缘最优svm参数
    int FindEdgeSVM(cv::Mat histMat, float& fp, float& fn);
    //直方图特征提取
    cv::Mat HistExtract(cv::String path, int resizenum);
    //边缘特征提取
    cv::Mat EdgeExtract(cv::String path);
    //横向n*n区域的平均像素值
    int KernelCurve(cv::Mat srcimg, int kernelsize, int column, vector<float>& curve);

    void SaveXml(std::string xmlname, cv::Mat xmldata);
    void ReadXml(std::string xmlname, cv::Mat& xmldata);//read xml to Mat
};

#endif // IMAGECHECK_H
