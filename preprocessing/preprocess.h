#ifndef PREPROCESS_H
#define PREPROCESS_H
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#define PI 3.1415926
using namespace std;

class preprocess
{
public:
    preprocess();
    //读取文件夹下所有图片路径
    int ReadImagePath(std::string filename, std::vector<cv::String>& image_files);
    //计算灰度直方图
    int CalcHistogram(cv::Mat srcimg,int thresh,int bin,cv::Mat& hist);
    //读取feature,txt转化为mat
    int ReadTxt(std::string txtname, cv::Mat& txtdata);//read txt to Mat
    void ReadXml(std::string xmlname, cv::Mat& xmldata);//read xml to Mat
    void ReadName(std::string txtname, vector<cv::String>& txtdata);
    //把mat输出为txt文件
    int OutputMat(std::string filename,cv::Mat outputMat);
    //分类图片
    int ClassifyImages(std::vector<cv::String> image_files,cv::Mat labels,std::string filename);
    //计算图中基准直线边缘与x轴的角度
    float CalibrationImgOri(cv::Mat srcimg, int orientation, int threshold, int radius=5, float percent=0.8);
    //旋转图片
    void RotateImg(cv::Mat srcimg, cv::Mat& dstimg, float angle);
    int PCADecreaseDim(cv::Mat iuputMat,cv::Mat& outputMat,float percentage);
    void SVMmodel(cv::Mat traindata,cv::Mat labels,cv::String path);
    void SVMTest();
    int GetDatum(cv::Mat srcimg,vector<cv::Point2f>& datum);
    float Point2Line(cv::Point Point, int flag);//0:y 1:x
    cv::Point top,left,cross;
    void ExpandImage(cv::Mat srcimg, cv::Mat& dstimg);
    //分类图片11,12,13
    void ClassifyImages2(std::string filename);
    int JPGtoPNG(std::string filename);
    int ClearDust(std::string filename);

private:
    //已知方向，通过从一边线扫求边缘
    int ScanRect(cv::Mat srcimg,int lineori, vector<cv::Point>& contours,int radius,float percent);//lineori: 1:top,2:right,3:bottom,4:left
    int PointDense(cv::Mat srcimg, cv::Point center, int rad, float thresh);
    int LineFit(vector<cv::Point> contours,cv::Vec2f& linepara);
    void conv2D(cv::InputArray _src, cv::InputArray _kernel, cv::OutputArray _dst, int ddepth, cv::Point anchor = cv::Point(-1, -1), int borderType = cv::BORDER_DEFAULT);
    cv::Mat krisch(cv::InputArray src,int borderType = cv::BORDER_DEFAULT);
    void CopyFile(std::string sourcefile, std::string destfile);
};

#endif // PREPROCESS_H
