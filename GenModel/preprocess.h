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
    //读取feature,txt转化为mat
    int ReadTxt(std::string txtname, cv::Mat& txtdata);//read txt to Mat000
    void ReadXml(std::string xmlname, cv::Mat& xmldata);//read xml to Mat000
    void ReadName(std::string txtname, vector<cv::String>& txtdata);//000
    //把mat输出为txt文件
    int OutputMat(std::string filename,cv::Mat outputMat);//000
    //分类图片
    int ClassifyImages(std::vector<cv::String> image_files,cv::Mat labels,std::string filename);//000
    void SVMmodel(cv::Mat traindata,cv::Mat labels,cv::String path);//000
    void CopyFile(std::string sourcefile, std::string destfile);//000
    //计算数据偏差
    float getoptimaloffset(vector<float> datavec0, vector<float> datavec2, float pthreshold, float nthreshold);
    //read csv
    int ReadCSVfile(std::string filename, vector<vector<std::string>>& data);
    int GetColdata(vector<vector<std::string>> data, vector<float>& coldata,int dataindex, int colindex, std::string neasureID);
    int ReadTxttoVec(std::string txtname, vector<vector<float>>& data);
    int GetColstddata(vector<vector<float>> data, vector<float>& coldata,int dataindex);
    int ReadTxttoString(std::string txtname, vector<string>& data);
};

#endif // PREPROCESS_H
