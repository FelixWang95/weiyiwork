#ifndef IMAGESCREEN_H
#define IMAGESCREEN_H

#include <opencv2/opencv.hpp>

#define PI 3.1415926

using namespace std;

enum ScanDirection {
    SCAN_DIRECTION_UP      = 1,              // from down to up
    SCAN_DIRECTION_LEFT    = 4,
    SCAN_DIRECTION_DOWN    = 3,
    SCAN_DIRECTION_RIGHT   = 2,
};


class ImageScreen
{
public:
    ImageScreen();
    //选出路径文件夹中所有图片的每一小块矩形范围图像
    int ImageCut(cv::String path, vector<vector<cv::Mat>>& partimgs);
    int ImageCuttest(cv::String path, vector<vector<cv::Mat>>& partimgs);
    //每个部分裁剪成六块小矩形
    vector<cv::Rect> RectCut(cv::Rect section);
    //通过数据生成pca模型
    int PCAgeneration(vector<vector<cv::Mat>> partimgs,vector<cv::Mat>& outputMat);
    //生成灰度直方图特征
    cv::Mat Histgeneration(cv::Mat srcimg);
    //从xml文件中读取pca模型
    cv::PCA ReadPCAFromXML(cv::String path);
    //读取SVM模型
    cv::Ptr<cv::ml::SVM> ReadSVMFromXML(cv::String path);
    //使用pca降维
    cv::Mat PCAdecrease(cv::PCA pca, cv::Mat inputMat);
    //单张图片获取亮度特征
    int SectionBrightness(vector<cv::Mat> partimgs, vector<cv::Mat>& PCAfeatures);
    //把mat写入csv
    void WriteCSV(string filename, cv::Mat m);
    //把mat输出为txt文件
    int OutputMat(std::string filename,cv::Mat outputMat);
    //把图片转化为灰度直方图特征,分类输出结果
    int FeatureGeneration(cv::Mat srcimg, vector<float>& labels);
    //测试用
    int FeatureGenerationtest(cv::Mat srcimg);
    //训练图像生成直方图模型
    int PathToModelHist(cv::String path);

    //选出文件夹中所有图片中边缘变化曲线
    int EdgeSelect(cv::String path, vector<vector<vector<float>>>& partcurves);
    int EdgeSelectTest(cv::String path, vector<vector<vector<float>>>& partcurves);
    //根据不同的边缘区域获取区域的几条变化曲线
    vector<vector<float>> GetCurve(cv::Mat srcimg, int num);
    //根据起始点和方向生成一条灰度变化曲线
    vector<vector<float>> SectionCurve(cv::Mat srcimg, vector<int> startpos, int orientation);
    //拟合灰度变化曲线
    vector<float> FitCurves(vector<float> curves, int times);
    //把图片转化为灰度曲线保存在csv中
    int EdgeCurves(cv::Mat srcimg);
    //读取feature,txt转化为mat
    int ReadTxt(std::string txtname, cv::Mat& txtdata);//read txt to Mat
    //生成SVM模型
    cv::Ptr<cv::ml::SVM> OneClassSVMmodel(cv::Mat traindata,float gamma, float nu, std::string path);
    //读取matlab生成的特征进行oneclasssvm
    int ReadandGen(std::string txtname);
    //vector data to svm model
    int GenSVM(vector<vector<float>> feadata, int index);
    //把图片转化为曲线特征,分类输出结果
    int CurveFeatureGeneration(cv::Mat srcimg, vector<float>& labels);
    //测试用
    int CurveFeatureGenerationtest(cv::Mat srcimg);
    //读取matlab生成的特征进行分类
    int ClassifyFeature(std::string txtname);
    //训练图像生成曲线模型
    int PathToModelEdge(cv::String path);

    //黑白转换坐标特征
    //以2*2矩阵求均值生成像素变化曲线
    int Pixel2Curve(cv::Mat srcimg, vector<int> testpos, int orientation, vector<vector<float>>& curves);
    //获取黑白转换位置
    int GetTransPos(vector<vector<float>> curves, vector<vector<int>>& pos, int thresh);
    //黑白转换位置转换为每个边缘部分的特征

    //判断分数
    bool ScoreEvaluation(vector<float> histlabels, vector<float> edgelabels);
};

#endif // IMAGESCREEN_H
