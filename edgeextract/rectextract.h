#ifndef RECTEXTRACT_H
#define RECTEXTRACT_H

#include <opencv2/opencv.hpp>
#include <cmath>
#include <ctime>
#define PI 3.1415926

using namespace std;

enum ScanDirection {
    SCAN_DIRECTION_UP      = 1,              // from down to up
    SCAN_DIRECTION_LEFT    = 4,
    SCAN_DIRECTION_DOWN    = 3,
    SCAN_DIRECTION_RIGHT   = 2,
};

class RectExtract
{
public:
    RectExtract();
    //计算两点上下平行线上的点集
    int calcpoints(int x1,int y1,int x2,int y2,int d,vector<cv::Point>& uppoints,vector<cv::Point>& downpoints);
    //用中心点加旋转射线方法找边缘
    int FindEdge(cv::Mat srcimg, cv::Point seed, vector<cv::Point>& contours, int radius, float percent, int radthresh);
    //已知方向，通过从一边线扫求边缘
    int ScanLine(cv::Mat srcimg,int lineori, vector<cv::Point>& contours,int startline,int radius,float percent);//lineori: 1:top,2:right,3:bottom,4:left
    //与一个5*5的边缘提取算子卷积
    int ConvImage5(cv::Mat srcimg, int threshold, vector<cv::Point>& contours);
    //用kirsch边缘提取求直线边缘，用线扫方式
    int KirschEdgeLine(cv::Mat srcimg,int threshold,int orientation, vector<cv::Point>& contours);
    int KirschEdgeInnerLine(cv::Mat srcimg,int threshold,int orientation, int startline, vector<int> segments, vector<cv::Point>& contours);
    //用kirsch边缘提取，用旋转射线方式
    vector<vector<cv::Point>> GetFourEdges(vector<cv::Point> contours,int angle);
    int FitLineAndDraw(cv::Mat srcimg, cv::Mat& dstimg, vector<vector<cv::Point>> linecontours);
    int GetRectEdge(cv::Mat srcimg,int threshold,int num, vector<float>& dists);
    //用kirsch边缘提取求弧边缘，用旋转射线方式
    int KirschEdgeCircle(cv::Mat srcimg,int threshold, int startangle, int endangle, vector<cv::Point>& contours);
    //L D R cricle
    int LDRCircleEdge(cv::Mat srcimg, int threshold, int num, vector<cv::Point>& contours);
    //计算小圆边缘
    int KirschEdgeSmallCircle(cv::Mat srcimg,int threshold, int num, vector<cv::Point>& contours);
    int GetSmallCircle(cv::Mat srcimg,int threshold, int num, vector<float>& result);
    vector<float> FitCircle(vector<cv::Point>& contours);
    int PrewittEdge(cv::Mat srcimg,int threshold, vector<cv::Point>& contours);
    int SobelEdge(cv::Mat srcimg,int threshold, vector<cv::Point>& contours);
    //计算图中基准直线边缘与x轴的角度
    float CalibrationImgOri(cv::Mat srcimg, int orientation, int threshold, int radius=5, float percent=0.8);
    //旋转图片
    void RotateImg(cv::Mat srcimg, cv::Mat& dstimg, float angle);
    int GetDatum(cv::Mat srcimg,vector<cv::Point2f>& datum);
    float Point2Line(cv::Point Point, vector<cv::Point2f> datum, int flag);//0:与左边基准的距离;1:与上边基准的距离
    //获取黑色缝隙位置
    int GetDarkGap(cv::Mat srcimg, vector<cv::Point>& seedpoints);
    //外边缘检测SE1-9
    int KirschEdgeOuter(cv::Mat srcimg, int thresh, int num, float& dist);
    //找到边缘检测起始位置,单次边缘
    void GetSeedPoints(cv::Mat srcimg,int orientation,int startline,int endline,int threshold, vector<cv::Point>& seedpoints);
    //获取图像边缘点,根据像素值无阈值分割
    void GetSrcEdgePoints(cv::Mat srcimg,int orientation,int offset,int startline,int endline,int threshold, int threshold1, vector<cv::Point>& edgepoints);
    //获取图像边缘点
    void GetEdgePoint(cv::Mat srcimg, vector<cv::Point> seedpoints, int orientation, int thresh,int radius,float percent, vector<cv::Point>& contours);
    //找到边缘检测起始位置,两次边缘
    void GetSeedPointsSec(cv::Mat srcimg,int orientation,int startline,int endline,int threshold, vector<cv::Point>& seedpoints);
    int GetDouLines(cv::Mat srcimg, vector<cv::Point>& linepoints);

    int GetNewDatum(cv::Mat srcimg, vector<cv::Point2f>& datum);


    int GetNewDatum575(cv::Mat srcimg, vector<cv::Point2f>& datum);
    int ScanLineRange(cv::Mat srcimg,int lineori, vector<cv::Point>& contours,int startline,int startedge, int endedge, int radius,float percent);
    float PointLineDist(cv::Point2f A,vector<float> Lineparas);
    int NewGetRectEdge(cv::Mat srcimg, const vector<int> paras, vector<float>& dists, vector<vector<cv::Point2f>>& contourres);
    int NewKirschEdgeOuter(cv::Mat srcimg, const vector<int> paras, vector<cv::Point2f>& dist);
    int NewCircleEdge(cv::Mat srcimg, const vector<int> paras, vector<cv::Point>& contours);
    int NewSmallCircle(cv::Mat srcimg, const vector<int> paras,  vector<float>& result);
    int LDmeasure(cv::Mat srcimg, const vector<int> paras, vector<cv::Point2f>& respts);
    //新螺纹孔边缘提取，返回圆参数
    int SmallCircle(cv::Mat srcimg, const vector<int> paras, vector<float>& res);
    int FindDatumPoint(vector<cv::Point2f> xpts, vector<cv::Point2f> ypts, vector<cv::Point2f>& datumpts);
    void LineFitLeastSquaresf(vector<cv::Point2f> contours, vector<float> &vResult);

    float GetParallelism(vector<cv::Point2f> contour,int flag);
    float GetLineDist(vector<cv::Point2f> contour1,vector<cv::Point2f> contour2,int flag);
    float GetRoundness(vector<cv::Point> contour, cv::Point2f center);
    float GetPosition(cv::Point2f stdposition,cv::Point2f measurepos);


    int Get2Datum616(cv::Mat srcimg,vector<cv::Point2f>& datum);

    //check camera Z position
    int CameraCheck(cv::Mat srcimg, int CamId, float& verdist, float& hordist);

    int GetNewDatum616(cv::Mat srcimg, vector<cv::Point2f>& datum);

    //卷积
    void conv2D(cv::InputArray _src, cv::InputArray _kernel, cv::OutputArray _dst, int ddepth, cv::Point anchor = cv::Point(-1, -1), int borderType = cv::BORDER_DEFAULT);
    //krisch边缘检测
    cv::Mat krisch(cv::InputArray src,int borderType = cv::BORDER_DEFAULT);
    void sepConv2D_X_Y(cv::InputArray src, cv::OutputArray src_kerX_kerY, int ddepth, cv::InputArray kernelX, cv::InputArray kernelY, cv::Point anchor = cv::Point(-1, -1), int borderType = cv::BORDER_DEFAULT);
    void sepConv2D_Y_X(cv::InputArray src, cv::OutputArray src_kerY_kerX, int ddepth, cv::InputArray kernelY, cv::InputArray kernelX, cv::Point anchor = cv::Point(-1, -1), int borderType = cv::BORDER_DEFAULT);
    void prewitt(cv::InputArray src,cv::OutputArray dst, int ddepth,int x, int y = 0, int borderType = cv::BORDER_DEFAULT);
    int factorial(int n);
    cv::Mat getPascalSmooth(int n);
    cv::Mat getPascalDiff(int n);
    cv::Mat sobel(cv::Mat image, int x_flag, int y_flag, int winSize, int borderType);
    cv::Mat getHistograph(const cv::Mat grayImage);
    //判断点周围密度
    int PointDense(cv::Mat srcimg, cv::Point center, int rad, float thresh);
    int ContourDense(cv::Mat srcimg,vector<cv::Point> contours,int startangle,int endangle, vector<cv::Point>& densecontours, int thickness);//把contours的边缘拓宽，保存thickness厚度内的边界点
    vector<cv::Point> ContoursCut(vector<cv::Point> contours,int startangle,int endangle);
    //拟合圆
    void GetCircle(vector<cv::Point> contours, cv::Point center, int threshold,cv::Vec3f& circlepara);
    //计算圆上中点坐标 0：上面 1：下面
    cv::Point GetTanPoint(int a, int b, int R, float k, int flag);
    //计算差距
    double CalcDiff(vector<cv::Point> contours,cv::Point center);
    //拟合直线
    int LineFit(vector<cv::Point> contours,cv::Vec2f& linepara);
    void LineFitLeastSquares(vector<cv::Point> contours, vector<float> &vResult);
    //截取片段
    vector<cv::Point> GetSegments(vector<cv::Point> coontours, vector<int> segments);
    //拟合直线并计算距离
    void FitLineAndGetDist(vector<vector<cv::Point>> linecontours,vector<int> coordinates,vector<float>& dist);
    //Rect num=3 图像右边缘检测
    void ScanRightLine(cv::Mat srcimg, vector<cv::Point>& contours);
    //像素密度判定
    bool PixelDense(cv::Mat srcimg, int thresh);
    //Kirsch边缘检测
    int KirschEdge(cv::Mat srcimg,int threshold, vector<cv::Point>& contours, vector<vector<cv::Point>>& linecontours, vector<int> segment);
    //求一阶导数
    void FirstDerivative(vector<float> data, vector<float>& firstdev);
    //求二阶导数
    void SecondDerivative(vector<float> data, vector<float>& seconddev);
    //以2*2矩阵求均值生成像素变化曲线
    int Pixel2Curve(cv::Mat srcimg,int startpos, int endpos, int orientation, vector<vector<float>>& curves);
    //找到边缘检测起始位置,单次边缘
    void GetSeedPos(vector<vector<float>> curves,int thresh, vector<int>& seedpoints);
    //找到边缘检测起始位置,两次边缘
    void GetSeedPosSecond(vector<vector<float>> curves,int thresh, vector<int>& seedpoints);
    //中分线Y轴坐标
    float GetMidLineY(vector<vector<cv::Point>> linecontours, int pos);
    //线X坐标fai52
    float GetLineX(vector<cv::Point> linecontour, int pos);

    template<typename T1,typename T2>
    float Point2Point(T1 A,T2 B);

    int vectortotxt(vector<vector<float>> curves);

    int FitCircleLeast(vector<cv::Point> contour, vector<float> cir);
    int getdatum001(vector<cv::Point2f> xpts, vector<cv::Point2f> ypts, vector<cv::Point2f>& datumpts);
};

#endif // RECTEXTRACT_H
