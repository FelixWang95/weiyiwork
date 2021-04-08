#ifndef EDGEDETECT_H
#define EDGEDETECT_H

#include <opencv2/opencv.hpp>
#include <cmath>
#include <ctime>
#define PI 3.1415926

using namespace std;



//class edgedetect
//{
//public:
//    edgedetect();
//    //求基准,575
//    int GetNewDatum575(cv::Mat srcimg, vector<cv::Point2f>& datum);
//    int GetDatum616(cv::Mat srcimg, const vector<int> paras,vector<cv::Point2f>& datum);
//    int GetSideDatum616(vector<cv::Point> contour1,vector<cv::Point> contour2,vector<cv::Point2f>& datum);
//    int GetDatum452(vector<cv::Point> contour1,vector<cv::Point> contour2,vector<cv::Point2f>& datum);
//    int FindDatumPoint(vector<cv::Point2f> xpts, vector<cv::Point2f> ypts, vector<cv::Point2f>& datumpts);
//    //矩形边缘提取
//    int NewGetRectEdge(cv::Mat srcimg, const vector<int> paras, vector<float>& dists, vector<vector<cv::Point2f>>& contourres);
//    int GetLineContours(cv::Mat srcimg, const vector<int> paras, vector<cv::Point>& contours);
//    //外边缘提取,返回点位
//    int NewKirschEdgeOuter(cv::Mat srcimg, const vector<int> paras, vector<cv::Point2f>& dist);
//    //背光图小半圆边缘提取
//    int NewCircleEdge(cv::Mat srcimg, const vector<int> paras, vector<cv::Point>& contours);
//    //螺纹孔边缘提取，返回圆参数
//    int NewSmallCircle(cv::Mat srcimg, const vector<int> paras,  vector<float>& result);
//    //LD部分边缘提取，返回点位
//    int LDmeasure(cv::Mat srcimg, const vector<int> paras, vector<cv::Point2f>& respts);
//    //新螺纹孔边缘提取，返回圆参数
//    int SmallCircle(cv::Mat srcimg, const vector<int> paras, vector<float>& res, int flag);

//    float GetParallelism(vector<cv::Point2f> contour,int flag);
//    float GetLineDist(vector<cv::Point2f> contour1,vector<cv::Point2f> contour2,int flag);
//    float GetRoundness(vector<cv::Point> contour, cv::Point2f center);
//    float GetPosition(cv::Point2f stdposition,cv::Point2f measurepos);
//    float GetLine2LineMid(vector<cv::Point2f> contour1,vector<cv::Point2f> contour2,int flag);
//    vector<float> FitCircle(vector<cv::Point>& contours);

//private:
//    int ScanLineRange(cv::Mat srcimg,int lineori, vector<cv::Point>& contours,int startline,int startedge, int endedge, int radius,float percent);
//    float PointLineDist(cv::Point2f A,vector<float> Lineparas);
//    void LineFitLeastSquaresf(vector<cv::Point2f> contours, vector<float> &vResult);
//    void LineFitLeastSquares(vector<cv::Point> contours, vector<float> &vResult);
//    //用中心点加旋转射线方法找边缘
//    int FindEdge(cv::Mat srcimg, cv::Point seed, vector<cv::Point>& contours, int radius, float percent, int radthresh);
//    //已知方向，通过从一边线扫求边缘
//    int ScanLine(cv::Mat srcimg,int lineori, vector<cv::Point>& contours,int startline,int radius,float percent);
//    //判断点周围密度
//    int PointDense(cv::Mat srcimg, cv::Point center, int rad, float thresh);
//    //krisch边缘检测
//    cv::Mat krisch(cv::InputArray src,int borderType = cv::BORDER_DEFAULT);
//    //卷积
//    void conv2D(cv::InputArray _src, cv::InputArray _kernel, cv::OutputArray _dst, int ddepth, cv::Point anchor = cv::Point(-1, -1), int borderType = cv::BORDER_DEFAULT);
//    //找到边缘检测起始位置,单次边缘
//    void GetSeedPoints(cv::Mat srcimg,int offset, int orientation,int startline,int endline,int threshold, vector<cv::Point>& seedpoints);
//    //获取图像边缘点,根据像素值无阈值分割
//    void GetSrcEdgePoints(cv::Mat srcimg,int offset0,int orientation,int offset,int startline,int endline,int threshold, int threshold1, vector<cv::Point>& edgepoints);
//    //获取图像边缘点
//    void GetEdgePoint(cv::Mat srcimg, vector<cv::Point> seedpoints, int orientation, int thresh,int radius,float percent, vector<cv::Point>& contours);
//    //找到边缘检测起始位置,两次边缘
//    void GetSeedPointsSec(cv::Mat srcimg,int offset,int orientation,int startline,int endline,int threshold, vector<cv::Point>& seedpoints);
//    //以2*2矩阵求均值生成像素变化曲线
//    int Pixel2Curve(cv::Mat srcimg,int startpos, int endpos, int orientation, vector<vector<float>>& curves);
//    //找到边缘检测起始位置,单次边缘
//    void GetSeedPos(vector<vector<float>> curves,int offset,int thresh, vector<int>& seedpoints);
//    //找到边缘检测起始位置,两次边缘
//    void GetSeedPosSecond(vector<vector<float>> curves,int offset,int thresh, vector<int>& seedpoints);

//    vector<cv::Point> ContoursCut(vector<cv::Point> contours,int startangle,int endangle);


//    template<typename T1,typename T2>
//    float Point2Point(T1 A,T2 B);
//};

//#endif // EDGEDETECT_H

class edgedetect
{
public:
    edgedetect();
    //求575基准，575基准区域写死在函数中，输入背光图srcimg，输出3个点datum（左上角点，上部点，左部点），根据这三个点可构造坐标系。
int GetNewDatum575(cv::Mat srcimg, vector<cv::Point2f>& datum);

/*求616,618基准，输入背光基准模板图srcimg，基准所需参数paras，输出3个点datum（左上角点，上部点，左部点），根据这三个点可构造坐标系。以618为例，paras中参数含义如下：
Paras[0]:模板序号；
Paras[1]:二值化方式，1：正，2：反
Paras[2]:二值化阈值
Paras[3]:边缘检测筛选区域圆半径
Paras[4]:边缘检测筛选区域圆比例，圆区域中白色像素大于该比例为边缘点
以下参数每4个表示一条边缘，共3个基准边缘：
Paras[5]:线扫方向
Paras[6]:线扫起始位置，若线扫方向为上下，起始位置为图像上的y坐标；左右为x坐标
Paras[7]:边缘起始位置，若线扫方向为上下，边缘的起始和结束位置为x坐标上的一段；若线扫方向为左右，边缘的起始和结束位置为y坐标上的一段
Paras[8]:边缘结束位置*/
int GetDatum616(cv::Mat srcimg, const vector<int> paras,vector<cv::Point2f>& datum);

//求616，618侧面图基准，输入contour1为垂直边缘点集，输出contour2为水平边缘点集，输出3个点datum（左上角点，上部点，左部点），根据这三个点可构造坐标系。
int GetSideDatum616(vector<cv::Point> contour1,vector<cv::Point> contour2,vector<cv::Point2f>& datum);

//求452背光基准，输入contour1为垂直边缘点集，输出contour2为水平边缘点集，输出3个点datum（左上角点，上部点，左部点），根据这三个点可构造坐标系。
int GetDatum452(vector<cv::Point> contour1,vector<cv::Point> contour2,vector<cv::Point2f>& datum);

//求最凸点基准，输入xpts为水平点集，输入ypts为垂直点集，输出3个点datum（左上角点，上部点，左部点），根据这三个点可构造坐标系。
int FindDatumPoint(vector<cv::Point2f> xpts, vector<cv::Point2f> ypts, vector<cv::Point2f>& datumpts);

    //矩形边缘提取，该函数是用于应对特殊部分的边缘检测，一般不使用。如果有无通用性的特殊部分，需要单独写在这个函数中。
int NewGetRectEdge(cv::Mat srcimg, const vector<int> paras, vector<float>& dists, vector<vector<cv::Point2f>>& contourres);

/*直线线扫边缘提取，输入为模板图像srcimg，线扫参数paras，输出边缘点集，该函数用于对一个方向上边缘的查找，如果同一个模板需要多条边缘，可重复使用该函数。结果可用于拟合直线，求中线，求平行度等
Paras中的参数释义如下：
Paras[0]:模板序号；
Paras[1]:二值化阈值；
Paras[2]:二值化方式，0：kirsch边缘检测求梯度后二值化；1：反二值化；2：二值化；
Paras[3]:边缘筛选区域半径；
Paras[4]:边缘筛选区域阈值，圆区域中白色像素大于该比例为边缘点；
Paras[5]:线扫方向
Paras[6]:线扫起始位置，若线扫方向为上下，起始位置为图像上的y坐标；左右为x坐标
Paras[7]:边缘起始位置，若线扫方向为上下，边缘的起始和结束位置为x坐标上的一段；若线扫方向为左右，边缘的起始和结束位置为y坐标上的一段
Paras[8]:边缘结束位置*/
int GetLineContours(cv::Mat srcimg, const vector<int> paras, vector<cv::Point>& contours);

/*外边缘提取,返回点位，输入模板图像，输入边缘检测参数paras，输出点位dist，几个点就有几个元素。该函数主要用于一条边取几个点位的情况。Paras参数的释义如下：
以下参数适用于paras[2]=0,1
Paras[0]:模板序号；
Paras[1]:二值化阈值；
Paras[2]:起始点寻找方式，0：第一次遇到高像素值停止，以高像素值结束（黑色缝隙的起始位置）到停止处（黑色缝隙的结束位置）的中点为线扫起始点，用kirsch梯度寻找边缘；1：第二次遇到高像素值停止，以最近一次高像素值结束（黑色缝隙的起始位置）到第二次高像素停止处（黑色缝隙的结束位置）的中点为线扫起始点，用kirsch梯度寻找边缘；2：第一次遇到高像素值停止，以高像素值结束（黑色缝隙的起始位置）到停止处（黑色缝隙的结束位置）的中点为线扫起始点，并直接用灰度值寻找边缘。
Paras[3]:寻找起始点灰度阈值，大于该值的被定为高像素值；
Paras[4]:起始点寻找方向；
Paras[5]:边缘线扫方向；
Paras[6]:边缘筛选区域的半径；
Paras[7]:边缘筛选区域的比例，圆区域中白色像素大于该比例为边缘点；
Paras[8]:起始点寻找起始位置，从哪一条边开始就用起始点到这条边的垂直距离；
Paras[9]:所需点位数量，表示后面有多少段，每一段对应一对起始点和结束点
Paras[10]:第一段的起始位置；
Paras[11]:第一段的结束位置；
Paras[12]:第二段的起始位置：
Paras[13]:第二段的结束位置；
......
当paras[2]=2时，
Paras[0]:模板序号；
Paras[1]:寻找起始点灰度阈值，大于该值的被定为高像素值；
Paras[2]:起始点寻找方式，0：第一次遇到高像素值停止，以高像素值结束（黑色缝隙的起始位置）到停止处（黑色缝隙的结束位置）的中点为线扫起始点，用kirsch梯度寻找边缘；1：第二次遇到高像素值停止，以最近一次高像素值结束（黑色缝隙的起始位置）到第二次高像素停止处（黑色缝隙的结束位置）的中点为线扫起始点，用kirsch梯度寻找边缘；2：第一次遇到高像素值停止，以高像素值结束（黑色缝隙的起始位置）到停止处（黑色缝隙的结束位置）的中点为线扫起始点，并直接用灰度值寻找边缘。
Paras[3]:边缘检测灰度阈值，大于该值作为边缘；
Paras[4]:起始点寻找及边缘检测方向；
Paras[5]:起始点偏移；
Paras[6]:起始点寻找起始位置，从哪一条边开始就用起始点到这条边的垂直距离；
Paras[7]:所需点位数量，表示后面有多少段，每一段对应一对起始点和结束点
Paras[8]:第一段的起始位置；
Paras[9]:第一段的结束位置；
......*/
int NewKirschEdgeOuter(cv::Mat srcimg, const vector<int> paras, vector<cv::Point2f>& dist);

/*背光图小半圆边缘提取，旋转线扫，输入模板图像srcimg，输入边缘检测参数paras，输出边缘点集contours。其中paras中参数的释义如下：
Paras[0]:模板序号；
Paras[1]:二值化参数；
Paras[2]:二值化方式，0：kirsch边缘检测后二值化；1：反二值化；2：二值化
Paras[3]:旋转线扫的圆心的x坐标；
Paras[4]:旋转线扫的圆心的y坐标；
Paras[5]:边缘检测筛选区域的半径；
Paras[6]:筛选区域的比例，圆区域中白色像素大于该比例为边缘点；
Paras[7]:旋转线扫起始点离圆心的距离；
Paras[8]:旋转线扫的起始角度；
Paras[9]:旋转线扫的结束角度；*/
    int NewCircleEdge(cv::Mat srcimg, const vector<int> paras, vector<cv::Point>& contours);
//螺纹孔边缘提取，返回圆参数，旋转线扫，输入模板图像srcimg，输入边缘提取参数paras，输出圆参数结果result。参数paras释义同NewCircleEdge，增加了拟合圆步骤。
    int NewSmallCircle(cv::Mat srcimg, const vector<int> paras,  vector<float>& result);
/*LD部分边缘提取，返回点位，直线线扫，输入模板图像srcimg，输入边缘提取参数paras，输出点位结果respts，paras参数释义如下：
Paras[0]:模板序号；
Paras[1]:二值化方式，1：反二值化；2：二值化；
Paras[2]:二值化阈值；
Paras[3]:所需点位数量；
Paras[4]:边缘检测筛选区域的半径；
Paras[5]:筛选区域的比例，圆区域中白色像素大于该比例为边缘点；
以下每4个参数确定一个测量点位：
Paras[6]:线扫起始位置，若线扫方向为上下，起始位置为图像上的y坐标；左右为x坐标；
Paras[7]:边缘起始位置，若线扫方向为上下，边缘的起始和结束位置为x坐标上的一段；若线扫方向为左右，边缘的起始和结束位置为y坐标上的一段；
Paras[8]:边缘结束位置；
Paras[9]:线扫方向；*/
int LDmeasure(cv::Mat srcimg, const vector<int> paras, vector<cv::Point2f>& respts);

/*新螺纹孔边缘提取，返回圆参数，蒙特卡洛算法。输入模板图像srcimg，输入边缘检测参数paras，输出圆参数res（圆心x坐标，y坐标，半径r），flag=1表示圆外像素值高，圆内像素值低；flag=2表示圆外像素值低，圆内像素值高（不完善）。由于现在只需要求表面光下的内径，仅使用flag=1即可。Paras参数释义如下：
Paras[0]:圆心的矩形随机范围在图像上的左上角点的x坐标；
Paras[1]:圆心的矩形随机范围在图像上的左上角点的y坐标；
Paras[2]:圆心的矩形随机范围的width；
Paras[3]:圆心的矩形随机范围的height；
Paras[4]:圆半径的起始大小；
Paras[5]:圆半径的随机范围；
Paras[6]:圆周内外取点的间隔；
Paras[7]:随机取圆次数；
Paras[8]:求内边缘时的基础内缩距离；*/
    int SmallCircle(cv::Mat srcimg, const vector<int> paras, vector<float>& res, int flag);

//求边缘的平行度，输入边缘点contour，输出平行度，flag=1，相对于y轴的平行度；flag=2，相对于x轴的平行度。
float GetParallelism(vector<cv::Point2f> contour,int flag);

//求线与线的距离，输入两条边缘点集，输出线与线的距离，flag=1,x方向上的距离；flag=2,y方向上的距离。
float GetLineDist(vector<cv::Point2f> contour1,vector<cv::Point2f> contour2,int flag);

//求圆度，输入圆的边缘点和圆心，输出圆度
float GetRoundness(vector<cv::Point> contour, cv::Point2f center);

//求位置度，输入标准位置stdposition（物理），测量结果measurepos（物理）
float GetPosition(cv::Point2f stdposition,cv::Point2f measurepos);

//求两条边缘的中心线质心坐标，输入两条边缘contour1,contour2，flag=1,求质心x坐标，flag=2,求质心y坐标
float GetLine2LineMid(vector<cv::Point2f> contour1,vector<cv::Point2f> contour2,int flag);

//拟合圆，输入圆边缘点，输出圆参数（圆心x坐标，y坐标，半径r）
    vector<float> FitCircle(vector<cv::Point>& contours);

//private:
/*直线线扫边缘，
输入模板图像srcimg
输入线扫方向lineori
输出线扫边缘结果contours
输入线扫起始坐标startline
输入边缘起始坐标startedge
输入边缘结束坐标endedge
输入筛选区域半径radius
输入筛选区域比例阈值，白色区域大于该比例为边缘点percent*/
int ScanLineRange(cv::Mat srcimg,int lineori, vector<cv::Point>& contours,int startline,int startedge, int endedge, int radius,float percent);
/*点到直线的距离，
输入点A
输入直线参数Lineparas（k,b）
返回值点到直线的距离*/
float PointLineDist(cv::Point2f A,vector<float> Lineparas);
/*拟合直线，
输入直线边缘点集contours  float
输出直线参数vResult（k,b）*/
void LineFitLeastSquaresf(vector<cv::Point2f> contours, vector<float> &vResult);
/*拟合直线，
输入直线边缘点集contours  int
输出直线参数vResult（k,b）*/
    void LineFitLeastSquares(vector<cv::Point> contours, vector<float> &vResult);
/*用中心点加旋转射线方法找边缘
输入模板图片srcimg
输入线扫起始点seed
输出边缘点集contours*/
    int FindEdge(cv::Mat srcimg, cv::Point seed, vector<cv::Point>& contours, int radius, float percent, int radthresh);
/*已知方向，通过从一边线扫求边缘
输入模板图像srcimg
输入线扫方向lineori
输出边缘检测结果contours
输入线扫起始坐标startline
输入筛选区域半径radius
输入筛选区域比例阈值，白色区域大于该比例为边缘点percent*/
    int ScanLine(cv::Mat srcimg,int lineori, vector<cv::Point>& contours,int startline,int radius,float percent);
/*判断点周围密度
输入模板图像srcimg
输入边缘检测筛选区域圆心center
输入边缘检测筛选区域半径rad
输入边缘检测筛选区域阈值，白色区域大于该比例为边缘点thresh
返回值1为边缘点，返回值0为非边缘点*/
    int PointDense(cv::Mat srcimg, cv::Point center, int rad, float thresh);
/*krisch边缘检测
输入原始图像src
返回边缘检测后的图像*/
    cv::Mat krisch(cv::InputArray src,int borderType = cv::BORDER_DEFAULT);
    //卷积
    void conv2D(cv::InputArray _src, cv::InputArray _kernel, cv::OutputArray _dst, int ddepth, cv::Point anchor = cv::Point(-1, -1), int borderType = cv::BORDER_DEFAULT);
/*找到边缘检测起始位置,单次边缘
输入模板图像srcimg
输入起始位置offset
输入线扫方向orientation
输入边缘起始坐标startline
输入边缘终点坐标endline
输入边缘点灰度阈值threshold
输出边缘检测起始点seedpoints*/
    void GetSeedPoints(cv::Mat srcimg,int offset, int orientation,int startline,int endline,int threshold, vector<cv::Point>& seedpoints);
/*获取图像边缘点,根据像素值，无阈值分割直接根据像素值求边缘
输入模板图像srcimg
输入起始点偏差offset0
输入线扫方向orientation
输入起始点坐标offset
输入边缘起始坐标startline
输入边缘结束坐标endline
输入边缘检测灰度阈值threshold
输入起始点定位灰度阈值threshold1
输出边缘点集结果edgepoints*/
    void GetSrcEdgePoints(cv::Mat srcimg,int offset0,int orientation,int offset,int startline,int endline,int threshold, int threshold1, vector<cv::Point>& edgepoints);
/*获取图像边缘点
输入模板图像srcimg
输入起始点点集seedpoints
输入边缘检测方向orientation
输入kirsch边缘检测结果二值化阈值thresh
输入筛选区域半径radius
输入筛选区域比例阈值percent，白色区域大于该比例为边缘点
输出边缘点集contours*/
    void GetEdgePoint(cv::Mat srcimg, vector<cv::Point> seedpoints, int orientation, int thresh,int radius,float percent, vector<cv::Point>& contours);
/*找到边缘检测起始点位置,两次边缘（越过一次高像素部分，再次遇到高像素部分后查找起始点的位置）
输入模板图像srcimg
输入起始点检测的起始位置偏差offset
输入起始点检测方向orientation
输入边缘起始坐标startline
输入边缘结束坐标endline
输入起始点检测灰度阈值threshold
输出边缘点检测的起始点集坐标seedpoints*/
    void GetSeedPointsSec(cv::Mat srcimg,int offset,int orientation,int startline,int endline,int threshold, vector<cv::Point>& seedpoints);
/*以2*2矩阵求均值生成像素变化曲线
输入模板图像srcimg
输入边缘起始位置startpos
输入边缘结束位置endpos
输入变化曲线方向orientation
输出结果curves，边缘起始位置startpos到边缘结束位置endpos之间每一条像素变化曲线*/
    int Pixel2Curve(cv::Mat srcimg,int startpos, int endpos, int orientation, vector<vector<float>>& curves);
/*找到边缘检测起始位置,单次边缘
输入像素变化曲线curves
输入变化曲线起始位置offset
输入起始点检测灰度阈值thresh
输出边缘点检测的起始点集再curve中的索引*/
    void GetSeedPos(vector<vector<float>> curves,int offset,int thresh, vector<int>& seedpoints);
/*找到边缘检测起始位置,两次边缘
输入像素变化曲线curves
输入变化曲线起始位置offset
输入起始点检测灰度阈值thresh
输出边缘点检测的起始点集再curve中的索引*/
    void GetSeedPosSecond(vector<vector<float>> curves,int offset,int thresh, vector<int>& seedpoints);
/*圆形边缘裁剪
输入圆形边缘点集contours
输入裁剪的起始角度startangle
输入裁剪的结束角度endangle
返回结果裁剪角度内的边缘点集*/
    vector<cv::Point> ContoursCut(vector<cv::Point> contours,int startangle,int endangle);
/*点到点的距离
输入A,B分别为两个二维点*/
    template<typename T1,typename T2>
    float Point2Point(T1 A,T2 B);
};

#endif // EDGEDETECT_H
