
#include "edgedetecction.h"

/*离散的二维卷积运算*/
void conv2D(InputArray _src, InputArray _kernel, OutputArray _dst, int ddepth, Point anchor, int borderType)
{
    //卷积核顺时针旋转180
    Mat kernelFlip;
    flip(_kernel, kernelFlip, -1);
    //针对每一个像素,领域对应元素相乘然后相加
    filter2D(_src, _dst, CV_32FC1, _kernel, anchor, 0.0, borderType);
}

/*roberts 卷积*/
void roberts(InputArray src, OutputArray dst, int ddepth, int x, int y, int borderType)
{
    CV_Assert(!(x == 0 && y == 0));
    Mat roberts_1 = (Mat_<float>(2, 2) << 1, 0, 0, -1);
    Mat roberts_2 = (Mat_<float>(2, 2) << 0, 1, -1, 0);
    //当 x 不等于零时，src 和 roberts_1 卷积
    if (x != 0)
    {
        conv2D(src, roberts_1, dst, ddepth, Point(0, 0), borderType);
    }
    //当 y 不等于零时，src 和 roberts_2 卷积
    if (y != 0)
    {
        conv2D(src, roberts_2, dst, ddepth, Point(0, 0), borderType);
    }
}

cv::Mat getHistograph(const cv::Mat grayImage)
{
    //定义求直方图的通道数目，从0开始索引
    int channels[]={0};
    //定义直方图的在每一维上的大小，例如灰度图直方图的横坐标是图像的灰度值，就一维，bin的个数
    //如果直方图图像横坐标bin个数为x，纵坐标bin个数为y，则channels[]={1,2}其直方图应该为三维的，Z轴是每个bin上统计的数目
    const int histSize[]={256};
    //每一维bin的变化范围
    float range[]={0,256};

    //所有bin的变化范围，个数跟channels应该跟channels一致
    const float* ranges[]={range};

    //定义直方图，这里求的是直方图数据
    cv::Mat hist;
    //opencv中计算直方图的函数，hist大小为256*1，每行存储的统计的该行对应的灰度值的个数
    cv::calcHist(&grayImage,1,channels,cv::Mat(),hist,1,histSize,ranges,true,false);//cv中是cvCalcHist

    //找出直方图统计的个数的最大值，用来作为直方图纵坐标的高
    double maxValue=0;
    //找矩阵中最大最小值及对应索引的函数
    cv::minMaxLoc(hist,0,&maxValue,0,0);
    //最大值取整
    int rows=cvRound(maxValue);
    //定义直方图图像，直方图纵坐标的高作为行数，列数为256(灰度值的个数)
    //因为是直方图的图像，所以以黑白两色为区分，白色为直方图的图像
    cv::Mat histImage=cv::Mat::zeros(rows,256,CV_8UC1);

    //直方图图像表示
    for(int i=0;i<256;i++)
    {
        //取每个bin的数目
        int temp=(int)(hist.at<float>(i,0));
        //如果bin数目为0，则说明图像上没有该灰度值，则整列为黑色
        //如果图像上有该灰度值，则将该列对应个数的像素设为白色
        if(temp)
        {
            //由于图像坐标是以左上角为原点，所以要进行变换，使直方图图像以左下角为坐标原点
            histImage.col(i).rowRange(cv::Range(rows-temp,rows))=255;
        }
    }
    //由于直方图图像列高可能很高，因此进行图像对列要进行对应的缩减，使直方图图像更直观
    cv::Mat resizeImage;
    cv::resize(histImage,resizeImage,cv::Size(256,256));
    return resizeImage;
}
