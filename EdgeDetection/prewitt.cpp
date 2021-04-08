#include "edgedetecction.h"

/*可分离的离散二维卷积,先水平方向的卷积,然后接着进行垂直方向的卷积*/
void sepConv2D_X_Y(InputArray src, OutputArray src_kerX_kerY, int ddepth, InputArray kernelX, InputArray kernelY, Point anchor, int borderType)
{
    //输入矩阵与水平方向卷积核的卷积
    Mat src_kerX;
    conv2D(src, kernelX, src_kerX, ddepth, anchor, borderType);
    //上面得到的卷积结果，然后接着和垂直方向的卷积核卷积，得到最终的输出
    conv2D(src_kerX, kernelY, src_kerX_kerY, ddepth, anchor, borderType);
}
/*可分离的离散二维卷积,先垂直方向的卷积，然后进行水平方向的卷积*/
void sepConv2D_Y_X(InputArray src, OutputArray src_kerY_kerX, int ddepth, InputArray kernelY, InputArray kernelX, Point anchor, int borderType)
{
    //输入矩阵与垂直方向卷积核的卷积
    Mat src_kerY;
    conv2D(src, kernelY, src_kerY, ddepth, anchor, borderType);
    //上面得到的卷积结果，然后接着和水平方向的卷积核卷积，得到最终的输出
    conv2D(src_kerY, kernelX, src_kerY_kerX, ddepth, anchor, borderType);
}

/*
    prewitt卷积运算
*/
void prewitt(InputArray src,OutputArray dst, int ddepth,int x, int y, int borderType)
{
    CV_Assert(!(x == 0 && y == 0));
    //如果 x!=0，src 和 prewitt_x卷积核进行卷积运算
    if (x != 0)
    {
        //可分离的prewitt_x卷积核
        Mat prewitt_x_y = (Mat_<float>(3, 1) << 1, 1, 1);
        Mat prewitt_x_x = (Mat_<float>(1, 3) << 1, 0, -1);
        //可分离的离散的二维卷积
        sepConv2D_Y_X(src, dst, ddepth, prewitt_x_y, prewitt_x_x, Point(-1, -1), borderType);
    }
    if (y != 0)
    {
        //可分离的prewitt_y卷积核
        Mat prewitt_y_x = (Mat_<float>(1, 3) << 1, 1, 1);
        Mat prewitt_y_y = (Mat_<float>(3, 1) << 1, 0, -1);
        //可分离的离散二维卷积
        sepConv2D_X_Y(src, dst, ddepth, prewitt_y_x, prewitt_y_y, Point(-1, -1), borderType);
    }
}
