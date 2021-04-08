#ifndef EDGEDETECCTION_H
#define EDGEDETECCTION_H
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void conv2D(InputArray _src, InputArray _kernel, OutputArray _dst, int ddepth, Point anchor = Point(-1, -1), int borderType = BORDER_DEFAULT);
void roberts(InputArray src, OutputArray dst, int ddepth, int x, int y = 0, int borderType = BORDER_DEFAULT);
void sepConv2D_X_Y(InputArray src, OutputArray src_kerX_kerY, int ddepth, InputArray kernelX, InputArray kernelY, Point anchor = Point(-1, -1), int borderType = BORDER_DEFAULT);
void sepConv2D_Y_X(InputArray src, OutputArray src_kerY_kerX, int ddepth, InputArray kernelY, InputArray kernelX, Point anchor = Point(-1, -1), int borderType = BORDER_DEFAULT);
void prewitt(InputArray src,OutputArray dst, int ddepth,int x, int y = 0, int borderType = BORDER_DEFAULT);
int factorial(int n);
Mat getPascalSmooth(int n);
Mat getPascalDiff(int n);
Mat sobel(Mat image, int x_flag, int y_flag, int winSize, int borderType);
void scharr(InputArray src, OutputArray dst, int ddepth, int x, int y = 0, int borderType = BORDER_DEFAULT);
Mat krisch(InputArray src,int borderType = BORDER_DEFAULT);
void laplacian(InputArray src, OutputArray dst, int ddepth,int borderType = BORDER_DEFAULT);
void getSepLoGKernel(float sigma,int length,Mat & kernelX,Mat & kernelY);
Mat LoG(InputArray image,float sigma,int win);
Mat gaussConv(Mat I,float sigma,int s);
Mat DoG(Mat I, float sigma, int s, float k=1.1);
cv::Mat getHistograph(const cv::Mat grayImage);

#endif // EDGEDETECCTION_H
