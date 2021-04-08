#include <iostream>
#include "edgedetecction.h"

using namespace std;

//int main()
//{
//    /*第一步：输入灰度图像矩阵*/
//    Mat image = imread("/home/adt/QTcode/edgeextract/001.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//    if (!image.data)
//    {
//        cout << "没有图片" << endl;
//        return -1;
//    }
//    image=image(cv::Rect(cv::Point(1857,2107),cv::Point(2085,2245)));
//    imshow("circle1",image);
//    Mat edge;
//    Canny(image, edge, 90, 180, 3);
//    imshow("cannyEdge", edge);
//    imwrite("cannyEdge.png", edge);
//    waitKey(0);
//    return 0;
//}

//roberts
//Mat edge;//边缘图
//int Thresh = 25;//阈值
//const int MAX_THRESH = 255;
//void callback_thresh(int, void*)
//{
//    Mat copyEdge = edge.clone();
//    Mat thresh_edge;//阈值处理后的阈值
//    threshold(copyEdge, thresh_edge, Thresh, MAX_THRESH, cv::THRESH_BINARY);
//    imshow("阈值处理后的边缘强度", thresh_edge);
//    imwrite("thresh_edge.jpg", thresh_edge);
//}
////主函数
//int main(int argc, char*argv[])
//{
//    /*第一步：输入灰度图像矩阵*/
//    Mat image = imread("/home/adt/QTcode/edgeextract/001.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//    if (!image.data)
//    {
//        cout << "没有图片" << endl;
//        return -1;
//    }
//    image=image(cv::Rect(cv::Point(1857,2107),cv::Point(2085,2245)));
//    imshow("circle1",image);
//    //imwrite("circle1.png",image);
//    /*第二步： roberts 卷积*/
//    //图像矩阵和 roberts_1 卷积核卷积
//    Mat img_roberts_1;
//    roberts(image, img_roberts_1, CV_32FC1, 1, 0,4);
//    //图像矩阵和 roberts_2 卷积核卷积
//    Mat img_roberts_2;
//    roberts(image, img_roberts_2, CV_32FC1, 0, 1,4);
//    //两个卷积结果的灰度级显示
//    Mat abs_img_roberts_1, abs_img_roberts_2;
//    convertScaleAbs(img_roberts_1, abs_img_roberts_1, 1, 0);
//    convertScaleAbs(img_roberts_2, abs_img_roberts_2, 1, 0);
//    imshow("135°方向上的边缘", abs_img_roberts_1);
//    imwrite("img_robert_135_edge.jpg", abs_img_roberts_1);
//    imshow("45°方向上的边缘", abs_img_roberts_2);
//    imwrite("img_robert_45_edge.jpg", abs_img_roberts_2);
//    /*第三步：通过第二步得到的两个卷积结果，求出最终的边缘强度*/
//    //这里采用平方根的方式
//    Mat img_roberts_1_2, img_roberts_2_2;
//    pow(img_roberts_1, 2.0, img_roberts_1_2);
//    pow(img_roberts_2, 2.0, img_roberts_2_2);
//    sqrt(img_roberts_1_2 + img_roberts_2_2, edge);
//    //数据类型转换，边缘强度的灰度级显示
//    edge.convertTo(edge, CV_8UC1);
//    imshow("边缘强度", edge);
//    imwrite("img_robert_edge.jpg", edge);
//    //阈值处理后的边缘强度
//    namedWindow("阈值处理后的边缘强度", cv::WINDOW_AUTOSIZE);
//    createTrackbar("阈值", "阈值处理后的边缘强度", &Thresh, MAX_THRESH, callback_thresh);
//    callback_thresh(0, 0);
//    //显示浮雕效果
//    Mat reliefFigure = img_roberts_1 + 128;
//    Mat reliefImage;
//    reliefFigure.convertTo(reliefImage, CV_8UC1);
//    imshow("浮雕图效果", reliefImage);
//    imwrite("reliefImage.jpg", reliefImage);
//    waitKey(0);
//    return 0;
//}

//prewitt
//Mat edge;//边缘图
//int Thresh = 255;//阈值
//const int MAX_THRSH = 255;
//void callback_thresh(int, void*)
//{
//    Mat copyEdge = edge.clone();
//    Mat thresh_edge;//阈值处理后的阈值
//    threshold(copyEdge, thresh_edge, Thresh, MAX_THRSH, cv::THRESH_BINARY);
//    imshow("阈值处理后的边缘强度", thresh_edge);
//    imwrite("prewitt.png",thresh_edge);
//}
////主函数
//int main(int argc, char*argv[])
//{
//    /*第一步：输入灰度图像矩阵*/
//    Mat image = imread("/home/adt/QTcode/edgeextract/001.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//    if (!image.data)
//    {
//        cout << "没有图片" << endl;
//        return -1;
//    }
//    image=image(cv::Rect(cv::Point(1857,2107),cv::Point(2085,2245)));
//    imshow("circle1",image);
//    /*第二步:previwitt卷积*/

//    //图像矩阵和 prewitt_x卷积核的卷积
//    Mat img_prewitt_x;
//    prewitt(image, img_prewitt_x,CV_32FC1,1, 0);
//    //图像矩阵与prewitt_y卷积核卷积
//    Mat img_prewitt_y;
//    prewitt(image, img_prewitt_y, CV_32FC1, 0, 1);

//    /*第三步:水平方向和垂直方向的边缘强度*/
//    //数据类型转换,边缘强度的灰度级显示
//    Mat abs_img_prewitt_x, abs_img_prewitt_y;
//    convertScaleAbs(img_prewitt_x, abs_img_prewitt_x, 1, 0);
//    convertScaleAbs(img_prewitt_y, abs_img_prewitt_y, 1, 0);
//    imshow("垂直方向的边缘", abs_img_prewitt_x);
//    //imwrite("img1_v_edge.jpg", abs_img_prewitt_x);
//    imshow("水平方向的边缘", abs_img_prewitt_y);
//    //imwrite("img1_h_edge.jpg", abs_img_prewitt_y);
//    /*第四步：通过第三步得到的两个方向的边缘强度,求出最终的边缘强度*/
//    //这里采用平方根的方式
//    Mat img_prewitt_x2, image_prewitt_y2;
//    pow(img_prewitt_x,2.0,img_prewitt_x2);
//    pow(img_prewitt_y,2.0,image_prewitt_y2);
//    sqrt(img_prewitt_x2 + image_prewitt_y2, edge);
//    //数据类型转换,边缘的强度灰度级显示
//    edge.convertTo(edge, CV_8UC1);
//    imshow("边缘强度",edge);
//    //imwrite("img1_edge.jpg", edge);
//    Mat tempedge;
//    threshold(edge, tempedge, 25, MAX_THRSH, cv::THRESH_BINARY);
//    imwrite("img3_thresh_edge_25.jpg", tempedge);
//    //阈值处理后的边缘强度
//    namedWindow("阈值处理后的边缘强度", cv::WINDOW_AUTOSIZE);
//    createTrackbar("阈值", "阈值处理后的边缘强度", &Thresh,MAX_THRSH,callback_thresh);
//    callback_thresh(0, 0);
//    waitKey(0);
//    return 0;
//}

//Sobel
//int main(int argc, char*argv[])
//{
//    /*第一步：输入灰度图像矩阵*/
//    Mat image = imread("/home/adt/QTcode/edgeextract/001.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//    if (!image.data)
//    {
//        cout << "没有图片" << endl;
//        return -1;
//    }
//    image=image(cv::Rect(cv::Point(1857,2107),cv::Point(2085,2245)));
//    imshow("circle1",image);
//    /* --- sobel 边缘检测 --- */
//    //与水平方向的 sobel 核卷积
//    Mat image_Y_X = sobel(image, 1, 0, 3, 4);
//    //垂直方向的边缘强度
//    Mat imageYX_abs = abs(image_Y_X);
//    //垂直方向边缘强度的灰度级显示
//    Mat imageYX_gray;
//    imageYX_abs.convertTo(imageYX_gray, CV_8UC1, 1.0, 0);
//    imshow("垂直方向的边缘强度", imageYX_gray);
//    //与垂直方向的 sobel 核卷积
//    Mat image_X_Y = sobel(image, 0, 1, 3, 4);
//    //水平方向的边缘强度
//    Mat imageXY_abs = abs(image_X_Y);
//    //水平方向边缘强度的灰度级显示
//    Mat imageXY_gray;
//    imageXY_abs.convertTo(imageXY_gray, CV_8UC1, 1.0, 0);
//    imshow("水平方向的边缘强度", imageXY_gray);
//    //根据垂直方向和水平方向边缘强度的平方和，得到最终的边缘强度
//    Mat edge;
//    magnitude(image_Y_X, image_X_Y, edge);
//    //边缘强度的灰度级显示
//    edge.convertTo(edge, CV_8UC1, 1.0, 0);
//    threshold(edge,edge,160,255,THRESH_BINARY);
//    imshow("边缘", edge);
//    imwrite("sobeledge.png", edge);
//    waitKey(0);
//    return 0;
//}

//scharr
//Mat edge;//边缘图
//int Thresh = 255;//阈值
//const int MAX_THRESH = 255;
//void callback_thresh(int, void*)
//{
//    Mat copyEdge = edge.clone();
//    Mat thresh_edge;//阈值处理后的阈值
//    threshold(copyEdge, thresh_edge, Thresh, MAX_THRESH, cv::THRESH_BINARY);
//    imshow("阈值处理后的边缘强度", thresh_edge);
//    imwrite("scharr.png",thresh_edge);
//}
////主函数
//int main(int argc, char*argv[])
//{
//    /*第一步：输入灰度图像矩阵*/
//    Mat image = imread("/home/adt/QTcode/edgeextract/001.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//    if (!image.data)
//    {
//        cout << "没有图片" << endl;
//        return -1;
//    }
//    image=image(cv::Rect(cv::Point(1857,2107),cv::Point(2085,2245)));
//    imshow("circle1",image);
//    /*第二步： scharr 卷积*/
//    //图像矩阵和 scharr_x 卷积核卷积
//    Mat img_scharr_x;
//    scharr(image, img_scharr_x, CV_32FC1, 1, 0);
//    //图像矩阵和 scharr_y 卷积核卷积
//    Mat img_scharr_y;
//    scharr(image, img_scharr_y, CV_32FC1, 0, 1);
//    //两个卷积结果的灰度级显示
//    Mat abs_img_scharr_x, abs_img_scharr_y;
//    convertScaleAbs(img_scharr_x, abs_img_scharr_x, 1, 0);
//    convertScaleAbs(img_scharr_y, abs_img_scharr_y, 1, 0);
//    imshow("垂直方向的边缘", abs_img_scharr_x);
//    imwrite("img1_sch_v_edge.jpg", abs_img_scharr_x);
//    imshow("水平方向的边缘", abs_img_scharr_y);
//    imwrite("img1_sch_h_edge.jpg", abs_img_scharr_y);
//    /*第三步：通过第二步得到的两个卷积结果，求出最终的边缘强度*/
//    //这里采用平方根的方式
//    Mat img_scharr_x2, img_scharr_y2;
//    pow(img_scharr_x, 2.0, img_scharr_x2);
//    pow(img_scharr_y, 2.0, img_scharr_y2);
//    sqrt(img_scharr_x2 + img_scharr_y2, edge);
//    //数据类型转换，边缘强度的灰度级显示
//    edge.convertTo(edge, CV_8UC1);
//    imshow("边缘强度", edge);
//    imwrite("img1_sch_edge.jpg", edge);
//    //阈值处理后的边缘强度
//    namedWindow("阈值处理后的边缘强度", cv::WINDOW_AUTOSIZE);
//    createTrackbar("阈值", "阈值处理后的边缘强度", &Thresh, MAX_THRESH, callback_thresh);
//    callback_thresh(0, 0);
//    waitKey(0);
//    return 0;
//}

//kirsch
//Mat edge;//边缘图
//int Thresh = 255;//阈值
//const int MAX_THRSH = 255;
//void callback_thresh(int, void*)
//{
//    Mat copyEdge = edge.clone();
//    Mat thresh_edge;//阈值处理后的阈值
//    threshold(copyEdge, thresh_edge, Thresh, MAX_THRSH, cv::THRESH_BINARY);
//    imshow("阈值处理后的边缘强度", thresh_edge);
//    imwrite("kirsch.png",thresh_edge);
//}
////主函数
//int main(int argc, char*argv[])
//{
//    /*第一步：输入灰度图像矩阵*/
//    Mat image = imread("/home/adt/QTcode/edgeextract/001.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//    if (!image.data)
//    {
//        cout << "没有图片" << endl;
//        return -1;
//    }
//    image=image(cv::Rect(cv::Point(1857,2107),cv::Point(2085,2245)));
//    imshow("circle1",image);
//    /*第二步：求边缘图*/
//    edge = krisch(image);
//    //显示边缘图
//    imshow("krischEdge", edge);
//    imwrite("img_kri_edge.jpg", edge);
//    //阈值处理后的边缘强度
//    namedWindow("阈值处理后的边缘强度", cv::WINDOW_AUTOSIZE);
//    createTrackbar("阈值", "阈值处理后的边缘强度", &Thresh, MAX_THRSH, callback_thresh);
//    callback_thresh(0, 0);
//    waitKey(0);
//    return 0;
//}

//Laplacian
Mat edge;//边缘图
int Thresh = 255;//阈值
const int MAX_THRSH = 255;
void callback_thresh(int, void*)
{
    Mat copyEdge = edge.clone();
    Mat thresh_edge;//阈值处理后的阈值
    threshold(copyEdge, thresh_edge, Thresh, MAX_THRSH, cv::THRESH_BINARY);
    imshow("阈值处理后的边缘强度", thresh_edge);
    imwrite("laplacian.png",thresh_edge);
}
//主函数
int main(int argc, char*argv[])
{
    /*第一步：输入灰度图像矩阵*/
    Mat image = imread("/home/adt/QTcode/edgeextract/001.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    if (!image.data)
    {
        cout << "没有图片" << endl;
        return -1;
    }
    image=image(cv::Rect(cv::Point(1857,2107),cv::Point(2085,2245)));
    imshow("circle1",image);
    /*第二步： laplacian 卷积*/
    Mat img_lap;
    laplacian(image, img_lap, CV_32FC1);
    //数据类型的转换，卷积结果的灰度级显示
    convertScaleAbs(img_lap, edge, 1, 0);
    imshow("边缘强度", edge);
//    cv::Mat histimg=getHistograph(edge);
//    cv::imshow("Hist",histimg);
//    cv::imwrite("histlaplacian.jpg",histimg);
    imwrite("img3_lap.jpg", edge);
    //阈值处理后的边缘强度
    namedWindow("阈值处理后的边缘强度", cv::WINDOW_AUTOSIZE);
    createTrackbar("阈值", "阈值处理后的边缘强度", &Thresh, MAX_THRSH, callback_thresh);
    callback_thresh(0, 0);
    waitKey(0);
    return 0;
}

//LOG
//int main(int argc, char*argv[])
//{
//    /*第一步：输入灰度图像矩阵*/
//    Mat image = imread("/home/adt/QTcode/edgeextract/001.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//    if (!image.data)
//    {
//        cout << "没有图片" << endl;
//        return -1;
//    }
//    image=image(cv::Rect(cv::Point(1857,2107),cv::Point(2085,2245)));
//    imshow("circle1",image);
//    // LoG 卷积
//    float sigma = 4;
//    int win = 25;
//    Mat loG = LoG(image, sigma, win);
//    //数据类型转换，转换为 CV_8U
//    /*
//    // 二值边缘
//    Mat threshEdge=Mat::zeros(loG.size(), CV_8UC1);
//    int rows = threshEdge.rows;
//    int cols = threshEdge.cols;
//    for (int r = 0; r < rows; r++)
//    {
//        for (int c = 0; c < cols; c++)
//        {
//            if (loG.at<float>(r, c) > 0)
//                threshEdge.at<uchar>(r, c) = 255;
//        }
//    }
//    */
//    //以 0 为阈值，生成边缘二值图
//    cout<<loG.type()<<endl;
//    Mat edge;
//    threshold(loG, edge, 20, 255, THRESH_BINARY);
//    cout<<edge.type()<<endl;
//    edge.convertTo(edge, CV_8UC1);
//    imshow("二值边缘图", edge);
//    imwrite("LoG.png",edge);
//    waitKey(0);
//    return 0;
//}

//DoG
//int main(int argc, char*argv[])
//{
//    /*第一步：输入灰度图像矩阵*/
//    Mat image = imread("/home/adt/QTcode/edgeextract/001.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//    if (!image.data)
//    {
//        cout << "没有图片" << endl;
//        return -1;
//    }
//    cv::Mat I=image(cv::Rect(cv::Point(1857,2107),cv::Point(2085,2245)));
//    imshow("circle1",I);
//    //高斯差分
//    float sigma = 6;
//    int s = 37;
//    float k = 1.05;
//    Mat doG=DoG(I, sigma, s, k);
//    //阈值处理
//    Mat edge;
//    threshold(doG, edge, 8, 255, THRESH_BINARY);
//    //显示二值边缘
//    imshow("高斯差分边缘", edge);
//    imwrite("DoG.png", edge);
//    waitKey(0);
//    return 0;
//}
