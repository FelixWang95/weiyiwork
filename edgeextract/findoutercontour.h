#ifndef FINDOUTERCONTOUR_H
#define FINDOUTERCONTOUR_H
#include "headerall.h"

#define PI 3.1415926

class FindOuterContour
{
public:
    FindOuterContour();
    void PeripheralTraversa(cv::Mat srcImg,std::vector<float> &source_x, std::vector<float> &source_y);
    void ScanPoints(cv::Mat imgGray,std::vector<float> &source_x, std::vector<float> &source_y);
    void setNN(int nn);
    void findSquares( const cv::Mat& image, vector<vector<Point> >& squares );
    void drawSquares( cv::Mat& image, const vector<vector<Point> >& squares );
    double angle( Point pt1, Point pt2, Point pt0 );
    void PerspectiveTransformation(const cv::Mat &image,cv::Mat &dstImage);
    void getsrcTri(vector<Point2f> &srcAns);
    //边缘点排序
    vector<cv::Point> SortContour(vector<cv::Point> contour);
    vector<cv::Point> SortContourCenter(vector<cv::Point> contour, cv::Point center);
    float GetAngle(cv::Point A, cv::Point B, cv::Point C, cv::Point D);
    int NN;

private:
    float th1;
    float th2;
    float th3;
    Point2f srcTri[4];
    Point2f dstTri[4];
};

#endif // FINDOUTERCONTOUR_H
