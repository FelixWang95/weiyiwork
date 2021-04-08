#ifndef DBSCAN_H
#define DBSCAN_H
#include <iostream>
using namespace std;

#include "opencv2/opencv.hpp"
#define PI 3.1415920
using namespace cv;
#include <algorithm>
#include <stdlib.h>

using namespace cv;
struct resdata
{
    int label;
    cv::Point2d kbcore;
    std::vector<Point>cluster;

};

class DBSCAN
{
    int r;
    int points;


public:
    DBSCAN();
    DBSCAN(float erps,int minpts);
    void res(std::vector<Point2d>p,std::vector<resdata>&s);
};

#endif // DBSCAN_H
