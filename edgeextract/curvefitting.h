#pragma once
#include<opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <math.h>
#include <fstream>
class curvefitting
{
public:
	curvefitting();
	~curvefitting();
    void operator()(std::vector<std::vector<cv::Point2d>> &ptVec,std::vector<std::vector<float>> &res);
};

