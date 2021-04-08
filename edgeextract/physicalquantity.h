#ifndef PHYSICALQUANTITY_H
#define PHYSICALQUANTITY_H
#include "headerall.h"

class PhysicalQuantity
{
public:
    PhysicalQuantity();
	//srcImg:输入的包含缺陷的小块截取图像;referImg:参考相邻良品图像块;
	//vector<float> &result:返回结果是6维的物理量参数依次为，面积，周长，长度，缺陷最暗20%部分的均值，缺陷最亮20%部分的均值,背景亮度
    void getObject(cv::Mat &srcImg,cv::Mat &referImg,vector<float> &result);
};

#endif // PHYSICALQUANTITY_H
