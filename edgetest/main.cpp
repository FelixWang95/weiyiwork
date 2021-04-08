#include <iostream>
#include "rectextract.h"

using namespace std;

int main()
{
    //    RectExtract rectex;
    //    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/capimg/classifiedimg/0_L/0_L2/classify4/0/35588-1-1-11.jpg",0);
    //    vector<cv::Point> contours;
    //    rectex.KirschEdgeCircle(srcimg,160,315, 115,contours);

        //GetNewDatumleft();
        cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/newtemplates/S/S3.png",0);
        RectExtract rectex;
        vector<cv::Point> contours;
        //rectex.LDRCircleEdge(srcimg,80,20,contours);
        float dist;
        //rectex.KirschEdgeOuter(srcimg,200,9,dist);
        //rectex.KirschEdgeSmallCircle(srcimg,120,7,contours);
        vector<float> dists;
        rectex.GetRectEdge(srcimg,120,3,dists);
    //    for(int i=0;i<contours.size();++i){
    //        cv::circle(srcimg,contours[i],1,255,-1);
    //    }
    //    cv::imshow("res",srcimg);
        cv::waitKey(0);
        return 0;
}
