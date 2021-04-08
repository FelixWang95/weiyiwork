#include "physicalquantity.h"

PhysicalQuantity::PhysicalQuantity()
{

}
void PhysicalQuantity::getObject(cv::Mat &srcImg,cv::Mat &referImg,vector<float> &result)
{
    cv::Mat binary;
    Scalar v=cv::mean(referImg);
    float thr=v.val[0]*0.9;
    threshold(srcImg,binary,thr,255,THRESH_BINARY_INV);
    cv::imshow("binimg",binary);
    cv::Mat  labels, stats, centroids;
    int num_labels = connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
    int maxareaId=-1,maxarea=-1;
    for (int i = 1; i < num_labels; i++)
    {
        int area = stats.at<int>(i, CC_STAT_AREA);
        if(area>maxarea)
        {
            maxarea=area;
            maxareaId=i;
        }
    }
    for (int row = 0; row < binary.rows; row++) {
        for (int col = 0; col < binary.cols; col++) {
            int label = labels.at<int>(row, col);
            label == maxareaId?binary.at<uchar>(row,col)=255:binary.at<uchar>(row,col)=0;
        }
    }
    multiset<uchar> light;
    cv::Mat nozcd;
    findNonZero(binary,nozcd);
    for (int i=0;i<int(nozcd.total());i++)
    {
        light.insert(srcImg.at<uchar>(nozcd.at<Point>(i).y,nozcd.at<Point>(i).x));
    }
    int cnt=0,cnt1=0,cnt2=0;
    float l1=0.0f,l2=0.0f;
    for(multiset<uchar>::iterator it=light.begin();it!=light.end();it++)
    {
        if(cnt<int(light.size()/5))
        {
            l1+=*it;
            cnt1++;
        }
        if(cnt>int(light.size()/5*4))
        {
            l2+=*it;
            cnt2++;
        }
        cnt++;
    }
    l1/=float(cnt1),l2/=float(cnt2);
    Canny(binary,binary,50,150);
    findNonZero(binary,nozcd);
    vector<float> disVec;
    for (int i=0;i<int(nozcd.total());i++)
    {
        for(int j=0;j<int(nozcd.total());j++)
        {
            if(j==i) continue;
            double xt=nozcd.at<Point>(i).x-nozcd.at<Point>(j).x;
            double yt=nozcd.at<Point>(i).y-nozcd.at<Point>(j).y;
            disVec.push_back(float(gsl_hypot(xt,yt)));
        }
    }
    float length=*max_element(disVec.begin(),disVec.end());
    result.push_back(maxarea),result.push_back(nozcd.total()),result.push_back(length);
    result.push_back(l1),result.push_back(l2),result.push_back(v.val[0]);

    //cout<<"area: "<<result[0]<<"\n"<<"perimeter:"<<result[1]<<"\n"
    //   <<"length: "<<result[2]<<"\n"<<"min light: "<<result[3]<<"\n"
    //    <<"max light: "<<result[4]<<"\n"<<"background: "<<result[5]<<endl;

}
