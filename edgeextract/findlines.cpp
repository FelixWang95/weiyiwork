#include "findlines.h"

FindLines::FindLines()
{

}

int FindLines::PointDense(cv::Mat srcimg, cv::Point center, int rad, float thresh){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat roiimg;
    if(center.x-rad-1<0||center.y-rad-1<0||center.x+rad+1>=srcimg.cols||center.y+rad+1>=srcimg.rows)
    {return 0;}
    else{
        roiimg=srcimg(cv::Rect(cv::Point(center.x-rad-1,center.y-rad-1),cv::Point(center.x+rad+1,center.y+rad+1)));
    }
    cv::Mat maskimg=roiimg.clone();
    cv::circle(maskimg,cv::Point(maskimg.cols/2,maskimg.rows/2),rad,255,-1);
    int sum=0,num=0;
    for(int i=0;i<maskimg.rows;++i){
        for(int j=0;j<maskimg.cols;++j){
            if(srcimg.at<uchar>(center.y-rad-1+i,center.x-rad-1+j)==255&&maskimg.at<uchar>(i,j)==255){
                num++;
            }
            if(maskimg.at<uchar>(i,j)==255)
            {sum++;}
        }
    }
    float percent=(float)num/(float)sum;
    if(percent>=thresh){
        return 1;
    }
    else{return 0;}
}

int FindLines::LineFit(vector<cv::Point> contours,cv::Vec2f& linepara){
    /*
     a = (n*C - B*D) / (n*A - B*B)
     b = (A*D - B*C) / (n*A - B*B)
    其中：
     A = sum(Xi * Xi)
     B = sum(Xi)
     C = sum(Xi * Yi)
     D = sum(Yi)    */
    int dx,flag=0;
    float k1;
    dx=abs(contours[0].x-contours[contours.size()-1].x);
    if(dx>0){
        k1=(float)(contours[0].y-contours[contours.size()-1].y)/(contours[0].x-contours[contours.size()-1].x);
    }
    float A = 0.0;
    float B = 0.0;
    float C = 0.0;
    float D = 0.0;
    if(dx==0||fabs(k1)>10){
        for(int i=0;i<contours.size();++i){
            A+=contours[i].y*contours[i].y;
            B+=contours[i].y;
            C+=contours[i].y*contours[i].x;
            D+=contours[i].x;
        }
        flag=1;
    }
    else{
        for(int i=0;i<contours.size();++i){
            A+=contours[i].x*contours[i].x;
            B+=contours[i].x;
            C+=contours[i].x*contours[i].y;
            D+=contours[i].y;
        }
    }
    double a,b,temp=0;
    temp = (contours.size()*A - B*B);
    if(temp)// 判断分母不为0
    {
        a = (contours.size()*C - B*D) / temp;
        b = (A*D - B*C) / temp;
        if(flag==0){
            linepara[0]=a;
            linepara[1]=b;
        }
        else{
            if(a==0){
                linepara[0]=1;
                linepara[1]=0;
                return -1;//ax+by+c=0:x=c
            }
            linepara[0]=1/a;
            linepara[1]=-b/a;
        }
        return 0;//y=ax+b
    }
    else
    {
        linepara[0]=1;
        linepara[1]=0;
        return -1;
    }
}

void FindLines::LineFitLeastSquares(vector<cv::Point> contours, vector<float> &vResult)
{
    float A = 0.0;
    float B = 0.0;
    float C = 0.0;
    float D = 0.0;
    float E = 0.0;
    float F = 0.0;

    for (int i = 0; i < contours.size(); i++)
    {
        A += contours[i].x * contours[i].x;
        B += contours[i].x;
        C += contours[i].x * contours[i].y;
        D += contours[i].y;
    }
    float a, b, temp = 0;
    if( temp = (contours.size()*A - B*B) )
    {
        a = (contours.size()*C - B*D) / temp;
        b = (A*D - B*C) / temp;
    }
    else
    {
        a = 1;
        b = 0;
    }
    float Xmean, Ymean;
    Xmean = B / contours.size();
    Ymean = D / contours.size();

    float tempSumXX = 0.0, tempSumYY = 0.0;
    for (int i=0; i<contours.size(); i++)
    {
        tempSumXX += (contours[i].x - Xmean) * (contours[i].x - Xmean);
        tempSumYY += (contours[i].y - Ymean) * (contours[i].y - Ymean);
        E += (contours[i].x - Xmean) * (contours[i].y - Ymean);
    }
    F = sqrt(tempSumXX) * sqrt(tempSumYY);

    float r;
    r = E / F;

    vResult.push_back(a);
    vResult.push_back(b);
    vResult.push_back(r*r);
}

int FindLines::GetDouLines(cv::Mat srcimg, vector<cv::Point>& linepoints, int num){
    cv::Mat edge;
    int radius=3;
    float percent=0.5;
    vector<cv::Point> contours;
    cv::Vec2f linepara;
    cv::threshold(srcimg, edge, 80, 255, cv::THRESH_BINARY_INV);
    cv::imshow("edge",edge);
    for(int i=80;i<180;++i){
        cv::Point seed(0,i);
        cv::Point contourpoint(seed.x+1,seed.y);
        while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
            contourpoint.x+=1;
            if(contourpoint.x>=edge.cols)
                break;
        }
        contours.push_back(contourpoint);
    }
    cv::Point startline1,endline1;
    LineFit(contours,linepara);
    startline1.y=80;
    startline1.x=(startline1.y-linepara[1])/linepara[0];
    endline1.y=180;
    endline1.x=(endline1.y-linepara[1])/linepara[0];
    linepoints.push_back(startline1);
    linepoints.push_back(endline1);
    contours.clear();
    for(int i=80;i<180;++i){
        cv::Point seed(edge.cols-1,i);
        cv::Point contourpoint(seed.x-1,seed.y);
        while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
            contourpoint.x-=1;
            if(contourpoint.x==0)
                break;
        }
        contours.push_back(contourpoint);
    }
    cv::Point startline2,endline2;
    LineFit(contours,linepara);
    startline2.y=80;
    startline2.x=(startline2.y-linepara[1])/linepara[0];
    endline2.y=180;
    endline2.x=(endline2.y-linepara[1])/linepara[0];
    linepoints.push_back(startline2);
    linepoints.push_back(endline2);
    for(int i=0;i<linepoints.size();++i){
        cv::circle(srcimg,linepoints[i],3,255,-1);
    }
    cv::imshow("src",srcimg);
    cv::imwrite("/mnt/hgfs/linuxsharefiles/rect_result1/"+std::to_string(num)+".jpg",srcimg);
    return 0;
}

int FindLines::GetLines(cv::Mat srcimg, vector<cv::Point>& linepoints, int num){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    vector<int> leftpoints,rightpoints;
    vector<cv::Point> leftcontours,rightcontours;
    for(int i=80;i<180;++i){
        for(int j=6;j<srcimg.cols/2;++j){
            leftpoints.push_back(srcimg.at<uchar>(i,j));
        }
        vector<int>::iterator leftMin = min_element(leftpoints.begin(), leftpoints.end());
        leftcontours.push_back(cv::Point(6+distance(leftpoints.begin(), leftMin),i));
        leftpoints.clear();
        for(int j=srcimg.cols-1-6;j>srcimg.cols/2;--j){
            rightpoints.push_back(srcimg.at<uchar>(i,j));
        }
        vector<int>::iterator rightMin = min_element(rightpoints.begin(), rightpoints.end());
        rightcontours.push_back(cv::Point(srcimg.cols-1-6-distance(rightpoints.begin(), rightMin),i));
        rightpoints.clear();
    }
    cv::Point startline1,endline1;
    vector<float> linepara;
    for(int i=0;i<leftcontours.size();++i){
        cv::Point temp=leftcontours[i];
        leftcontours[i].x=temp.y;
        leftcontours[i].y=temp.x;
    }
    LineFitLeastSquares(leftcontours,linepara);
    startline1.x=80;
    startline1.y=startline1.x*linepara[0]+linepara[1];
    endline1.x=180;
    endline1.y=endline1.x*linepara[0]+linepara[1];
    linepoints.push_back(startline1);
    linepoints.push_back(endline1);
    cv::Point startline2,endline2;
    linepara.clear();
    for(int i=0;i<rightcontours.size();++i){
        cv::Point temp=rightcontours[i];
        rightcontours[i].x=temp.y;
        rightcontours[i].y=temp.x;
    }
    LineFitLeastSquares(rightcontours,linepara);
    startline2.x=80;
    startline2.y=startline2.x*linepara[0]+linepara[1];
    endline2.x=180;
    endline2.y=endline2.x*linepara[0]+linepara[1];
    linepoints.push_back(startline2);
    linepoints.push_back(endline2);
    for(int i=0;i<leftcontours.size();++i){
        cv::Point temp=leftcontours[i];
        leftcontours[i].x=temp.y;
        leftcontours[i].y=temp.x;
    }
    for(int i=0;i<rightcontours.size();++i){
        cv::Point temp=rightcontours[i];
        rightcontours[i].x=temp.y;
        rightcontours[i].y=temp.x;
    }
    for(int i=0;i<linepoints.size();++i){
        cv::Point temp=linepoints[i];
        linepoints[i].x=temp.y;
        linepoints[i].y=temp.x;
    }
    for(int i=0;i<leftcontours.size();++i){
        cv::circle(srcimg,leftcontours[i],1,255,-1);
    }
    for(int i=0;i<rightcontours.size();++i){
        cv::circle(srcimg,rightcontours[i],1,255,-1);
    }
    for(int i=0;i<linepoints.size();++i){
        cv::circle(srcimg,linepoints[i],5,255,-1);
    }
    cv::imshow("src",srcimg);
    cv::imwrite("/mnt/hgfs/linuxsharefiles/rect_result1/"+std::to_string(num)+".jpg",srcimg);
    return 0;
}
