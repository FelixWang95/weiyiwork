#include "rectextract.h"

cv::RNG rng((unsigned)time(NULL));

RectExtract::RectExtract()
{

}

/*离散的二维卷积运算*/
void RectExtract::conv2D(cv::InputArray _src, cv::InputArray _kernel, cv::OutputArray _dst, int ddepth, cv::Point anchor, int borderType)
{
    //卷积核顺时针旋转180
    cv::Mat kernelFlip;
    cv::flip(_kernel, kernelFlip, -1);
    //针对每一个像素,领域对应元素相乘然后相加
    cv::filter2D(_src, _dst, CV_32FC1, _kernel, anchor, 0.0, borderType);
}
/* Krisch 边缘检测算法*/
cv::Mat RectExtract::krisch(cv::InputArray src,int borderType)
{
    //存储八个卷积结果
    vector<cv::Mat> eightEdge;
    eightEdge.clear();
    /*第1步：图像矩阵与8 个 卷积核卷积*/
    /*Krisch 的 8 个卷积核均不是可分离的*/
    //图像矩阵与 k1 卷积
    cv::Mat k1 = (cv::Mat_<float>(3, 3) << 5, 5, 5, -3, 0, -3, -3, -3, -3);
    cv::Mat src_k1;
    conv2D(src, k1, src_k1, CV_32FC1);
    cv::convertScaleAbs(src_k1, src_k1);
    eightEdge.push_back(src_k1);
    //图像矩阵与 k2 卷积
    cv::Mat k2 = (cv::Mat_<float>(3, 3) << -3, -3, -3, -3, 0, -3, 5, 5, 5);
    cv::Mat src_k2;
    conv2D(src, k2, src_k2, CV_32FC1);
    cv::convertScaleAbs(src_k2, src_k2);
    eightEdge.push_back(src_k2);
    //图像矩阵与 k3 卷积
    cv::Mat k3 = (cv::Mat_<float>(3, 3) << -3, 5, 5, -3, 0, 5, -3, -3, -3);
    cv::Mat src_k3;
    conv2D(src, k3, src_k3, CV_32FC1);
    cv::convertScaleAbs(src_k3, src_k3);
    eightEdge.push_back(src_k3);
    //图像矩阵与 k4 卷积
    cv::Mat k4 = (cv::Mat_<float>(3, 3) << -3, -3, -3, 5, 0, -3, 5, 5, -3);
    cv::Mat src_k4;
    conv2D(src, k4, src_k4, CV_32FC1);
    cv::convertScaleAbs(src_k4, src_k4);
    eightEdge.push_back(src_k4);
    //图像矩阵与 k5 卷积
    cv::Mat k5 = (cv::Mat_<float>(3, 3) << -3, -3, 5, -3, 0, 5, -3, -3, 5);
    cv::Mat src_k5;
    conv2D(src, k5, src_k5, CV_32FC1);
    cv::convertScaleAbs(src_k5, src_k5);
    eightEdge.push_back(src_k5);
    //图像矩阵与 k6 卷积
    cv::Mat k6 = (cv::Mat_<float>(3, 3) << 5, -3, -3, 5, 0, -3, 5, -3, -3);
    cv::Mat src_k6;
    conv2D(src, k6, src_k6, CV_32FC1);
    cv::convertScaleAbs(src_k6, src_k6);
    eightEdge.push_back(src_k6);
    //图像矩阵与 k7 卷积
    cv::Mat k7 = (cv::Mat_<float>(3, 3) << -3, -3, -3, -3, 0, 5, -3, 5, 5);
    cv::Mat src_k7;
    conv2D(src, k7, src_k7, CV_32FC1);
    cv::convertScaleAbs(src_k7, src_k7);
    eightEdge.push_back(src_k7);
    //图像矩阵与 k8 卷积
    cv::Mat k8 = (cv::Mat_<float>(3, 3) << 5, 5, -3, 5, 0, -3, -3, -3, -3);
    cv::Mat src_k8;
    conv2D(src, k8, src_k8, CV_32FC1);
    cv::convertScaleAbs(src_k8, src_k8);
    eightEdge.push_back(src_k8);
    /*第二步：将求得的八个卷积结果,取对应位置的最大值，作为最后的边缘输出*/
    cv::Mat krischEdge = eightEdge[0].clone();
    for (int i = 0; i < 8; i++)
    {
        cv::max(krischEdge, eightEdge[i], krischEdge);
    }
    return krischEdge;
}

int RectExtract::KirschEdgeLine(cv::Mat srcimg,int threshold,int orientation, vector<cv::Point>& contours){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    int radius=5;
    float percent=0.6;
    cv::Mat edge = krisch(srcimg);
    int startline;
    cv::threshold(edge, edge, threshold, 255, cv::THRESH_BINARY);
    if(orientation==SCAN_DIRECTION_UP){
        startline=srcimg.rows/2;
        for(int i=radius+1;i<edge.cols-radius-1;++i){
            cv::Point seed(i,startline);
            cv::Point contourpoint(i,seed.y-1);
            while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y-=1;
                if(contourpoint.y==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
        contours=GetSegments(contours,vector<int>{45,60,235,250});
    }
    else if(orientation==SCAN_DIRECTION_RIGHT){
        startline=srcimg.cols/2;
        for(int i=radius+1;i<edge.rows-radius-1;++i){
            cv::Point seed(startline,i);
            cv::Point contourpoint(seed.x+1,i);
            while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x+=1;
                if(contourpoint.x>=edge.cols)
                    break;
            }
            contours.push_back(contourpoint);
        }
        contours=GetSegments(contours,vector<int>{50,65,240,255});
    }
    else if(orientation==SCAN_DIRECTION_DOWN){
        startline=0;//srcimg.rows/2
        for(int i=radius+1;i<edge.cols-radius-1;++i){
            cv::Point seed(i,startline);
            cv::Point contourpoint(i,seed.y+1);
            while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y+=1;
                if(contourpoint.y>=edge.rows)
                    break;
            }
            contours.push_back(contourpoint);
        }
        contours=GetSegments(contours,vector<int>{10,30,200,220});//65,80,250,265
    }
    else if(orientation==SCAN_DIRECTION_LEFT){
        startline=srcimg.cols/2;
        for(int i=radius+1;i<edge.rows-radius-1;++i){
            cv::Point seed(startline,i);
            cv::Point contourpoint(seed.x-1,i);
            while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x-=1;
                if(contourpoint.x==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
        contours=GetSegments(contours,vector<int>{65,80,250,265});
    }
    return 0;
}

int RectExtract::KirschEdgeInnerLine(cv::Mat srcimg,int threshold,int orientation, int startline, vector<int> segments, vector<cv::Point>& contours){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    int radius=5;
    float percent=0.8;
    //cv::Mat edge = krisch(srcimg);
    cv::Mat edge=srcimg;
    cv::threshold(edge, edge, threshold, 255, cv::THRESH_BINARY);
    //cv::imshow("edgeimg",edge);
    if(orientation==SCAN_DIRECTION_UP){
        for(int i=radius+1;i<edge.cols-radius-1;++i){
            if(startline<=0)
                return -1;
            cv::Point seed(i,startline);
            edge.at<uchar>(seed.y,seed.x)=0;
            cv::Point contourpoint(i,seed.y-1);
            while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y-=1;
                if(contourpoint.y==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(orientation==SCAN_DIRECTION_RIGHT){
        for(int i=radius+1;i<edge.rows-radius-1;++i){
            if(startline<0)
                return -1;
            cv::Point seed(startline,i);
            edge.at<uchar>(seed.y,seed.x)=0;
            cv::Point contourpoint(seed.x+1,i);
            while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x+=1;
                if(contourpoint.x>=edge.cols)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(orientation==SCAN_DIRECTION_DOWN){
        for(int i=radius+1;i<edge.cols-radius-1;++i){
            if(startline<0)
                return -1;
            cv::Point seed(i,startline);
            edge.at<uchar>(seed.y,seed.x)=0;
            cv::Point contourpoint(i,seed.y+1);
            while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y+=1;
                if(contourpoint.y>=edge.rows)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(orientation==SCAN_DIRECTION_LEFT){
        for(int i=radius+1;i<edge.rows-radius-1;++i){
            if(startline<=0)
                return -1;
            cv::Point seed(startline,i);
            edge.at<uchar>(seed.y,seed.x)=0;
            cv::Point contourpoint(seed.x-1,i);
            while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x-=1;
                if(contourpoint.x==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    contours=GetSegments(contours,segments);
    return 0;
}

vector<cv::Point> RectExtract::GetSegments(vector<cv::Point> contours, vector<int> segments){
    int segnum=segments.size()/2;
    vector<cv::Point> segcontour;
    for(int i=0;i<segnum;++i){
        for(int j=segments[i*2];j<segments[i*2+1];++j){
            segcontour.push_back(contours[j]);
        }
    }
    return segcontour;
}

int RectExtract::KirschEdge(cv::Mat srcimg,int threshold, vector<cv::Point>& contours, vector<vector<cv::Point>>& linecontours, vector<int> segment){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    int radius=5;
    float percent=0.8;
    int radthresh=0;
    cv::Mat point=srcimg.clone();
    cv::Mat edge = krisch(srcimg);
    cv::threshold(edge, edge, threshold, 255, cv::THRESH_BINARY);
    cv::imshow("edgethreshrect.jpg",edge);
    cv::imwrite("edgethreshrect.jpg",edge);
    cv::Point center(edge.cols/2,edge.rows/2);
    edge.at<uchar>(edge.rows/2,edge.cols/2)=0;
    FindEdge(edge, center, contours, radius,percent,radthresh);
    linecontours.clear();
    for(int i=0;i<segment.size()/2;++i){
        vector<cv::Point> segmentedge=ContoursCut(contours, segment[i*2+0], segment[i*2+1]);
        linecontours.push_back(segmentedge);
    }
    for(int i=0;i<linecontours.size();++i){
        for(int j=0;j<linecontours[i].size();++j){
            cv::circle(point,linecontours[i][j],1,255,-1);
        }
    }
    cv::imshow("dst",point);
    cv::imwrite("rectedge.jpg",point);
    return 0;
}

vector<vector<cv::Point>> RectExtract::GetFourEdges(vector<cv::Point> contours,int angle){
    vector<cv::Point> topedge=ContoursCut(contours, angle+10, 180-angle-10);
    vector<cv::Point> leftedge=ContoursCut(contours, 180-angle+10, 180+angle-10);
    vector<cv::Point> bottomedge=ContoursCut(contours, 180+angle+10, 360-angle-10);
    vector<cv::Point> rightedge=ContoursCut(contours, 360-angle+10, angle-10);
    vector<vector<cv::Point>> edgecontours;
    edgecontours.push_back(topedge);
    edgecontours.push_back(leftedge);
    edgecontours.push_back(bottomedge);
    edgecontours.push_back(rightedge);
    return edgecontours;
}

int RectExtract::FitLineAndDraw(cv::Mat srcimg, cv::Mat& dstimg, vector<vector<cv::Point>> linecontours){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    dstimg=srcimg.clone();
    for(int i=0;i<linecontours.size();++i){
        cv::Vec2f linepara;
        if(LineFit(linecontours[i],linepara)==0){
            for(int j=0;j<srcimg.cols;++j){
                cv::circle(dstimg,cv::Point(j,j*linepara[0]+linepara[1]),1,255,-1);
            }
            for(int j=0;j<srcimg.rows;++j){
                cv::circle(dstimg,cv::Point((j-linepara[1])/linepara[0],j),1,255,-1);
            }
        }
        else{
            for(int j=0;j<srcimg.rows;++j){
                cv::circle(dstimg,cv::Point(linecontours[i][0].x,j),1,255,-1);
            }
        }
    }
    //cv::imshow("lines",dstimg);
    return 0;
}

void RectExtract::FitLineAndGetDist(vector<vector<cv::Point>> linecontours,vector<int> coordinates, vector<float>& dist){
    dist.clear();
    if(linecontours.size()==4){
        vector<float> toplinepara,leftlinepara,bottomlinepara,rightlinepara;
        LineFitLeastSquares(linecontours[0],toplinepara);
        for(int i=0;i<linecontours[1].size();++i){
            cv::Point temp=linecontours[1][i];
            linecontours[1][i].x=temp.y;
            linecontours[1][i].y=temp.x;
        }
        LineFitLeastSquares(linecontours[1],leftlinepara);
        LineFitLeastSquares(linecontours[2],bottomlinepara);
        for(int i=0;i<linecontours[3].size();++i){
            cv::Point temp=linecontours[3][i];
            linecontours[3][i].x=temp.y;
            linecontours[3][i].y=temp.x;
        }
        LineFitLeastSquares(linecontours[3],rightlinepara);
        float sum=0;
        for(int i=coordinates[0];i<coordinates[1];++i){
            sum+=fabs((i*toplinepara[0]+toplinepara[1])-(i*bottomlinepara[0]+bottomlinepara[1]));
        }
        dist.push_back(sum/(coordinates[1]-coordinates[0]));
        sum=0;
        for(int i=coordinates[2];i<coordinates[3];++i){
            sum+=fabs((i*leftlinepara[0]+leftlinepara[1])-(i*rightlinepara[0]+rightlinepara[1]));
        }
        dist.push_back(sum/(coordinates[3]-coordinates[2]));
    }
    if(linecontours.size()==2&&coordinates.size()==2){
        vector<float> leftlinepara,rightlinepara;
        for(int i=0;i<linecontours[0].size();++i){
            cv::Point temp=linecontours[0][i];
            linecontours[0][i].x=temp.y;
            linecontours[0][i].y=temp.x;
        }
        LineFitLeastSquares(linecontours[0],leftlinepara);
        for(int i=0;i<linecontours[1].size();++i){
            cv::Point temp=linecontours[1][i];
            linecontours[1][i].x=temp.y;
            linecontours[1][i].y=temp.x;
        }
        LineFitLeastSquares(linecontours[1],rightlinepara);
        float sum=0;
        for(int i=coordinates[0];i<coordinates[1];++i){
            sum+=fabs((i*leftlinepara[0]+leftlinepara[1])-(i*rightlinepara[0]+rightlinepara[1]));
        }
        dist.push_back(sum/(coordinates[1]-coordinates[0]));
    }
    if(linecontours.size()==2&&coordinates.size()==4){
        vector<float> bottomlinepara;
        LineFitLeastSquares(linecontours[0],bottomlinepara);
        float sum=0;
        for(int i=coordinates[0];i<coordinates[1];++i){
            sum+=fabs(i*bottomlinepara[0]+bottomlinepara[1]);
        }
        dist.push_back(sum/(coordinates[1]-coordinates[0]));
        vector<float> toplinepara;
        LineFitLeastSquares(linecontours[1],toplinepara);
        sum=0;
        for(int i=coordinates[0];i<coordinates[1];++i){
            sum+=fabs(i*toplinepara[0]+toplinepara[1]);
        }
        dist.push_back(sum/(coordinates[1]-coordinates[0]));
    }
}

int RectExtract::GetRectEdge(cv::Mat srcimg,int threshold,int num, vector<float>& dists){
    dists.clear();
    vector<cv::Point> contours;
    vector<vector<cv::Point>> edgecontours;
    if(num==1){
        vector<int> segments{40,140,160,200,220,320,340,20};
        vector<int> distseg{220,810,160,480};
        KirschEdge(srcimg,threshold,contours,edgecontours, segments);
        FitLineAndGetDist(edgecontours,distseg,dists);
    }
    else if(num==2){
        vector<int> segments{45,135,155,180,225,315,345,25};
        vector<int> distseg{130,420,100,270};
        KirschEdge(srcimg,threshold,contours,edgecontours, segments);
        vector<vector<cv::Point>> topbottomcontours;
        topbottomcontours.push_back(edgecontours[0]);
        topbottomcontours.push_back(edgecontours[2]);
        float ypos=GetMidLineY(topbottomcontours, srcimg.cols/2);
        FitLineAndGetDist(edgecontours,distseg,dists);
        float xpos=GetLineX(edgecontours[1],srcimg.rows/2);
        dists.push_back(ypos);
        dists.push_back(xpos);
    }
    else if(num==3){
        cv::Mat edge = krisch(srcimg.clone());
        cv::threshold(edge, edge, threshold, 255, cv::THRESH_BINARY);
        int startline=srcimg.cols/2;
        for(int i=5+1;i<edge.rows-5-1;++i){
            cv::Point seed(startline,i);
            cv::Point contourpoint(seed.x-1,i);
            while(!PointDense(edge,contourpoint,5,0.8)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x-=1;
                if(contourpoint.x==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
        contours=GetSegments(contours,vector<int>{80,180});
        edgecontours.push_back(contours);
        for(int i=0;i<contours.size();++i){
            cv::circle(srcimg,contours[i],1,255,-1);
        }
        ScanRightLine(srcimg,contours);
        edgecontours.push_back(contours);
        for(int i=0;i<contours.size();++i){
            cv::circle(srcimg,contours[i],1,255,-1);
        }
        cv::imshow("11",srcimg);
        vector<int> distseg{110,250};
        FitLineAndGetDist(edgecontours,distseg,dists);
    }
    else if(num==4){
        vector<int> segments{264,276,84,96};
        vector<int> distseg{55,85,55,85};
        KirschEdge(srcimg,threshold,contours,edgecontours, segments);
        FitLineAndGetDist(edgecontours,distseg,dists);
    }
    else if(num==5){
        vector<int> segments{262,274,86,98};
        vector<int> distseg{40,70,40,70};
        KirschEdge(srcimg,threshold,contours,edgecontours, segments);
        FitLineAndGetDist(edgecontours,distseg,dists);
    }
}

bool RectExtract::PixelDense(cv::Mat srcimg, int thresh){
    int posnum=0;
    for(int i=0;i<srcimg.rows;++i){
        for(int j=0;j<srcimg.cols;++j){
            if(srcimg.at<uchar>(i,j)<thresh){
                posnum++;
            }
        }
    }
    if(posnum>=(srcimg.rows*srcimg.cols*0.45)){
        return true;
    }
    else{
        return false;
    }
}

void RectExtract::ScanRightLine(cv::Mat srcimg, vector<cv::Point>& contours){
    contours.clear();
    for(int i=110; i<250; i=i+2){
        for(int j=srcimg.cols-4;j>srcimg.cols-50;--j){
            cv::Mat roi=srcimg(cv::Rect(j-3,i-3,7,7));
            if(cv::mean(roi)[0]>60.0){
                continue;}
            else{
                if(PixelDense(roi,40)){
                    contours.push_back(cv::Point(j,i));
                    break;
                }
            }
        }
    }
}



float RectExtract::CalibrationImgOri(cv::Mat srcimg, int orientation, int threshold, int radius, float percent){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat edge = krisch(srcimg);
    cv::imshow("dst",edge);
    cv::threshold(edge, edge, threshold, 255, cv::THRESH_BINARY);
    vector<cv::Point> linecontour;
    //ScanRect(edge,orientation,linecontour,radius,percent);
    for(int i=0;i<linecontour.size();++i){
        cv::circle(srcimg,linecontour[i],1,255,-1);
    }
    cv::imshow("linepoints",srcimg);
    cv::Vec2f linepara;
    int rtn=LineFit(linecontour,linepara);
    if(rtn==-1)
        return 90.0;
    cout<<linepara<<endl;
    float angle=atan(linepara[0])*180.0/PI;
    return angle;
}

void RectExtract::RotateImg(cv::Mat srcimg, cv::Mat& dstimg, float angle){
    cv::Point center = cv::Point(srcimg.cols / 2, srcimg.rows / 2);//旋转中心
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);//获得仿射变换矩阵
    cv::Size dst_sz(srcimg.cols, srcimg.rows);
    cv::warpAffine(srcimg, dstimg, rot_mat, dst_sz);
}

int RectExtract::LineFit(vector<cv::Point> contours,cv::Vec2f& linepara){
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
    if(dx==0||fabs(k1)>100){
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

void RectExtract::LineFitLeastSquares(vector<cv::Point> contours, vector<float> &vResult)
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

void RectExtract::LineFitLeastSquaresf(vector<cv::Point2f> contours, vector<float> &vResult)
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

int RectExtract::KirschEdgeCircle(cv::Mat srcimg,int threshold, int startangle, int endangle, vector<cv::Point>& contours){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat edge = krisch(srcimg);
    int radius=5;
    float percent=0.8;
    int radthresh=0;
    cv::threshold(edge, edge, threshold, 255, cv::THRESH_BINARY);
    cv::imshow("edge",edge);
    cv::imwrite("edgethresh.jpg",edge);
    cv::Point center(edge.cols/2,edge.rows/2);
    edge.at<uchar>(center.y,center.x)=0;
    FindEdge(edge, center, contours, radius,percent,radthresh);
    contours=ContoursCut(contours, startangle, endangle);
    for(int i=0;i<contours.size();++i){
        cv::circle(srcimg,contours[i],1,255,-1);
    }
    cv::imshow("edgepoint",srcimg);
    cv::imwrite("edgepoint.jpg",srcimg);
    return 0;
}

int RectExtract::LDRCircleEdge(cv::Mat srcimg, int threshold, int num, vector<cv::Point>& contours){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat edge = krisch(srcimg);
    cv::threshold(edge, edge, threshold, 255, cv::THRESH_BINARY);
    cv::imshow("edge",edge);
    if(num==1){
        int radius=4;
        float percent=0.8;
        cv::Point center(158,232);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 10, 120);
    }
    else if(num==2){
        int radius=4;
        float percent=0.8;
        cv::Point center(142,184);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 325, 75);
    }
    else if(num==3){
        int radius=4;
        float percent=0.8;
        cv::Point center(110,154);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 280, 30);
    }
    else if(num==4){
        int radius=4;
        float percent=0.8;
        cv::Point center(166,128);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 240, 350);
    }
    else if(num==5){
        int radius=4;
        float percent=0.8;
        cv::Point center(194,150);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 190, 300);
    }
    else if(num==6){
        int radius=4;
        float percent=0.8;
        cv::Point center(220,148);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 150, 260);
    }
    else if(num==7){
        int radius=4;
        float percent=0.8;
        cv::Point center(240,188);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 100, 210);
    }
    else if(num==8){
        int radius=4;
        float percent=0.8;
        cv::Point center(202,220);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 60, 170);
    }
    else if(num==9){
        int radius=4;
        float percent=0.8;
        cv::Point center(110,260);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 350, 100);
    }
    else if(num==10){
        int radius=4;
        float percent=0.8;
        cv::Point center(104,104);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 260, 10);
    }
    else if(num==11){
        int radius=4;
        float percent=0.8;
        cv::Point center(254,114);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 170, 280);
    }
    else if(num==12){
        int radius=4;
        float percent=0.8;
        cv::Point center(260,254);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 80, 190);
    }
    else if(num==13){
        int radius=4;
        float percent=0.8;
        cv::Point center(178,250);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 10, 120);
    }
    else if(num==14){
        int radius=4;
        float percent=0.8;
        cv::Point center(124,200);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 335, 85);
    }
    else if(num==15){
        int radius=4;
        float percent=0.8;
        cv::Point center(136,160);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 280, 30);
    }
    else if(num==16){
        int radius=4;
        float percent=0.8;
        cv::Point center(154,150);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 245, 355);
    }
    else if(num==17){
        int radius=4;
        float percent=0.8;
        cv::Point center(200,166);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 185, 295);
    }
    else if(num==18){
        int radius=4;
        float percent=0.8;
        cv::Point center(196,152);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 150, 260);
    }
    else if(num==19){
        int radius=4;
        float percent=0.8;
        cv::Point center(190,196);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 100, 210);
    }
    else if(num==20){
        int radius=4;
        float percent=0.8;
        cv::Point center(190,222);
        edge.at<uchar>(center.y,center.x)=0;
        FindEdge(edge, center, contours, radius, percent, 0);
        contours=ContoursCut(contours, 65, 175);
    }
    for(int i=0;i<contours.size();++i){
        cv::circle(srcimg,contours[i],1,255,-1);
    }
//    if(contours.size()!=50){
//        return -2;
//    }
    cout<<contours.size()<<endl;
    cv::imshow("edgepoint",srcimg);
    return 0;
}

int RectExtract::KirschEdgeSmallCircle(cv::Mat srcimg,int threshold,int num, vector<cv::Point>& contours){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat edge = krisch(srcimg);
    cv::threshold(edge, edge, threshold, 255, cv::THRESH_BINARY);
    cv::imshow("edge",edge);
    cv::imwrite("edgethresh.jpg",edge);
    cv::Point center(edge.cols/2,edge.rows/2);
    edge.at<uchar>(edge.rows/2,edge.cols/2)=0;
    if(num==1){
        FindEdge(edge, center, contours, 5,0.6,50);
        contours=ContoursCut(contours, 0, 360);
    }
    else if(num==2){
        FindEdge(edge, center, contours, 4,0.8,50);
        contours=ContoursCut(contours, 0, 360);
    }
    else if(num==3){
        FindEdge(edge, center, contours, 4,0.8,50);
        contours=ContoursCut(contours, 0, 360);
    }
    else if(num==4){
        FindEdge(edge, center, contours, 4,0.8,50);
        contours=ContoursCut(contours, 0, 360);
    }
    else if(num==5){
        FindEdge(edge, center, contours, 4,0.6,50);
        contours=ContoursCut(contours, 0, 360);
    }
    else if(num==6){
        FindEdge(edge, center, contours, 4,0.6,50);
        contours=ContoursCut(contours, 0, 360);
    }
    else if(num==7){
        FindEdge(edge, center, contours, 4,0.6,50);
        contours=ContoursCut(contours, 0, 360);
    }
    return 0;
}

int RectExtract::GetSmallCircle(cv::Mat srcimg,int threshold, int num, vector<float>& result){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    vector<cv::Point> contours;
    KirschEdgeSmallCircle(srcimg,threshold,num,contours);
    result=FitCircle(contours);
    return 0;
}

vector<float> RectExtract::FitCircle(vector<cv::Point>& contours){
    float a=0,b=0,r=0;
    double sumX,sumY,sumR;
    for(int i=0;i<contours.size();++i){
        sumX+=contours[i].x;
        sumY+=contours[i].y;
    }
    a=sumX/contours.size();
    b=sumY/contours.size();
    for(int i=0;i<contours.size();++i){
        sumR+=sqrt((contours[i].x-a)*(contours[i].x-a)+(contours[i].y-b)*(contours[i].y-b));
    }
    r=sumR/contours.size();
    return vector<float>{a,b,r};
}

void RectExtract::GetCircle(vector<cv::Point> contours, cv::Point center,int threshold,cv::Vec3f& circlepara){
    vector<double> sumvec;
    vector<cv::Point> pointvec;
    for(int i=center.x-threshold;i<center.x+threshold;++i){
        for(int j=center.y-threshold;j<center.y+threshold;++j){
            sumvec.push_back(CalcDiff(contours,cv::Point(i,j)));
            pointvec.push_back(cv::Point(i,j));
        }
    }
    vector<double>::iterator smallest = min_element(begin(sumvec), end(sumvec));
    int index=distance(std::begin(sumvec), smallest);
    circlepara[0]=pointvec[index].x;
    circlepara[1]=pointvec[index].y;
    circlepara[2]=sqrt((pointvec[index].x-contours[contours.size()/2].x)*(pointvec[index].x-contours[contours.size()/2].x)+(pointvec[index].y-contours[contours.size()/2].y)*(pointvec[index].y-contours[contours.size()/2].y));
}

cv::Point RectExtract::GetTanPoint(int a, int b, int R, float k, int flag){
    cv::Point tanpoint(0,0);
    if(flag==0){
        tanpoint.y=b-R/sqrt(1+k*k);
        tanpoint.x=a+k*R/sqrt(1+k*k);
    }
    else if(flag==1){
        tanpoint.y=b+R/sqrt(1+k*k);
        tanpoint.x=a-k*R/sqrt(1+k*k);
    }
    return tanpoint;
}

double RectExtract::CalcDiff(vector<cv::Point> contours,cv::Point center){
    float rad=(contours[contours.size()/2].x-center.x)*(contours[contours.size()/2].x-center.x)+(contours[contours.size()/2].y-center.y)*(contours[contours.size()/2].y-center.y);
    double sum=0;
    for(int i=0;i<contours.size();++i){
        float dist=fabs((contours[i].x-center.x)*(contours[i].x-center.x)+(contours[i].y-center.y)*(contours[i].y-center.y)-rad);
        sum+=dist;
    }
    return sum;
}

int RectExtract::PrewittEdge(cv::Mat srcimg,int threshold, vector<cv::Point>& contours){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    //图像矩阵和 prewitt_x卷积核的卷积
    cv::Mat img_prewitt_x;
    prewitt(srcimg, img_prewitt_x,CV_32FC1,1, 0);
    //图像矩阵与prewitt_y卷积核卷积
    cv::Mat img_prewitt_y;
    prewitt(srcimg, img_prewitt_y, CV_32FC1, 0, 1);
    cv::Mat edge;
    /*第三步:水平方向和垂直方向的边缘强度*/
    //数据类型转换,边缘强度的灰度级显示
    cv::Mat abs_img_prewitt_x, abs_img_prewitt_y;
    cv::convertScaleAbs(img_prewitt_x, abs_img_prewitt_x, 1, 0);
    cv::convertScaleAbs(img_prewitt_y, abs_img_prewitt_y, 1, 0);
    //cv::imshow("垂直方向的边缘", abs_img_prewitt_x);
    //imwrite("img1_v_edge.jpg", abs_img_prewitt_x);
    //cv::imshow("水平方向的边缘", abs_img_prewitt_y);
    //imwrite("img1_h_edge.jpg", abs_img_prewitt_y);
    /*第四步：通过第三步得到的两个方向的边缘强度,求出最终的边缘强度*/
    //这里采用平方根的方式
    cv::Mat img_prewitt_x2, image_prewitt_y2;
    cv::pow(img_prewitt_x,2.0,img_prewitt_x2);
    cv::pow(img_prewitt_y,2.0,image_prewitt_y2);
    cv::sqrt(img_prewitt_x2 + image_prewitt_y2, edge);
    //数据类型转换,边缘的强度灰度级显示
    edge.convertTo(edge, CV_8UC1);
    cv::imshow("边缘强度",edge);
    //imwrite("img1_edge.jpg", edge);
    cv::threshold(edge, edge, threshold, 255, cv::THRESH_BINARY);
    cv::imshow("thresholdedge",edge);
    //findRect(edge, contours);
    return 0;
}

/*可分离的离散二维卷积,先水平方向的卷积,然后接着进行垂直方向的卷积*/
void RectExtract::sepConv2D_X_Y(cv::InputArray src, cv::OutputArray src_kerX_kerY, int ddepth, cv::InputArray kernelX, cv::InputArray kernelY, cv::Point anchor, int borderType)
{
    //输入矩阵与水平方向卷积核的卷积
    cv::Mat src_kerX;
    conv2D(src, kernelX, src_kerX, ddepth, anchor, borderType);
    //上面得到的卷积结果，然后接着和垂直方向的卷积核卷积，得到最终的输出
    conv2D(src_kerX, kernelY, src_kerX_kerY, ddepth, anchor, borderType);
}
/*可分离的离散二维卷积,先垂直方向的卷积，然后进行水平方向的卷积*/
void RectExtract::sepConv2D_Y_X(cv::InputArray src, cv::OutputArray src_kerY_kerX, int ddepth, cv::InputArray kernelY, cv::InputArray kernelX, cv::Point anchor, int borderType)
{
    //输入矩阵与垂直方向卷积核的卷积
    cv::Mat src_kerY;
    conv2D(src, kernelY, src_kerY, ddepth, anchor, borderType);
    //上面得到的卷积结果，然后接着和水平方向的卷积核卷积，得到最终的输出
    conv2D(src_kerY, kernelX, src_kerY_kerX, ddepth, anchor, borderType);
}

/*
    prewitt卷积运算
*/
void RectExtract::prewitt(cv::InputArray src,cv::OutputArray dst, int ddepth,int x, int y, int borderType)
{
    CV_Assert(!(x == 0 && y == 0));
    //如果 x!=0，src 和 prewitt_x卷积核进行卷积运算
    if (x != 0)
    {
        //可分离的prewitt_x卷积核
        cv::Mat prewitt_x_y = (cv::Mat_<float>(3, 1) << 1, 1, 1);
        cv::Mat prewitt_x_x = (cv::Mat_<float>(1, 3) << 1, 0, -1);
        //可分离的离散的二维卷积
        sepConv2D_Y_X(src, dst, ddepth, prewitt_x_y, prewitt_x_x, cv::Point(-1, -1), borderType);
    }
    if (y != 0)
    {
        //可分离的prewitt_y卷积核
        cv::Mat prewitt_y_x = (cv::Mat_<float>(1, 3) << 1, 1, 1);
        cv::Mat prewitt_y_y = (cv::Mat_<float>(3, 1) << 1, 0, -1);
        //可分离的离散二维卷积
        sepConv2D_X_Y(src, dst, ddepth, prewitt_y_x, prewitt_y_y, cv::Point(-1, -1), borderType);
    }
}

int RectExtract::SobelEdge(cv::Mat srcimg,int threshold, vector<cv::Point>& contours){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    /* --- sobel 边缘检测 --- */
    //与水平方向的 sobel 核卷积
    cv::Mat image_Y_X = sobel(srcimg, 1, 0, 3, 4);
    //垂直方向的边缘强度
    cv::Mat imageYX_abs = abs(image_Y_X);
    //垂直方向边缘强度的灰度级显示
    cv::Mat imageYX_gray;
    imageYX_abs.convertTo(imageYX_gray, CV_8UC1, 1.0, 0);
    //cv::imshow("垂直方向的边缘强度", imageYX_gray);
    //与垂直方向的 sobel 核卷积
    cv::Mat image_X_Y = sobel(srcimg, 0, 1, 3, 4);
    //水平方向的边缘强度
    cv::Mat imageXY_abs = abs(image_X_Y);
    //水平方向边缘强度的灰度级显示
    cv::Mat imageXY_gray;
    imageXY_abs.convertTo(imageXY_gray, CV_8UC1, 1.0, 0);
    //cv::imshow("水平方向的边缘强度", imageXY_gray);
    //根据垂直方向和水平方向边缘强度的平方和，得到最终的边缘强度
    cv::Mat edge;
    cv::magnitude(image_Y_X, image_X_Y, edge);
    //边缘强度的灰度级显示
    edge.convertTo(edge, CV_8UC1, 1.0, 0);
    cv::threshold(edge,edge,threshold,255,cv::THRESH_BINARY);
    cv::imshow("边缘强度",edge);
    //findRect(edge, contours);
    return 0;
}

//计算阶乘
int RectExtract::factorial(int n)
{
    int fac = 1;
    // 0 的阶乘等于 1
    if (n == 0)
        return fac;
    for (int i = 1; i <= n; i++)
        fac *= i;
    return fac;
}
//计算平滑系数
cv::Mat RectExtract::getPascalSmooth(int n)
{
    cv::Mat pascalSmooth = cv::Mat::zeros(cv::Size(n, 1), CV_32FC1);
    for (int i = 0; i < n; i++)
        pascalSmooth.at<float>(0, i) = factorial(n - 1) / (factorial(i) * factorial(n - 1 - i));
    return pascalSmooth;
}
//计算差分
cv::Mat RectExtract::getPascalDiff(int n)
{
    cv::Mat pascalDiff = cv::Mat::zeros(cv::Size(n, 1), CV_32FC1);
    cv::Mat pascalSmooth_previous = getPascalSmooth(n - 1);
    for (int i = 0; i<n; i++)
    {
        if (i == 0)
            pascalDiff.at<float>(0, i) = 1;
        else if (i == n - 1)
            pascalDiff.at<float>(0, i) = -1;
        else
            pascalDiff.at<float>(0, i) = pascalSmooth_previous.at<float>(0, i) - pascalSmooth_previous.at<float>(0, i - 1);
    }
    return pascalDiff;
}

// sobel 边缘检测
cv::Mat RectExtract::sobel(cv::Mat image, int x_flag, int y_flag, int winSize, int borderType)
{
    // sobel 卷积核的窗口大小为大于 3 的奇数
    CV_Assert(winSize >= 3 && winSize % 2 == 1);
    //平滑系数
    cv::Mat pascalSmooth = getPascalSmooth(winSize);
    //差分系数
    cv::Mat pascalDiff = getPascalDiff(winSize);
    cv::Mat image_con_sobel;
    /* 当 x_falg != 0 时，返回图像与水平方向的 Sobel 核的卷积*/
    if (x_flag != 0)
    {
        //根据可分离卷积核的性质
        //先进行一维垂直方向上的平滑，再进行一维水平方向的差分
        sepConv2D_Y_X(image, image_con_sobel, CV_32FC1, pascalSmooth.t(), pascalDiff, cv::Point(-1, -1), borderType);
    }
    /* 当 x_falg == 0 且 y_flag != 0 时，返回图像与垂直 方向的 Sobel 核的卷积*/
    if (x_flag == 0 && y_flag != 0)
    {
        //根据可分离卷积核的性质
        //先进行一维水平方向上的平滑，再进行一维垂直方向的差分
        sepConv2D_X_Y(image, image_con_sobel, CV_32FC1, pascalSmooth, pascalDiff.t(), cv::Point(-1, -1), borderType);
    }
    return image_con_sobel;
}

//计算与两点构成的直线距离为d的两条直线上的点
int RectExtract::calcpoints(int x1,int y1,int x2,int y2,int d,vector<cv::Point>& uppoints,vector<cv::Point>& downpoints){
    uppoints.clear();
    downpoints.clear();
    if(x1==x2&&y1==y2){return -1;}
    if(x1==x2){
        if(y1>y2){
            int tx=x2;
            int ty=y2;
            x2=x1;y2=y1;
            x1=tx;y1=ty;
        }
        for(int i=y1;i<=y2;i+=1){
            uppoints.push_back(cv::Point(x1-d,i));
            downpoints.push_back(cv::Point(x1+d,i));
        }
    }
    if(x1>x2){
        int tx=x2;int ty=y2;
        x2=x1;y2=y1;
        x1=tx;y1=ty;
    }
    if(y1==y2){
        for(int i=x1;i<=x2;i+=1){
            uppoints.push_back(cv::Point(i,y1+d));
            downpoints.push_back(cv::Point(i,y1-d));
        }
    }
    else if(y1!=y2&&x1!=x2){
        float k=(y2-y1)/(x2-x1);
        float invk=-1/k;
        float deltaX=sqrt(d*d/(1+invk*invk));
        cv::Point Upt1,Upt2,Dpt1,Dpt2;
        if(invk<0){
            Upt1.x=x1-deltaX;
            Upt1.y=y1-deltaX*invk;
            Upt2.x=x2-deltaX;
            Upt2.y=y2-deltaX*invk;
            Dpt1.x=x1+deltaX;
            Dpt1.y=y1+deltaX*invk;
            Dpt2.x=x2+deltaX;
            Dpt2.y=y2+deltaX*invk;
            if(invk<=-1){
                for(int i=Upt1.x;i<=Upt2.x;i+=1){
                    uppoints.push_back(cv::Point(i,(i-Upt1.x)*k+Upt1.y));
                }
                for(int i=Dpt1.x;i<=Dpt2.x;i+=1){
                    downpoints.push_back(cv::Point(i,(i-Dpt1.x)*k+Dpt1.y));
                }
            }
            else{
                for(int i=Upt1.y;i<=Upt2.y;i+=1){
                    uppoints.push_back(cv::Point((i-Upt1.y)/k+Upt1.x,i));
                }
                for(int i=Dpt1.y;i<=Dpt2.y;i+=1){
                    downpoints.push_back(cv::Point((i-Dpt1.y)/k+Dpt1.x,i));
                }
            }
        }
        else{
            Upt1.x=x1+deltaX;
            Upt1.y=y1+deltaX*invk;
            Upt2.x=x2+deltaX;
            Upt2.y=y2+deltaX*invk;
            Dpt1.x=x1-deltaX;
            Dpt1.y=y1-deltaX*invk;
            Dpt2.x=x2-deltaX;
            Dpt2.y=y2-deltaX*invk;
            if(invk>=1){
                for(int i=Upt1.x;i<=Upt2.x;i+=1){
                    uppoints.push_back(cv::Point(i,(i-Upt1.x)*k+Upt1.y));
                }
                for(int i=Dpt1.x;i<=Dpt2.x;i+=1){
                    downpoints.push_back(cv::Point(i,(i-Dpt1.x)*k+Dpt1.y));
                }
            }
            else{
                for(int i=Upt2.y;i<=Upt1.y;i+=1){
                    uppoints.push_back(cv::Point((i-Upt2.y)/k+Upt2.x,i));
                }
                for(int i=Dpt2.y;i<=Dpt1.y;i+=1){
                    downpoints.push_back(cv::Point((i-Dpt2.y)/k+Dpt2.x,i));
                }
            }
        }
    }
    return 0;
}

int RectExtract::FindEdge(cv::Mat srcimg, cv::Point seed, vector<cv::Point>& contours, int radius, float percent, int radthresh){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    contours.clear();
    cv::Point center=seed;
    float theta;
    for(int i=0;i<180;++i){
        theta=i;
        float k=tan(theta/180.0*PI);
        float b=center.y-k*center.x;
        if(theta>=0&&theta<=45){
            cv::Point contourPoint1(center.x+1+radthresh*fabs(cos(theta/180.0*PI)),(center.x+1+radthresh*fabs(cos(theta/180.0*PI)))*k+b);
            cv::Point contourPoint2(center.x-1-radthresh*fabs(cos(theta/180.0*PI)),(center.x-1-radthresh*fabs(cos(theta/180.0*PI)))*k+b);
            while(!PointDense(srcimg,contourPoint1,radius,percent)||abs(srcimg.at<uchar>(contourPoint1.y,contourPoint1.x)-srcimg.at<uchar>(center.y,center.x))<15){
                contourPoint1.x=contourPoint1.x+1;
                contourPoint1.y=contourPoint1.x*k+b;
                if(contourPoint1.y>=srcimg.rows||contourPoint1.x>=srcimg.cols||contourPoint1.y<0||contourPoint1.x<0)
                    break;
            }
            contours.push_back(cv::Point(contourPoint1.x,(contourPoint1.x)*k+b));
            while(!PointDense(srcimg,contourPoint2,radius,percent)||abs(srcimg.at<uchar>(contourPoint2.y,contourPoint2.x)-srcimg.at<uchar>(center.y,center.x))<15){
                contourPoint2.x=contourPoint2.x-1;
                contourPoint2.y=contourPoint2.x*k+b;
                if(contourPoint2.y>=srcimg.rows||contourPoint2.x>=srcimg.cols||contourPoint2.y<0||contourPoint2.x<0)
                    break;
            }
            contours.push_back(cv::Point(contourPoint2.x,(contourPoint2.x)*k+b));
        }
        else if(theta>=135&&theta<=180){
            cv::Point contourPoint1(center.x-1-radthresh*fabs(cos(theta/180.0*PI)),(center.x-1-radthresh*fabs(cos(theta/180.0*PI)))*k+b);
            cv::Point contourPoint2(center.x+1+radthresh*fabs(cos(theta/180.0*PI)),(center.x+1+radthresh*fabs(cos(theta/180.0*PI)))*k+b);
            while(!PointDense(srcimg,contourPoint1,radius,percent)||abs(srcimg.at<uchar>(contourPoint1.y,contourPoint1.x)-srcimg.at<uchar>(center.y,center.x))<15){
                contourPoint1.x=contourPoint1.x-1;
                contourPoint1.y=contourPoint1.x*k+b;
                if(contourPoint1.y>=srcimg.rows||contourPoint1.x>=srcimg.cols||contourPoint1.y<0||contourPoint1.x<0)
                    break;
            }
            contours.push_back(cv::Point(contourPoint1.x,(contourPoint1.x)*k+b));
            while(!PointDense(srcimg,contourPoint2,radius,percent)||abs(srcimg.at<uchar>(contourPoint2.y,contourPoint2.x)-srcimg.at<uchar>(center.y,center.x))<15){
                contourPoint2.x=contourPoint2.x+1;
                contourPoint2.y=contourPoint2.x*k+b;
                if(contourPoint2.y>=srcimg.rows||contourPoint2.x>=srcimg.cols||contourPoint2.y<0||contourPoint2.x<0)
                    break;
            }
            contours.push_back(cv::Point(contourPoint2.x,(contourPoint2.x)*k+b));
        }
        else{
            cv::Point contourPoint1((center.y+1+radthresh*fabs(sin(theta/180.0*PI))-b)/k,center.y+1+radthresh*fabs(sin(theta/180.0*PI)));
            cv::Point contourPoint2((center.y-1-radthresh*fabs(sin(theta/180.0*PI))-b)/k,center.y-1-radthresh*fabs(sin(theta/180.0*PI)));
            while(!PointDense(srcimg,contourPoint1,radius,percent)||abs(srcimg.at<uchar>(contourPoint1.y,contourPoint1.x)-srcimg.at<uchar>(center.y,center.x))<15){
                contourPoint1.y=contourPoint1.y+1;
                contourPoint1.x=(contourPoint1.y-b)/k;
                if(contourPoint1.y>=srcimg.rows||contourPoint1.x>=srcimg.cols||contourPoint1.y<0||contourPoint1.x<0)
                    break;
            }
            contours.push_back(cv::Point((contourPoint1.y-b)/k,contourPoint1.y));
            while(!PointDense(srcimg,contourPoint2,radius,percent)||abs(srcimg.at<uchar>(contourPoint2.y,contourPoint2.x)-srcimg.at<uchar>(center.y,center.x))<15){
                contourPoint2.y=contourPoint2.y-1;
                contourPoint2.x=(contourPoint2.y-b)/k;
                if(contourPoint2.y>=srcimg.rows||contourPoint2.x>=srcimg.cols||contourPoint2.y<0||contourPoint2.x<0)
                    break;
            }
            contours.push_back(cv::Point((contourPoint2.y-b)/k,contourPoint2.y));
        }
    }
    return 0;
}

vector<cv::Point> RectExtract::ContoursCut(vector<cv::Point> contours,int startangle,int endangle){
    vector<cv::Point> segcontours;
    if(startangle>endangle){
        for(int i=startangle;i<360;++i){
            if(i<180&&i>=1){segcontours.push_back(contours[359-(i-1)*2]);}
            else if(i>180&&i<=360){segcontours.push_back(contours[359-2*(i-180)+1]);}
            else if(i==180){segcontours.push_back(contours[1]);}
            else{segcontours.push_back(contours[0]);}
        }
        for(int i=0;i<endangle;++i){
            if(i<180&&i>=1){segcontours.push_back(contours[359-(i-1)*2]);}
            else if(i>180&&i<=360){segcontours.push_back(contours[359-2*(i-180)+1]);}
            else if(i==180){segcontours.push_back(contours[1]);}
            else{segcontours.push_back(contours[0]);}
        }
    }
    else{
        for(int i=startangle;i<endangle;++i){
            if(i<180&&i>=1){segcontours.push_back(contours[359-(i-1)*2]);}
            else if(i>180&&i<=360){segcontours.push_back(contours[359-2*(i-180)+1]);}
            else if(i==180){segcontours.push_back(contours[1]);}
            else{segcontours.push_back(contours[0]);}
        }
    }
    return segcontours;
}

int RectExtract::ContourDense(cv::Mat srcimg,vector<cv::Point> contours,int startangle,int endangle, vector<cv::Point>& densecontours, int thickness){
    densecontours.clear();
    vector<cv::Point> segcontours;
    if(startangle>endangle){
        for(int i=startangle;i<360;++i){
            if(i<180&&i>=1){segcontours.push_back(contours[359-(i-1)*2]);}
            else if(i>180&&i<=360){segcontours.push_back(contours[359-2*(i-180)+1]);}
            else if(i==180){segcontours.push_back(contours[1]);}
            else{segcontours.push_back(contours[0]);}
        }
        for(int i=0;i<endangle;++i){
            if(i<180&&i>=1){segcontours.push_back(contours[359-(i-1)*2]);}
            else if(i>180&&i<=360){segcontours.push_back(contours[359-2*(i-180)+1]);}
            else if(i==180){segcontours.push_back(contours[1]);}
            else{segcontours.push_back(contours[0]);}
        }
    }
    else{
        for(int i=startangle;i<endangle;++i){
            if(i<180&&i>=1){segcontours.push_back(contours[359-(i-1)*2]);}
            else if(i>180&&i<=360){segcontours.push_back(contours[359-2*(i-180)+1]);}
            else if(i==180){segcontours.push_back(contours[1]);}
            else{segcontours.push_back(contours[0]);}
        }
    }
    cv::Mat contourimg=cv::Mat::zeros(srcimg.rows,srcimg.cols,CV_8UC1);
    vector<vector<cv::Point>> contourgroup;
    contourgroup.push_back(segcontours);
    //cv::drawContours(contourimg,contourgroup,0,255,3);
    for(int i=0;i<segcontours.size()-1;++i){
        cv::line(contourimg,segcontours[i],segcontours[i+1],255,thickness);
    }
    cv::imshow("contourss",contourimg);
    for(int i=0;i<srcimg.rows;++i){
        for(int j=0;j<srcimg.cols;++j){
            int a=(int) srcimg.at<uchar>(i,j);
            int b=(int) contourimg.at<uchar>(i,j);
            if(a==255&&b==255){
                densecontours.push_back(cv::Point(j,i));
            }
        }
    }
    return 0;
}

int RectExtract::PointDense(cv::Mat srcimg, cv::Point center, int rad, float thresh){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    if(center.x-rad-1<0||center.y-rad-1<0||center.x+rad+1>=srcimg.cols||center.y+rad+1>=srcimg.rows)
    {return 0;}
    cv::Mat maskimg=cv::Mat::zeros(rad*2+2,rad*2+2,CV_8UC1);
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

int RectExtract::ScanLine(cv::Mat srcimg, int lineori, vector<cv::Point>& contours,int startline, int radius, float percent){
    if(srcimg.empty()||srcimg.cols<radius*2+2||srcimg.rows<radius*2+2||startline<0)
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    if(lineori==SCAN_DIRECTION_UP){
        for(int i=radius+1;i<srcimg.cols-radius-1;++i){
            cv::Point seed(i,startline);
            cv::Point contourpoint(i,seed.y-1);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y-=1;
                if(contourpoint.y==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(lineori==SCAN_DIRECTION_RIGHT){
        for(int i=radius+1;i<srcimg.rows-radius-1;++i){
            cv::Point seed(startline,i);
            cv::Point contourpoint(seed.x+1,i);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x+=1;
                if(contourpoint.x>=srcimg.cols)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(lineori==SCAN_DIRECTION_DOWN){
        for(int i=radius+1;i<srcimg.cols-radius-1;++i){
            cv::Point seed(i,startline);
            cv::Point contourpoint(i,seed.y+1);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y+=1;
                if(contourpoint.y>=srcimg.rows)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(lineori==SCAN_DIRECTION_LEFT){
        for(int i=radius+1;i<srcimg.rows-radius-1;++i){
            cv::Point seed(startline,i);
            cv::Point contourpoint(seed.x-1,i);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x-=1;
                if(contourpoint.x==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    return 0;
}

int RectExtract::ConvImage5(cv::Mat srcimg, int threshold, vector<cv::Point>& contours){
    cv::Mat kernelWin=(cv::Mat_<float>(5, 5) << -2, -4, -4, -4, -2,
                       -4,0,8,0,-4,
                       -4,8,24,8,-4,
                       -4,0,8,0,-4,
                       -2, -4, -4, -4, -2);
    cv::Mat dstimg;
    cv::filter2D(srcimg, dstimg, CV_32FC1, kernelWin);
    //两个卷积结果的灰度级显示
    cv::convertScaleAbs(dstimg, dstimg, 1, 0);
    cv::Mat histimg=getHistograph(dstimg);
    cv::imshow("dst",dstimg);
    cv::imshow("Hist",histimg);
    cv::imwrite("histk5.jpg",histimg);
    cv::threshold(dstimg,dstimg,threshold,255,cv::THRESH_BINARY);
    cv::imshow("边缘强度",dstimg);
    cv::imwrite("circle1k5.jpg",dstimg);
    //findRect(dstimg, contours);
    return 0;
}


cv::Mat RectExtract::getHistograph(const cv::Mat grayImage)
{
    //定义求直方图的通道数目，从0开始索引
    int channels[]={0};
    //定义直方图的在每一维上的大小，例如灰度图直方图的横坐标是图像的灰度值，就一维，bin的个数
    //如果直方图图像横坐标bin个数为x，纵坐标bin个数为y，则channels[]={1,2}其直方图应该为三维的，Z轴是每个bin上统计的数目
    const int histSize[]={256};
    //每一维bin的变化范围
    float range[]={0,256};

    //所有bin的变化范围，个数跟channels应该跟channels一致
    const float* ranges[]={range};

    //定义直方图，这里求的是直方图数据
    cv::Mat hist;
    //opencv中计算直方图的函数，hist大小为256*1，每行存储的统计的该行对应的灰度值的个数
    cv::calcHist(&grayImage,1,channels,cv::Mat(),hist,1,histSize,ranges,true,false);//cv中是cvCalcHist

    //找出直方图统计的个数的最大值，用来作为直方图纵坐标的高
    double maxValue=0;
    //找矩阵中最大最小值及对应索引的函数
    cv::minMaxLoc(hist,0,&maxValue,0,0);
    //最大值取整
    int rows=cvRound(maxValue);
    //定义直方图图像，直方图纵坐标的高作为行数，列数为256(灰度值的个数)
    //因为是直方图的图像，所以以黑白两色为区分，白色为直方图的图像
    cv::Mat histImage=cv::Mat::zeros(rows,256,CV_8UC1);

    //直方图图像表示
    for(int i=0;i<256;i++)
    {
        //取每个bin的数目
        int temp=(int)(hist.at<float>(i,0));
        //如果bin数目为0，则说明图像上没有该灰度值，则整列为黑色
        //如果图像上有该灰度值，则将该列对应个数的像素设为白色
        if(temp)
        {
            //由于图像坐标是以左上角为原点，所以要进行变换，使直方图图像以左下角为坐标原点
            histImage.col(i).rowRange(cv::Range(rows-temp,rows))=255;
        }
    }
    //由于直方图图像列高可能很高，因此进行图像对列要进行对应的缩减，使直方图图像更直观
    cv::Mat resizeImage;
    cv::resize(histImage,resizeImage,cv::Size(256,256));
    return resizeImage;
}

float RectExtract::Point2Line(cv::Point Point, vector<cv::Point2f> datum, int flag){
    float A,B,C;
    if(flag==1){
        A=datum[0].y-datum[2].y;
        B=datum[2].x-datum[0].x;
        C=(datum[0].x-datum[2].x)*datum[2].y-datum[2].x*(datum[0].y-datum[2].y);
    }
    else if(flag==0){
        A=datum[1].y-datum[2].y;
        B=datum[2].x-datum[1].x;
        C=(datum[1].x-datum[2].x)*datum[2].y-datum[2].x*(datum[1].y-datum[2].y);
    }
    else{
        return -1;
    }
    if(A*A+B*B==0){
        return -1;
    }
    float dist;
    dist=fabs((A*Point.x+B*Point.y+C)/sqrt(A*A+B*B));
    return dist;
}

int RectExtract::GetDatum(cv::Mat srcimg,vector<cv::Point2f>& datum){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat toplineimg=srcimg(cv::Rect(cv::Point(1806,56),cv::Point(2702,309)));
    cv::Mat topedge = krisch(toplineimg);
    cv::threshold(topedge, topedge, 150, 255, cv::THRESH_BINARY);
    //cv::imshow("top",topedge);
    vector<cv::Point> toplinecontour;
    ScanLine(topedge,3,toplinecontour,0,5,0.8);
    for(int i=0;i<toplinecontour.size();++i){
        toplinecontour[i].x+=1806;
        toplinecontour[i].y+=56;
    }
    vector<float> toplinepara;
    LineFitLeastSquares(toplinecontour,toplinepara);
    cv::Point2f topline;
    topline.x=(2702+1806)/2;
    topline.y=toplinepara[0]*topline.x+toplinepara[1];
    //cout<<topline<<endl;
    cv::Mat leftlineimg=srcimg(cv::Rect(cv::Point(32,1120),cv::Point(192,1760)));
    cv::Mat leftedge = krisch(leftlineimg);
    cv::threshold(leftedge, leftedge, 150, 255, cv::THRESH_BINARY);
    //cv::imshow("left",leftedge);
    vector<cv::Point> leftlinecontour;
    ScanLine(leftedge,2,leftlinecontour,0,3,0.8);
    for(int i=0;i<leftlinecontour.size();++i){
        leftlinecontour[i].x+=32;
        leftlinecontour[i].y+=1120;
    }
    for(int i=0;i<leftlinecontour.size();++i){
        cv::Point temp=leftlinecontour[i];
        leftlinecontour[i].x=temp.y;
        leftlinecontour[i].y=temp.x;
    }
    vector<float> leftlinepara;
    LineFitLeastSquares(leftlinecontour,leftlinepara);
    cv::Point2f leftline;
    leftline.y=(1120+1760)/2;
    leftline.x=leftlinepara[0]*leftline.x+leftlinepara[1];
    //cout<<leftline<<endl;
    cv::Point2f crosspoint;
    crosspoint.x=(-leftlinepara[1]/leftlinepara[0]-toplinepara[1])/(toplinepara[0]-1/leftlinepara[0]);
    crosspoint.y=crosspoint.x*toplinepara[0]+toplinepara[1];
    //cout<<crosspoint<<endl;
    datum.clear();
    datum.push_back(topline);
    datum.push_back(leftline);
    datum.push_back(crosspoint);
    return 0;
}

int RectExtract::GetNewDatum(cv::Mat srcimg, vector<cv::Point2f>& datum){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::threshold(srcimg, srcimg, 160, 255, cv::THRESH_BINARY);
    cv::Mat TopRight=srcimg(cv::Rect(cv::Point(4220,170),cv::Point(4690,220)));
    cv::Mat TopLeft=srcimg(cv::Rect(cv::Point(150,170),cv::Point(290,210)));
    cv::Mat Bottom=srcimg(cv::Rect(cv::Point(370,2220),cv::Point(420,2730)));
    vector<cv::Point> contourstr;
    ScanLine(TopRight,1,contourstr,TopRight.rows-1,5,0.5);
    for(int i=0;i<contourstr.size();++i){
        contourstr[i].x+=4220;
        contourstr[i].y+=170;
    }
    vector<cv::Point> contourstl;
    ScanLine(TopLeft,1,contourstl,TopLeft.rows-1,5,0.5);
    for(int i=0;i<contourstl.size();++i){
        contourstl[i].x+=150;
        contourstl[i].y+=170;
    }
    vector<float> top;
    contourstr.insert(contourstr.end(),contourstl.begin(),contourstl.end());
    LineFitLeastSquares(contourstr,top);
    vector<cv::Point> contoursl;
    ScanLine(Bottom,4,contoursl,Bottom.cols-1,5,0.5);
    for(int i=0;i<contoursl.size();++i){
        cv::Point temp=contoursl[i];
        contoursl[i].x=temp.y+2220;
        contoursl[i].y=temp.x+370;
    }
    vector<float> left;
    LineFitLeastSquares(contoursl,left);
    cv::Point2f topleft(0,0);
    if(top[0]-1/left[0]<-0.000001||top[0]-1/left[0]>0.000001){
        topleft.x=(-left[1]/left[0]-top[1])/(top[0]-1/left[0]);
    }
    topleft.y=topleft.x*top[0]+top[1];
    cv::Point2f toppoint(0,0),leftpoint(0,0);
    toppoint.x=4500;
    toppoint.y=4500*top[0]+top[1];
    leftpoint.y=2500;
    leftpoint.x=leftpoint.y*left[0]+left[1];
    datum.clear();
    datum.push_back(topleft);
    datum.push_back(toppoint);
    datum.push_back(leftpoint);
    return 0;
}

void RectExtract::FirstDerivative(vector<float> data, vector<float>& firstdev){
    firstdev.clear();
    for(int i=4;i<data.size()-4;++i){
        float m0=(3*data[i-4]-16*data[i-3]+36*data[i-2]-48*data[i-1]+25*data[i])/12;
        float m1=(-1*data[i-3]+6*data[i-2]-18*data[i-1]+10*data[i]+3*data[i+1])/12;
        float m2=(1*data[i-2]-8*data[i-1]+8*data[i+1]-1*data[i+2])/12;
        float m3=(-3*data[i-1]+6*data[i]-18*data[i+1]+10*data[i+2]+3*data[i+3])/12;
        float m4=(-25*data[i]+48*data[i+1]-36*data[i+2]+16*data[i+3]-3*data[i+4])/12;
        firstdev.push_back((m0+m1+m2+m3+m4)/5);
    }
}

void RectExtract::SecondDerivative(vector<float> data, vector<float>& seconddev){
    seconddev.clear();
    for(int i=4;i<data.size()-4;++i){
        float m0=(11*data[i-4]-56*data[i-3]+114*data[i-2]-104*data[i-1]+35*data[i])/12;
        float m1=(-1*data[i-3]+4*data[i-2]+6*data[i-1]-20*data[i]+11*data[i+1])/12;
        float m2=(-1*data[i-2]+16*data[i-1]-30*data[i]+16*data[i+1]-1*data[i+2])/12;
        float m3=(11*data[i-1]-20*data[i]+6*data[i+1]+4*data[i+2]-1*data[i+3])/12;
        float m4=(35*data[i]-104*data[i+1]+114*data[i+2]-56*data[i+3]+11*data[i+4])/12;
        seconddev.push_back((m0+m1+m2+m3+m4)/5);
    }
}

int RectExtract::Pixel2Curve(cv::Mat srcimg,int startpos, int endpos, int orientation, vector<vector<float>>& curves){
    curves.clear();
    if(orientation==SCAN_DIRECTION_RIGHT){//从左往右
        for(int i=startpos;i<endpos-1;i=i+1){
            vector<float> curve;
            for(int j=5;j<srcimg.cols-1;++j){
                float kerneldata=(srcimg.at<uchar>(i,j)+srcimg.at<uchar>(i,j+1)+srcimg.at<uchar>(i+1,j)+srcimg.at<uchar>(i+1,j+1))/4;
                curve.push_back(kerneldata);
            }
            curves.push_back(curve);
        }
    }
    else if(orientation==SCAN_DIRECTION_DOWN){//从上往下
        for(int i=startpos;i<endpos-1;i=i+1){
            vector<float> curve;
            for(int j=5;j<srcimg.rows-1;++j){
                float kerneldata=(srcimg.at<uchar>(j,i)+srcimg.at<uchar>(j,i+1)+srcimg.at<uchar>(j+1,i)+srcimg.at<uchar>(j+1,i+1))/4;
                curve.push_back(kerneldata);
            }
            curves.push_back(curve);
        }
    }
    else if(orientation==SCAN_DIRECTION_LEFT){//从右往左
        for(int i=startpos;i<endpos-1;i=i+1){
            vector<float> curve;
            for(int j=srcimg.cols-1-5;j>0;--j){
                float kerneldata=(srcimg.at<uchar>(i,j)+srcimg.at<uchar>(i,j-1)+srcimg.at<uchar>(i+1,j)+srcimg.at<uchar>(i+1,j-1))/4;
                curve.push_back(kerneldata);
            }
            curves.push_back(curve);
        }
    }
    else if(orientation==SCAN_DIRECTION_UP){//从下往上
        for(int i=startpos;i<endpos-1;i=i+1){
            vector<float> curve;
            for(int j=srcimg.rows-1-5;j>0;--j){
                float kerneldata=(srcimg.at<uchar>(j,i)+srcimg.at<uchar>(j,i+1)+srcimg.at<uchar>(j-1,i)+srcimg.at<uchar>(j-1,i+1))/4;
                curve.push_back(kerneldata);
            }
            curves.push_back(curve);
        }
    }
    return 0;
}

int RectExtract::GetDarkGap(cv::Mat srcimg, vector<cv::Point>& seedpoints){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    vector<vector<float>> pixelcurves;
    int orientation=3;
    int startline=700;
    //ofstream pixeldata;
    //pixeldata.open("data.txt");
    Pixel2Curve(srcimg, startline, startline+100, orientation, pixelcurves);
    for(int i=0;i<pixelcurves.size();++i){
        for(int j=0;j<pixelcurves[i].size();++j){
            if(j==pixelcurves[i].size()-1){
                //pixeldata<<pixelcurves[i][j]<<endl;
            }
            else{
                //pixeldata<<pixelcurves[i][j]<<" ";
            }
        }
    }
    //pixeldata.close();
    for(int i=0;i<pixelcurves.size();++i){
        vector<float> firstdev,seconddev;
        FirstDerivative(pixelcurves[i], firstdev);
        SecondDerivative(pixelcurves[i], seconddev);
        vector<cv::Point> seedline;
        vector<int> firstdevres;
        for(int j=0;j<firstdev.size();++j){
            if(fabs(firstdev[j])<1.0&&seconddev[j]>0&&seconddev[j]<1.0){
                if(orientation==0&&srcimg.at<uchar>(startline+i,j)<30){
                    firstdevres.push_back(fabs(firstdev[j]));
                    seedline.push_back(cv::Point(j,startline+i));
                }
                else if(orientation==1&&srcimg.at<uchar>(j,startline+i)<30){
                    firstdevres.push_back(fabs(firstdev[j]));
                    seedline.push_back(cv::Point(startline+i,j));
                }
                else if(orientation==2&&srcimg.at<uchar>(startline+i,firstdev.size()-1-j)<30){
                    firstdevres.push_back(fabs(firstdev[j]));
                    seedline.push_back(cv::Point(firstdev.size()-1-j,startline+i));
                }
                else if(orientation==3&&srcimg.at<uchar>(firstdev.size()-1-j,startline+i)<30){
                    firstdevres.push_back(fabs(firstdev[j]));
                    seedline.push_back(cv::Point(startline+i,firstdev.size()-1-j));
                }
            }
        }
        auto smallest=min_element(firstdevres.begin(),firstdevres.end());
        seedpoints.push_back(seedline[std::distance(std::begin(firstdevres), smallest)]);
    }
    return 0;
}

void RectExtract::GetSeedPos(vector<vector<float>> curves, int thresh, vector<int>& seedpoints){
    seedpoints.clear();
    for(int i=0;i<curves.size();++i){
        int startpos=0,endpos=curves[i].size()-2;
        int flag=0;
        for(int j=2;j<curves[i].size()-2;++j){
            if(curves[i][j-2]+curves[i][j-1]+curves[i][j]+curves[i][j+1]+curves[i][j+2]<thresh*5){
                if(!flag){
                    flag=1;
                    startpos=j;
                }
            }
            else{
                if(flag){
                    endpos=j;
                    break;
                }
            }
        }
        if((startpos+endpos)/2>=2&&(startpos+endpos)/2<curves[i].size()-2){
            seedpoints.push_back((startpos+endpos)/2);
        }
    }
}

void RectExtract::GetSeedPosSecond(vector<vector<float>> curves,int thresh, vector<int>& seedpoints){
    seedpoints.clear();
    for(int i=0;i<curves.size();++i){
        int startpos=0,endpos=curves[i].size()-2;
        int flag=0,num=0;
        for(int j=2;j<curves[i].size()-2;++j){
            if(curves[i][j-2]+curves[i][j-1]+curves[i][j]+curves[i][j+1]+curves[i][j+2]<thresh*5){
                flag=1;
                if(flag&&num==1){
                    startpos=j;
                    num++;
                }
            }
            else{
                if(flag){
                    num++;
                }
                if(num==3){
                    endpos=j;
                    break;
                }
                flag=0;
            }
        }
        if((startpos+endpos)/2>=2&&(startpos+endpos)/2<curves[i].size()-2){
            seedpoints.push_back((startpos+endpos)/2);
        }
    }
}

void RectExtract::GetSeedPoints(cv::Mat srcimg,int orientation,int startline,int endline,int threshold, vector<cv::Point>& seedpoints){
    vector<vector<float>> pixelcurves;
    Pixel2Curve(srcimg, startline, endline, orientation, pixelcurves);
    vector<int> seedpointspos;
    GetSeedPos(pixelcurves,threshold,seedpointspos);
    for(int i=0;i<seedpointspos.size();++i){
        if(orientation==SCAN_DIRECTION_RIGHT){
            seedpoints.push_back(cv::Point(seedpointspos[i]+5,startline+i*2));
        }
        else if(orientation==SCAN_DIRECTION_DOWN){
            seedpoints.push_back(cv::Point(startline+i*2,seedpointspos[i]+5));
        }
        else if(orientation==SCAN_DIRECTION_LEFT){
            seedpoints.push_back(cv::Point(srcimg.cols-1-5-seedpointspos[i],startline+i*2));
        }
        else if(orientation==SCAN_DIRECTION_UP){
            seedpoints.push_back(cv::Point(startline+i*2,srcimg.rows-1-5-seedpointspos[i]));
        }
    }
}

void RectExtract::GetSrcEdgePoints(cv::Mat srcimg,int orientation,int offset,int startline,int endline,int threshold, int threshold1, vector<cv::Point>& edgepoints){
    vector<vector<float>> pixelcurves;
    Pixel2Curve(srcimg, startline, endline, orientation, pixelcurves);
    vector<int> seedpointspos;
    vector<int> edgepointspos;
    GetSeedPos(pixelcurves,threshold1,seedpointspos);
    for(int i=0;i<seedpointspos.size();++i){
        seedpointspos[i]+=offset;
    }
    if(seedpointspos.size()==0){
        return;
    }
    for(int i=0;i<pixelcurves.size();++i){
        for(int j=seedpointspos[i];j<pixelcurves[i].size();++j){
            if(pixelcurves[i][j]>threshold){
                edgepointspos.push_back(j);
                break;
            }
        }
    }
    for(int i=0;i<edgepointspos.size();++i){
        if(orientation==SCAN_DIRECTION_RIGHT){
            edgepoints.push_back(cv::Point(edgepointspos[i]+5,startline+i*2));
        }
        else if(orientation==SCAN_DIRECTION_DOWN){
            edgepoints.push_back(cv::Point(startline+i*2,edgepointspos[i]+5));
        }
        else if(orientation==SCAN_DIRECTION_LEFT){
            edgepoints.push_back(cv::Point(srcimg.cols-1-5-edgepointspos[i],startline+i*2));
        }
        else if(orientation==SCAN_DIRECTION_UP){
            edgepoints.push_back(cv::Point(startline+i*2,srcimg.rows-1-5-edgepointspos[i]));
        }
    }
}

void RectExtract::GetSeedPointsSec(cv::Mat srcimg,int orientation,int startline,int endline,int threshold, vector<cv::Point>& seedpoints){
    vector<vector<float>> pixelcurves;
    Pixel2Curve(srcimg, startline, endline, orientation, pixelcurves);
    vector<int> seedpointspos;
    GetSeedPosSecond(pixelcurves,threshold,seedpointspos);
    for(int i=0;i<seedpointspos.size();++i){
        if(orientation==SCAN_DIRECTION_RIGHT){
            seedpoints.push_back(cv::Point(seedpointspos[i]+5,startline+i*2));
        }
        else if(orientation==SCAN_DIRECTION_DOWN){
            seedpoints.push_back(cv::Point(startline+i*2,seedpointspos[i]+5));
        }
        else if(orientation==SCAN_DIRECTION_LEFT){
            seedpoints.push_back(cv::Point(srcimg.cols-1-5-seedpointspos[i],startline+i*2));
        }
        else if(orientation==SCAN_DIRECTION_UP){
            seedpoints.push_back(cv::Point(startline+i*2,srcimg.rows-1-5-seedpointspos[i]));
        }
    }
}


void RectExtract::GetEdgePoint(cv::Mat srcimg, vector<cv::Point> seedpoints, int orientation, int thresh,int radius,float percent,  vector<cv::Point>& contours){
    cv::Mat edge = krisch(srcimg);
    cv::threshold(edge, edge, thresh, 255, cv::THRESH_BINARY);
    cv::imshow("edge",edge);
    if(orientation==SCAN_DIRECTION_UP){
        for(int i=0;i<seedpoints.size();++i){
            cv::Point seed=seedpoints[i];
            cv::Point contourpoint(seed.x,seed.y-1);
            while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y-=1;
                if(contourpoint.y==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(orientation==SCAN_DIRECTION_RIGHT){
        for(int i=0;i<seedpoints.size();++i){
            cv::Point seed=seedpoints[i];
            cv::Point contourpoint(seed.x+1,seed.y);
            while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x+=1;
                if(contourpoint.x>=edge.cols)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(orientation==SCAN_DIRECTION_DOWN){
        for(int i=0;i<seedpoints.size();++i){
            cv::Point seed=seedpoints[i];
            cv::Point contourpoint(seed.x,seed.y+1);
            while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y+=1;
                if(contourpoint.y>=edge.rows)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(orientation==SCAN_DIRECTION_LEFT){
        for(int i=0;i<seedpoints.size();++i){
            cv::Point seed=seedpoints[i];
            cv::Point contourpoint(seed.x-1,seed.y);
            while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x-=1;
                if(contourpoint.x==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
}

int RectExtract::KirschEdgeOuter(cv::Mat srcimg, int thresh, int num, float& dist){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    vector<cv::Point> seedpoints;
    vector<cv::Point> contours;
    int orientation;
    if(num==1){
        orientation=SCAN_DIRECTION_UP;
        GetSeedPoints(srcimg,orientation,290,310,30,seedpoints);
        if(seedpoints.size()!=10){
            dist=0.0f;
            return -2;
        }
        GetEdgePoint(srcimg, seedpoints,orientation,thresh,4,0.6,contours);
    }
    else if(num==2){
        orientation=SCAN_DIRECTION_UP;
        GetSeedPoints(srcimg,orientation,800,820,30,seedpoints);
        if(seedpoints.size()!=10){
            dist=0.0f;
            return -2;
        }
        GetEdgePoint(srcimg, seedpoints,orientation,thresh,4,0.6,contours);
    }
    else if(num==3){
        orientation=SCAN_DIRECTION_RIGHT;
//        GetSrcEdgePoints(srcimg,orientation,290,310,90, contours);
//        GetSrcEdgePoints(srcimg,orientation,860,880,90, contours);
//        GetSrcEdgePoints(srcimg,orientation,1440,1460,90, contours);
    }
    else if(num==4){
        orientation=SCAN_DIRECTION_DOWN;
        GetSeedPoints(srcimg,orientation,240,260,40,seedpoints);
        GetSeedPoints(srcimg,orientation,700,720,40,seedpoints);
        if(seedpoints.size()!=20){
            dist=0.0f;
            return -2;
        }
        GetEdgePoint(srcimg, seedpoints,orientation,thresh,4,0.6,contours);
    }
    else if(num==5){
        orientation=SCAN_DIRECTION_RIGHT;
        GetSeedPoints(srcimg,SCAN_DIRECTION_RIGHT,440,460,40,seedpoints);
        GetSeedPoints(srcimg,SCAN_DIRECTION_RIGHT,130,150,40,seedpoints);
        if(seedpoints.size()!=20){
            dist=0.0f;
            return -2;
        }
        GetEdgePoint(srcimg, seedpoints,SCAN_DIRECTION_RIGHT,thresh,4,0.6,contours);
    }
    else if(num==6){
        orientation=SCAN_DIRECTION_DOWN;
        GetSeedPointsSec(srcimg,SCAN_DIRECTION_UP,110,130,60,seedpoints);
        GetSeedPointsSec(srcimg,SCAN_DIRECTION_UP,760,780,60,seedpoints);
        if(seedpoints.size()!=20){
            dist=0.0f;
            return -2;
        }
        GetEdgePoint(srcimg, seedpoints,SCAN_DIRECTION_DOWN,thresh,4,0.5,contours);
    }
    else if(num==7){
        orientation=SCAN_DIRECTION_LEFT;
        GetSeedPointsSec(srcimg,SCAN_DIRECTION_RIGHT,130,150,40,seedpoints);
        GetSeedPointsSec(srcimg,SCAN_DIRECTION_RIGHT,700,720,40,seedpoints);
        if(seedpoints.size()!=20){
            dist=0.0f;
            return -2;
        }
        GetEdgePoint(srcimg, seedpoints,SCAN_DIRECTION_LEFT,thresh,4,0.6,contours);
    }
    else if(num==8){
        orientation=SCAN_DIRECTION_DOWN;
        GetSeedPointsSec(srcimg,SCAN_DIRECTION_UP,300,320,40,seedpoints);
        GetSeedPointsSec(srcimg,SCAN_DIRECTION_UP,400,420,40,seedpoints);
        GetSeedPointsSec(srcimg,SCAN_DIRECTION_UP,590,610,40,seedpoints);
        if(seedpoints.size()!=30){
            dist=0.0f;
            return -2;
        }
        GetEdgePoint(srcimg, seedpoints,SCAN_DIRECTION_DOWN,thresh,4,0.6,contours);
    }
    else if(num==9){
        orientation=SCAN_DIRECTION_LEFT;
        GetSeedPointsSec(srcimg,SCAN_DIRECTION_RIGHT,100,120,40,seedpoints);//
        GetSeedPointsSec(srcimg,SCAN_DIRECTION_RIGHT,640,660,40,seedpoints);//
        GetSeedPointsSec(srcimg,SCAN_DIRECTION_RIGHT,930,950,40,seedpoints);//
        if(seedpoints.size()!=30){
            dist=0.0f;
            return -2;
        }
        GetEdgePoint(srcimg, seedpoints,SCAN_DIRECTION_LEFT,thresh,4,0.6,contours);
    }
    float sum=0;
    for(int i=0;i<contours.size();++i){
        if(orientation==0||orientation==2){
            sum+=contours[i].x;
        }
        else{
            sum+=contours[i].y;
        }
    }
    dist=sum/contours.size();
    for(int i=0;i<seedpoints.size();++i){
        cv::circle(srcimg,seedpoints[i],1,255,-1);
    }
    for(int i=0;i<contours.size();++i){
        cv::circle(srcimg,contours[i],1,255,-1);
    }
    cv::imshow("dst",srcimg);
    cv::imwrite("a.jpg",srcimg);
    return 0;
}

int RectExtract::GetDouLines(cv::Mat srcimg, vector<cv::Point>& linepoints){
    cv::Mat edge;
    int radius=3;
    float percent=0.55;
    vector<cv::Point> contours;
    cv::Vec2f linepara;
    cv::threshold(srcimg, edge, 80, 255, cv::THRESH_BINARY_INV);
    //cv::imshow("edge",edge);
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
//    for(int i=0;i<linepoints.size();++i){
//        cv::circle(srcimg,linepoints[i],3,255,-1);
//    }
    //cv::imshow("src",srcimg);
    return 0;
}

float RectExtract::GetMidLineY(vector<vector<cv::Point>> linecontours, int pos){
    vector<float> toplinepara,bottomlinepara;
    LineFitLeastSquares(linecontours[0],toplinepara);
    LineFitLeastSquares(linecontours[1],bottomlinepara);
    float ypos=(toplinepara[0]*pos+toplinepara[1]+bottomlinepara[0]*pos+bottomlinepara[1])/2.0;
    return ypos;
}

float RectExtract::GetLineX(vector<cv::Point> linecontour, int pos){
    vector<float> leftlinepara;
    for(int i=0;i<linecontour.size();++i){
        cv::Point temp=linecontour[i];
        linecontour[i].x=temp.y;
        linecontour[i].y=temp.x;
    }
    LineFitLeastSquares(linecontour,leftlinepara);
    float xpos=leftlinepara[0]*pos+leftlinepara[1];
    return xpos;
}














int RectExtract::NewKirschEdgeOuter(cv::Mat srcimg, const vector<int> paras, vector<cv::Point2f>& dist){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    vector<cv::Point> seedpoints;
    vector<cv::Point> contours;
    dist.clear();
    if(paras[2]==0){
        int len=(paras[11]-paras[10])/2;
        for(int i=0;i<paras[9];++i){
            GetSeedPoints(srcimg,paras[4],paras[9+2*i+1],paras[9+2*i+2],paras[3],seedpoints);
        }
        GetEdgePoint(srcimg, seedpoints,paras[5],paras[1],paras[6],(float)paras[7]/100,contours);
        for(int i=0;i<paras[9];++i){
            cv::Point2f selectpoint(0,0);
            for(int j=i*len;j<i*len+len;++j){
                selectpoint.x+=contours[j].x;
                selectpoint.y+=contours[j].y;
            }
            selectpoint.x=selectpoint.x/len;
            selectpoint.y=selectpoint.y/len;
            dist.push_back(selectpoint);
        }
    }
    else if(paras[2]==1){
        int len=(paras[11]-paras[10])/2;
        for(int i=0;i<paras[9];++i){
            GetSeedPointsSec(srcimg,paras[4],paras[9+2*i+1],paras[9+2*i+2],paras[3],seedpoints);
        }
        GetEdgePoint(srcimg, seedpoints,paras[5],paras[1],paras[6],(float)paras[7]/100,contours);
        for(int i=0;i<paras[9];++i){
            cv::Point2f selectpoint(0,0);
            for(int j=i*len;j<i*len+len;++j){
                selectpoint.x+=contours[j].x;
                selectpoint.y+=contours[j].y;
            }
            selectpoint.x=selectpoint.x/len;
            selectpoint.y=selectpoint.y/len;
            dist.push_back(selectpoint);
        }
    }
    else if(paras[2]==2){
        int len=(paras[9]-paras[8])/2;
        for(int i=0;i<paras[7];++i){
            GetSrcEdgePoints(srcimg,paras[4],paras[5],paras[7+2*i+1],paras[7+2*i+2],paras[3],paras[1],contours);
        }
        for(int i=0;i<paras[7];++i){
            cv::Point2f selectpoint(0,0);
            for(int j=i*len;j<i*len+len;++j){
                selectpoint.x+=contours[j].x;
                selectpoint.y+=contours[j].y;
            }
            selectpoint.x=selectpoint.x/len;
            selectpoint.y=selectpoint.y/len;
            dist.push_back(selectpoint);
        }
    }
    for(int i=0;i<seedpoints.size();++i){
        cv::circle(srcimg,seedpoints[i],1,255,-1);
    }
    for(int i=0;i<contours.size();++i){
        cv::circle(srcimg,contours[i],1,255,-1);
    }
    cv::imshow("dst",srcimg);
    cv::imwrite("datum2-3.jpg",srcimg);
    return 0;
}

int RectExtract::NewCircleEdge(cv::Mat srcimg, const vector<int> paras, vector<cv::Point>& contours){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat edge;
    if(paras[2]==0){
        edge = krisch(srcimg);
        cv::threshold(edge, edge, paras[1], 255, cv::THRESH_BINARY);
    }
    else if(paras[2]==1){
        cv::threshold(srcimg, edge, paras[1], 255, cv::THRESH_BINARY_INV);
    }
    else if(paras[2]==2){
        cv::threshold(srcimg, edge, paras[1], 255, cv::THRESH_BINARY);
    }
    cv::imshow("edge",edge);
    edge.at<uchar>(paras[4],paras[3])=0;
    FindEdge(edge, cv::Point(paras[3],paras[4]), contours, paras[5], (float)paras[6]/100, paras[7]);
    contours=ContoursCut(contours, paras[8], paras[9]);
    for(int i=0;i<contours.size();++i){
        cv::circle(srcimg,contours[i],2,160,-1);
    }
    cout<<contours.size()<<endl;
    cv::imshow("edgepoint",srcimg);
    cv::imwrite("edgepoint.jpg",srcimg);
    return 0;
}

int RectExtract::NewSmallCircle(cv::Mat srcimg, const vector<int> paras,  vector<float>& result){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    vector<cv::Point> contours;
    NewCircleEdge(srcimg, paras,contours);
    result=FitCircle(contours);
    float roundness=GetRoundness(contours,cv::Point2f(result[0],result[1]));
    return 0;
}

int RectExtract::ScanLineRange(cv::Mat srcimg,int lineori, vector<cv::Point>& contours,int startline,int startedge, int endedge, int radius,float percent){
    if(srcimg.empty()||srcimg.cols<radius*2+2||srcimg.rows<radius*2+2||startline<0)
        return -1;
    if(startedge<radius+1){
        startedge=radius+1;
    }
    contours.clear();
    if(lineori==SCAN_DIRECTION_UP){
        if(endedge>srcimg.cols-radius-1){
            endedge=srcimg.cols-radius-1;
        }
        for(int i=startedge;i<endedge;++i){
            cv::Point seed(i,startline);
            cv::Point contourpoint(i,seed.y-1);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y-=1;
                if(contourpoint.y==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(lineori==SCAN_DIRECTION_RIGHT){
        if(endedge>srcimg.rows-radius-1){
            endedge=srcimg.rows-radius-1;
        }
        for(int i=startedge;i<endedge;++i){
            cv::Point seed(startline,i);
            cv::Point contourpoint(seed.x+1,i);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x+=1;
                if(contourpoint.x>=srcimg.cols)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(lineori==SCAN_DIRECTION_DOWN){
        if(endedge>srcimg.cols-radius-1){
            endedge=srcimg.cols-radius-1;
        }
        for(int i=startedge;i<endedge;++i){
            cv::Point seed(i,startline);
            cv::Point contourpoint(i,seed.y+1);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y+=1;
                if(contourpoint.y>=srcimg.rows)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(lineori==SCAN_DIRECTION_LEFT){
        if(endedge>srcimg.rows-radius-1){
            endedge=srcimg.rows-radius-1;
        }
        for(int i=startedge;i<endedge;++i){
            cv::Point seed(startline,i);
            cv::Point contourpoint(seed.x-1,i);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x-=1;
                if(contourpoint.x==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    return 0;
}

float RectExtract::PointLineDist(cv::Point2f A,vector<float> Lineparas){
    if(Lineparas.size()<2){
        return -1;
    }
    float T=fabs(Lineparas[0]*A.x-A.y+Lineparas[1]);
    float B=sqrt(Lineparas[0]*Lineparas[0]+1);
    return T/B;
}

int RectExtract::NewGetRectEdge(cv::Mat srcimg, const vector<int> paras, vector<float>& dists, vector<vector<cv::Point2f>>& contourres){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat edge;
    if(paras[2]==0){
        edge = krisch(srcimg);
        cv::threshold(edge, edge, paras[1], 255, cv::THRESH_BINARY);
    }
    else if(paras[2]==1){
        cv::threshold(srcimg, edge, paras[1], 255, cv::THRESH_BINARY_INV);
    }
    cv::imshow("edge",edge);
    dists.clear();
    contourres.clear();
    if(paras[0]==1){
        vector<cv::Point> contours;
        //top line
        ScanLineRange(edge,SCAN_DIRECTION_UP,contours,100,paras[5],paras[6],paras[3],(float)paras[4]/100);
        vector<cv::Point> fitpoints(10);
        for(int i=0;i<fitpoints.size();++i){
            fitpoints[i]=contours[i];
            if(i>4){
                fitpoints[i]=contours[contours.size()-10+i];
            }
        }
        vector<float> topres;
        LineFitLeastSquares(fitpoints, topres);
        float maxdist1=0;
        for(int i=5;i<contours.size()-5;++i){
            float dist=PointLineDist(contours[i],topres);
            if(maxdist1<dist){
                maxdist1=dist;
            }
        }
        dists.push_back(maxdist1);
        //bottom line
        ScanLineRange(edge,SCAN_DIRECTION_DOWN,contours,edge.rows-100,paras[7],paras[8],paras[3],(float)paras[4]/100);
        for(int i=0;i<fitpoints.size();++i){
            fitpoints[i]=contours[i];
            if(i>4){
                fitpoints[i]=contours[contours.size()-10+i];
            }
        }
        vector<float> botres;
        LineFitLeastSquares(fitpoints, botres);
        float maxdist2=0;
        for(int i=5;i<contours.size()-5;++i){
            float dist=PointLineDist(contours[i],botres);
            if(maxdist2<dist){
                maxdist2=dist;
            }
        }
        dists.push_back(maxdist2);
        //left line & right line and parallel
        ScanLineRange(edge,SCAN_DIRECTION_LEFT,contours,100,paras[9],paras[10],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf[i]=temp;
        }
        contourres.push_back(contoursf);
        for(int i=0;i<contours.size();++i){
            cv::Point temp=contours[i];
            contours[i].x=temp.y;
            contours[i].y=temp.x;
        }
        vector<float> leftres;
        LineFitLeastSquares(contours, leftres);
        ScanLineRange(edge,SCAN_DIRECTION_RIGHT,contours,edge.cols-100,paras[9],paras[10],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf1(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf1[i]=temp;
        }
        contourres.push_back(contoursf1);
        for(int i=0;i<contours.size();++i){
            cv::Point temp=contours[i];
            contours[i].x=temp.y;
            contours[i].y=temp.x;
        }
        vector<float> rightres;
        LineFitLeastSquares(contours, rightres);
        vector<cv::Point2f> centerline(paras[10]-paras[9]);
        for(int i=paras[9];i<paras[10];++i){
            centerline[i-paras[9]].y=i;
            centerline[i-paras[9]].x=fabs(i*leftres[0]+leftres[1]+i*rightres[0]+rightres[1])/2.0;
        }
        contourres.push_back(centerline);
        //top lines parallel
        vector<cv::Point> contours2;
        ScanLineRange(edge,SCAN_DIRECTION_UP,contours,100,paras[11],paras[12],paras[3],(float)paras[4]/100);
        ScanLineRange(edge,SCAN_DIRECTION_UP,contours2,100,paras[13],paras[14],paras[3],(float)paras[4]/100);
        contours.insert(contours.end(),contours2.begin(),contours2.end());
        vector<float> topres1;
        LineFitLeastSquares(contours, topres1);
        ScanLineRange(edge,SCAN_DIRECTION_DOWN,contours,edge.rows-100,paras[15],paras[16],paras[3],(float)paras[4]/100);
        vector<float> botres1;
        LineFitLeastSquares(contours, botres1);
        vector<cv::Point2f> centerline2(paras[13]-paras[12]);
        for(int i=paras[12];i<paras[13];++i){
            centerline2[i-paras[12]].x=i;
            centerline2[i-paras[12]].y=fabs(i*botres1[0]+botres1[1]+i*topres1[0]+topres1[1])/2.0;
        }
        contourres.push_back(centerline2);
    }
    else if(paras[0]==2){
        vector<cv::Point> contours;
        //top line
        ScanLineRange(edge,SCAN_DIRECTION_UP,contours,100,paras[5],paras[6],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf[i]=temp;
        }
        contourres.push_back(contoursf);
        //bottom line
        ScanLineRange(edge,SCAN_DIRECTION_DOWN,contours,edge.rows-100,paras[5],paras[6],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf1(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf1[i]=temp;
        }
        contourres.push_back(contoursf1);
    }
    return 0;
}

int RectExtract::LDmeasure(cv::Mat srcimg, const vector<int> paras, vector<cv::Point2f>& respts){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat edge;
    if(paras[0]==1){
        cv::threshold(srcimg, edge, paras[1], 255, cv::THRESH_BINARY_INV);
    }
    else if(paras[0]==2){
        cv::threshold(srcimg, edge, paras[1], 255, cv::THRESH_BINARY);
    }
    respts.clear();
    for(int i=0;i<paras[2];++i){
        vector<cv::Point> contours;
        ScanLineRange(edge,paras[8+i*4],contours,paras[5+i*4],paras[6+i*4],paras[7+i*4],paras[3],(float)paras[4]/100);
        cv::Point2f point1(0,0);
        for(int j=0;j<contours.size();++j){
            point1.x+=contours[j].x;
            point1.y+=contours[j].y;
            cv::circle(edge,contours[j],2,160,-1);
        }
        point1.x=point1.x/contours.size();
        point1.y=point1.y/contours.size();
        respts.push_back(point1);
    }
    cv::imshow("edge",edge);
    return 0;
}

int RectExtract::SmallCircle(cv::Mat srcimg, const vector<int> paras, vector<float>& res){
    cv::Rect2f centerarea(paras[0],paras[1],paras[2],paras[3]);
    float r=paras[4];
    float rd=paras[5];
    float d=paras[6];
    int randnum=paras[7];
    vector<vector<float>> circleparas(randnum);
    vector<double> sumres(randnum,0);
    cv::RNG rng((unsigned)time(NULL));
    for(int i=0;i<randnum;++i){
        vector<float> para(3);
        para[0]=rng.uniform(centerarea.x, centerarea.x+centerarea.width);
        para[1]=rng.uniform(centerarea.y, centerarea.y+centerarea.height);
        para[2]=rng.uniform(r, r+rd);
        circleparas[i]=para;
        float theta;
        cv::Point2f center(para[0],para[1]);
        for(int j=0;j<180;++j){
            theta=j;
            float sum=0;
            if(theta>=0&&theta<=90){
                cv::Point contourPoint11(center.x+(para[2]+d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint12(center.x+(para[2]+2*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint13(center.x+(para[2]+3*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+3*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint14(center.x+(para[2]-d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint15(center.x+(para[2]-2*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint16(center.x+(para[2]-3*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-3*d)*fabs(sin(theta/180.0*PI)));
                sum=srcimg.at<uchar>(contourPoint11.y,contourPoint11.x)+srcimg.at<uchar>(contourPoint12.y,contourPoint12.x)+srcimg.at<uchar>(contourPoint13.y,contourPoint13.x)-srcimg.at<uchar>(contourPoint14.y,contourPoint14.x)-srcimg.at<uchar>(contourPoint15.y,contourPoint15.x)-srcimg.at<uchar>(contourPoint16.y,contourPoint16.x);
                sum=sum/6;
                cv::Point contourPoint21(center.x-(para[2]+d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint22(center.x-(para[2]+2*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint23(center.x-(para[2]+3*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint24(center.x-(para[2]-d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint25(center.x-(para[2]-2*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint26(center.x-(para[2]-3*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-3*d)*fabs(sin(theta/180.0*PI)));
                float sum2=srcimg.at<uchar>(contourPoint21.y,contourPoint21.x)+srcimg.at<uchar>(contourPoint22.y,contourPoint22.x)+srcimg.at<uchar>(contourPoint23.y,contourPoint23.x)-srcimg.at<uchar>(contourPoint24.y,contourPoint24.x)-srcimg.at<uchar>(contourPoint25.y,contourPoint25.x)-srcimg.at<uchar>(contourPoint26.y,contourPoint26.x);
                sum2=sum2/6;
                sum=sum+sum2;
            }
            else if(theta>90&&theta<180){
                cv::Point contourPoint11(center.x-(para[2]+d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+d)*fabs(cos(theta/180.0*PI)));
                cv::Point contourPoint12(center.x-(para[2]+2*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+2*d)*fabs(cos(theta/180.0*PI)));
                cv::Point contourPoint13(center.x-(para[2]+3*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+3*d)*fabs(cos(theta/180.0*PI)));
                cv::Point contourPoint14(center.x-(para[2]-d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-d)*fabs(cos(theta/180.0*PI)));
                cv::Point contourPoint15(center.x-(para[2]-2*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-2*d)*fabs(cos(theta/180.0*PI)));
                cv::Point contourPoint16(center.x-(para[2]-3*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-3*d)*fabs(cos(theta/180.0*PI)));
                sum=srcimg.at<uchar>(contourPoint11.y,contourPoint11.x)+srcimg.at<uchar>(contourPoint12.y,contourPoint12.x)+srcimg.at<uchar>(contourPoint13.y,contourPoint13.x)-srcimg.at<uchar>(contourPoint14.y,contourPoint14.x)-srcimg.at<uchar>(contourPoint15.y,contourPoint15.x)-srcimg.at<uchar>(contourPoint16.y,contourPoint16.x);
                sum=sum/6;
                cv::Point contourPoint21(center.x+(para[2]+d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+d)*fabs(cos(theta/180.0*PI)));
                cv::Point contourPoint22(center.x+(para[2]+2*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+2*d)*fabs(cos(theta/180.0*PI)));
                cv::Point contourPoint23(center.x+(para[2]+3*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+2*d)*fabs(cos(theta/180.0*PI)));
                cv::Point contourPoint24(center.x+(para[2]-d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-d)*fabs(cos(theta/180.0*PI)));
                cv::Point contourPoint25(center.x+(para[2]-2*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-2*d)*fabs(cos(theta/180.0*PI)));
                cv::Point contourPoint26(center.x+(para[2]-3*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-3*d)*fabs(cos(theta/180.0*PI)));
                float sum2=srcimg.at<uchar>(contourPoint21.y,contourPoint21.x)+srcimg.at<uchar>(contourPoint22.y,contourPoint22.x)+srcimg.at<uchar>(contourPoint23.y,contourPoint23.x)-srcimg.at<uchar>(contourPoint24.y,contourPoint24.x)-srcimg.at<uchar>(contourPoint25.y,contourPoint25.x)-srcimg.at<uchar>(contourPoint26.y,contourPoint26.x);
                sum2=sum2/6;
                sum=sum+sum2;
            }
            sumres[i]+=sum;
        }
    }
    auto maxPosition = max_element(sumres.begin(), sumres.end());
    int pos=maxPosition-sumres.begin();
    res=circleparas[pos];
    return 0;
}

int RectExtract::GetNewDatum575(cv::Mat srcimg, vector<cv::Point2f>& datum){
    cv::Mat top=srcimg(cv::Rect(cv::Point(40,1010),cv::Point(130,1080)));
    cv::Mat top2=srcimg(cv::Rect(cv::Point(4480,980),cv::Point(4900,1080)));
    cv::Mat left=srcimg(cv::Rect(cv::Point(390,370),cv::Point(550,700)));
    cv::threshold(top,top,60,255,CV_THRESH_BINARY);
    cv::threshold(top2,top2,60,255,CV_THRESH_BINARY);
    cv::threshold(left,left,60,255,CV_THRESH_BINARY);
    vector<cv::Point> contourstr;
    ScanLine(top,1,contourstr,top.rows-1,5,0.2);
    for(int i=0;i<contourstr.size();++i){
        contourstr[i].x+=40;
        contourstr[i].y+=1010;
    }
    vector<cv::Point> contourstr1;
    ScanLine(top2,3,contourstr1,1,5,0.2);
    for(int i=0;i<contourstr1.size();++i){
        contourstr1[i].x+=4480;
        contourstr1[i].y+=980;
    }
    vector<float> toppara;
    contourstr.insert(contourstr.end(),contourstr1.begin(),contourstr1.end());
    LineFitLeastSquares(contourstr,toppara);
    vector<cv::Point> contoursl;
    ScanLine(left,4,contoursl,left.cols-1,5,0.2);
    for(int i=0;i<contoursl.size();++i){
        cv::Point temp=contoursl[i];
        contoursl[i].x=temp.y+370;
        contoursl[i].y=temp.x+390;
    }
    vector<float> leftpara;
    LineFitLeastSquares(contoursl,leftpara);
    for(int i=0;i<srcimg.rows;++i){
        cv::circle(srcimg,cv::Point(i*leftpara[0]+leftpara[1],i),3,255,-1);
    }
    for(int i=0;i<srcimg.cols;++i){
        cv::circle(srcimg,cv::Point(i,i*toppara[0]+toppara[1]),3,255,-1);
    }
    cv::Point2f topleft(0,0);
    if(toppara[0]-1/leftpara[0]<-0.000001||toppara[0]-1/leftpara[0]>0.000001){
        topleft.x=(-leftpara[1]/leftpara[0]-toppara[1])/(toppara[0]-1/leftpara[0]);
    }
    topleft.y=topleft.x*toppara[0]+toppara[1];
    cv::Point2f toppoint(0,0),leftpoint(0,0);
    toppoint.x=3800;
    toppoint.y=3800*toppara[0]+toppara[1];
    leftpoint.y=2200;
    leftpoint.x=leftpoint.y*leftpara[0]+leftpara[1];
    datum.clear();
    datum.push_back(topleft);
    datum.push_back(toppoint);
    datum.push_back(leftpoint);
    cv::circle(srcimg,toppoint,5,255,-1);
    cv::circle(srcimg,leftpoint,5,255,-1);
    cv::circle(srcimg,topleft,5,255,-1);
    cv::imwrite("res.jpg",srcimg);
    return 0;
}

int RectExtract::GetNewDatum616(cv::Mat srcimg, vector<cv::Point2f>& datum){
    cv::Mat left=srcimg(cv::Rect(cv::Point(3629,501),cv::Point(3849,861)));
    cv::Mat left2=srcimg(cv::Rect(cv::Point(3709,4373),cv::Point(3905,4733)));
    cv::Mat top=srcimg(cv::Rect(cv::Point(3761,901),cv::Point(3965,989)));
    cv::threshold(left,left,60,255,CV_THRESH_BINARY);
    cv::threshold(left2,left2,60,255,CV_THRESH_BINARY);
    cv::threshold(top,top,60,255,CV_THRESH_BINARY);
    vector<cv::Point> contourstr;
    ScanLine(top,3,contourstr,1,5,0.5);
    for(int i=0;i<contourstr.size();++i){
        contourstr[i].x+=3761;
        contourstr[i].y+=901;
    }
//    vector<cv::Point> contourstr1;
//    ScanLine(top2,3,contourstr1,1,5,0.2);
//    for(int i=0;i<contourstr1.size();++i){
//        contourstr1[i].x+=3709;
//        contourstr1[i].y+=4373;
//    }
    vector<float> toppara;
//    contourstr.insert(contourstr.end(),contourstr1.begin(),contourstr1.end());
    LineFitLeastSquares(contourstr,toppara);
    vector<cv::Point> contoursl;
    ScanLine(left,4,contoursl,left.cols-1,5,0.5);
    for(int i=0;i<contoursl.size();++i){
        cv::Point temp=contoursl[i];
        contoursl[i].x=temp.y+501;
        contoursl[i].y=temp.x+3629;
    }
    vector<cv::Point> contoursl1;
    ScanLine(left2,4,contoursl1,left2.cols-1,5,0.5);
    for(int i=0;i<contoursl1.size();++i){
        cv::Point temp=contoursl1[i];
        contoursl1[i].x=temp.y+4373;
        contoursl1[i].y=temp.x+3709;
    }
    vector<float> leftpara;
    contoursl.insert(contoursl.end(),contoursl1.begin(),contoursl1.end());
    LineFitLeastSquares(contoursl,leftpara);
    for(int i=0;i<srcimg.rows;++i){
        cv::circle(srcimg,cv::Point(i*leftpara[0]+leftpara[1],i),3,255,-1);
    }
    for(int i=0;i<srcimg.cols;++i){
        cv::circle(srcimg,cv::Point(i,i*toppara[0]+toppara[1]),3,255,-1);
    }
    cv::Point2f topleft(0,0);
    if(toppara[0]-1/leftpara[0]<-0.000001||toppara[0]-1/leftpara[0]>0.000001){
        topleft.x=(-leftpara[1]/leftpara[0]-toppara[1])/(toppara[0]-1/leftpara[0]);
    }
    topleft.y=topleft.x*toppara[0]+toppara[1];
    cv::Point2f toppoint(0,0),leftpoint(0,0);
    toppoint.x=3800;
    toppoint.y=3800*toppara[0]+toppara[1];
    leftpoint.y=2200;
    leftpoint.x=leftpoint.y*leftpara[0]+leftpara[1];
    datum.clear();
    datum.push_back(topleft);
    datum.push_back(toppoint);
    datum.push_back(leftpoint);
    cv::circle(srcimg,toppoint,5,255,-1);
    cv::circle(srcimg,leftpoint,5,255,-1);
    cv::circle(srcimg,topleft,5,255,-1);
    cv::imwrite("res.jpg",srcimg);
    return 0;
}

int RectExtract::CameraCheck(cv::Mat srcimg, int CamId, float& verdist, float& hordist){
    cv::Mat top,bottom,left,right;
    int topleft1,topleft2,topleft3,topleft4;
    cv::threshold(srcimg, srcimg, 160, 255, cv::THRESH_BINARY_INV);
    if(CamId==1){
        topleft1=165;
        topleft2=2141;
        topleft3=65;
        topleft4=4132;
        top=srcimg(cv::Rect(cv::Point(152,165),cv::Point(276,228)));
        bottom=srcimg(cv::Rect(cv::Point(448,2141),cv::Point(631,2220)));
        left=srcimg(cv::Rect(cv::Point(65,227),cv::Point(162,365)));
        right=srcimg(cv::Rect(cv::Point(4132,237),cv::Point(4225,322)));
    }
    else if(CamId==2){
        topleft1=140;
        topleft2=2123;
        topleft3=88;
        topleft4=4160;
        top=srcimg(cv::Rect(cv::Point(173,140),cv::Point(284,208)));
        bottom=srcimg(cv::Rect(cv::Point(470,2123),cv::Point(633,2191)));
        left=srcimg(cv::Rect(cv::Point(88,217),cv::Point(186,358)));
        right=srcimg(cv::Rect(cv::Point(4160,236),cv::Point(4230,306)));
    }
    vector<cv::Point> contour1;
    ScanLine(left,SCAN_DIRECTION_LEFT,contour1,left.cols-1,4,0.5);
    vector<cv::Point> contour2;
    ScanLine(right,SCAN_DIRECTION_LEFT,contour2,right.cols-1,4,0.5);
    float leftx=0;
    for(int i=0;i<contour1.size();++i){
        leftx+=contour1[i].x;
    }
    leftx=leftx/contour1.size();
    float rightx=0;
    for(int i=0;i<contour2.size();++i){
        rightx+=contour2[i].x;
    }
    rightx=rightx/contour2.size();
    hordist=rightx-leftx+topleft4-topleft3;
    vector<cv::Point> contour3;
    ScanLine(top,SCAN_DIRECTION_UP,contour3,top.rows-1,4,0.5);
    vector<cv::Point> contour4;
    ScanLine(bottom,SCAN_DIRECTION_UP,contour4,bottom.rows-1,4,0.5);
    float topy=0;
    for(int i=0;i<contour3.size();++i){
        topy+=contour3[i].y;
    }
    topy=topy/contour3.size();
    float bottomy=0;
    for(int i=0;i<contour4.size();++i){
        bottomy+=contour4[i].y;
    }
    bottomy=bottomy/contour4.size();
    verdist=bottomy-topy+topleft2-topleft1;
    return 0;
}

int RectExtract::FindDatumPoint(vector<cv::Point2f> xpts, vector<cv::Point2f> ypts, vector<cv::Point2f>& datumpts){
    datumpts.resize(3);
    vector<float> toppara;
    LineFitLeastSquaresf(xpts,toppara);
    if(toppara[0]<-0.000001||toppara[0]>0.000001){
        float yres=-10000;
        for(int i=0;i<xpts.size();++i){
            float y=xpts[i].y-(xpts[i].x*toppara[0]+toppara[1]);
            if(yres<y){
                yres=y;
                datumpts[1]=xpts[i];
            }
        }
    }
    else{
        float yres=-10000;
        for(int i=0;i<xpts.size();++i){
            if(yres<xpts[i].y){
                yres=xpts[i].y;
                datumpts[1]=xpts[i];
            }
        }
    }
    for(int i=0;i<ypts.size();++i){
        cv::Point2f temp=ypts[i];
        ypts[i].x=temp.y;
        ypts[i].y=temp.x;
    }
    vector<float> leftpara;
    LineFitLeastSquaresf(ypts,leftpara);
    if(leftpara[0]<-0.000001||leftpara[0]>0.000001){
        float yres=-10000;
        for(int i=0;i<ypts.size();++i){
            float y=ypts[i].y-(ypts[i].x*leftpara[0]+leftpara[1]);
            if(yres<y){
                yres=y;
                datumpts[2]=cv::Point2f(ypts[i].y,ypts[i].x);
            }
        }
    }
    else{
        float yres=-10000;
        for(int i=0;i<ypts.size();++i){
            if(yres<ypts[i].y){
                yres=ypts[i].y;
                datumpts[2]=cv::Point2f(ypts[i].y,ypts[i].x);
            }
        }
    }
    if(toppara[0]<-0.000001||toppara[0]>0.000001){
        float k=toppara[0];
        float b=datumpts[1].y-k*datumpts[1].x;
        float b2=datumpts[2].y+datumpts[2].x/k;
        cv::Point2f topleft;
        topleft.x=(b2-b)/(k+1.0/k);
        topleft.y=topleft.x*k+b;
        datumpts[0]=topleft;
    }
    else{
        datumpts[0]=cv::Point2f(datumpts[2].x,datumpts[1].y);
    }
    return 0;
}

float RectExtract::GetParallelism(vector<cv::Point2f> contour,int flag){
    if(flag==1){//X
        float max=0;
        float min=10000;
        for(int i=0;i<contour.size();++i){
            if(max<contour[i].x){
                max=contour[i].x;
            }
            if(min>contour[i].x){
                min=contour[i].x;
            }
        }
        return max-min;
    }
    else if(flag==2){//Y
        float max=0;
        float min=10000;
        for(int i=0;i<contour.size();++i){
            if(max<contour[i].y){
                max=contour[i].y;
            }
            if(min>contour[i].y){
                min=contour[i].y;
            }
        }
        return max-min;
    }
    return -1;
}

float RectExtract::GetLineDist(vector<cv::Point2f> contour1,vector<cv::Point2f> contour2,int flag){
    if(flag==1){//X
        float sum1=0;
        float sum2=0;
        for(int i=0;i<contour1.size();++i){
            sum1+=contour1[i].x;
        }
        for(int i=0;i<contour2.size();++i){
            sum2+=contour2[i].x;
        }
        return fabs(sum1/contour1.size()-sum2/contour2.size());
    }
    else if(flag==2){//Y
        float sum1=0;
        float sum2=0;
        for(int i=0;i<contour1.size();++i){
            sum1+=contour1[i].y;
        }
        for(int i=0;i<contour2.size();++i){
            sum2+=contour2[i].y;
        }
        return fabs(sum1/contour1.size()-sum2/contour2.size());
    }
    return -1;
}

float RectExtract::GetRoundness(vector<cv::Point> contour, cv::Point2f center){
    float maxrad=0;
    float minrad=10000;
    for(int i=0;i<contour.size();++i){
        float rad=Point2Point<cv::Point,cv::Point2f>(contour[i],center);
        if(maxrad<rad){
            maxrad=rad;
        }
        if(minrad>rad){
            minrad=rad;
        }
    }
    return fabs(minrad-maxrad);
}

template<typename T1,typename T2>
float RectExtract::Point2Point(T1 A,T2 B){
    return sqrt((A.x-B.x)*(A.x-B.x)+(A.y-B.y)*(A.y-B.y));
}

float RectExtract::GetPosition(cv::Point2f stdposition,cv::Point2f measurepos){
    return 2*Point2Point<cv::Point2f,cv::Point2f>(stdposition,measurepos);
}

int RectExtract::Get2Datum616(cv::Mat srcimg,vector<cv::Point2f>& datum){
    cv::Mat top=srcimg(cv::Rect(cv::Point(2075,2316),cv::Point(2554,2446))).clone();
    cv::Mat left=srcimg(cv::Rect(cv::Point(1122,1626),cv::Point(1222,1813))).clone();
    cv::threshold(left,left,230,255,CV_THRESH_BINARY);
    vector<vector<float>> curves;
    vector<cv::Point> contourstr;
    Pixel2Curve(top,10,top.cols-10,3,curves);
    for(int i=0;i<curves.size();++i){
        for(int j=0;j<curves[i].size()-1;++j){
            if(curves[i][j]-curves[i][j+1]>15){
                contourstr.push_back(cv::Point(10+i,j));
            }
        }
    }
    for(int i=0;i<contourstr.size();++i){
        contourstr[i].x+=2075;
        contourstr[i].y+=2316;
    }
    vector<float> toppara;
    LineFitLeastSquares(contourstr,toppara);
    vector<cv::Point> contoursl;
    ScanLine(left,2,contoursl,1,5,0.2);
    for(int i=0;i<contoursl.size();++i){
        cv::Point temp=contoursl[i];
        contoursl[i].x=temp.y+1626;
        contoursl[i].y=temp.x+1122;
    }
    vector<float> leftpara;
    LineFitLeastSquares(contoursl,leftpara);
    for(int i=0;i<srcimg.rows;++i){
        cv::circle(srcimg,cv::Point(i*leftpara[0]+leftpara[1],i),3,255,-1);
    }
    for(int i=0;i<srcimg.cols;++i){
        cv::circle(srcimg,cv::Point(i,i*toppara[0]+toppara[1]),3,255,-1);
    }
    cv::Point2f topleft(0,0);
    if(toppara[0]-1/leftpara[0]<-0.000001||toppara[0]-1/leftpara[0]>0.000001){
        topleft.x=(-leftpara[1]/leftpara[0]-toppara[1])/(toppara[0]-1/leftpara[0]);
    }
    topleft.y=topleft.x*toppara[0]+toppara[1];
    cv::Point2f toppoint(0,0),leftpoint(0,0);
    toppoint.x=3800;
    toppoint.y=3800*toppara[0]+toppara[1];
    leftpoint.y=2200;
    leftpoint.x=leftpoint.y*leftpara[0]+leftpara[1];
    datum.clear();
    datum.push_back(topleft);
    datum.push_back(toppoint);
    datum.push_back(leftpoint);
    cv::circle(srcimg,toppoint,5,255,-1);
    cv::circle(srcimg,leftpoint,5,255,-1);
    cv::circle(srcimg,topleft,5,255,-1);
    cv::imwrite("res.jpg",srcimg);
    return 0;
}


int RectExtract::vectortotxt(vector<vector<float>> curves){
    ofstream outputfile;
    outputfile.open("edgecurves.txt");
    for(int i=0;i<curves.size();++i){
        for(int j=0;j<curves[i].size();++j){
            outputfile<<curves[i][j]<<" ";
        }
        outputfile<<endl;
    }
    return 0;
}

int RectExtract::FitCircleLeast(vector<cv::Point> contour, vector<float> cir){
    cir.clear();
    if (contour.size() < 3)
    {
     return -1;
    }

    double sum_x = 0.0f, sum_y = 0.0f;
    double sum_x2 = 0.0f, sum_y2 = 0.0f;
    double sum_x3 = 0.0f, sum_y3 = 0.0f;
    double sum_xy = 0.0f, sum_x1y2 = 0.0f, sum_x2y1 = 0.0f;

    int N = contour.size();
    for (int i = 0; i < N; i++)
    {
     double x = contour[i].x;
     double y = contour[i].y;
     double x2 = x * x;
     double y2 = y * y;
     sum_x += x;
     sum_y += y;
     sum_x2 += x2;
     sum_y2 += y2;
     sum_x3 += x2 * x;
     sum_y3 += y2 * y;
     sum_xy += x * y;
     sum_x1y2 += x * y2;
     sum_x2y1 += x2 * y;
    }

    double C, D, E, G, H;
    double a, b, c;

    C = N * sum_x2 - sum_x * sum_x;
    D = N * sum_xy - sum_x * sum_y;
    E = N * sum_x3 + N * sum_x1y2 - (sum_x2 + sum_y2) * sum_x;
    G = N * sum_y2 - sum_y * sum_y;
    H = N * sum_x2y1 + N * sum_y3 - (sum_x2 + sum_y2) * sum_y;
    a = (H * D - E * G) / (C * G - D * D);
    b = (H * C - E * D) / (D * D - G * C);
    c = -(a * sum_x + b * sum_y + sum_x2 + sum_y2) / N;

    cir.push_back(a / (-2));
    cir.push_back(b / (-2));
    cir.push_back(sqrt(a * a + b * b - 4 * c) / 2);
    return 0;
}

int RectExtract::getdatum001(vector<cv::Point2f> xpts, vector<cv::Point2f> ypts, vector<cv::Point2f>& datumpts){
    datumpts.resize(3);
    vector<float> toppara;
    LineFitLeastSquaresf(xpts,toppara);
    datumpts[1]=cv::Point2f(3000,3000*toppara[0]+toppara[1]);
    for(int i=0;i<ypts.size();++i){
        cv::Point2f temp=ypts[i];
        ypts[i].x=temp.y;
        ypts[i].y=temp.x;
    }
    vector<float> leftpara;
    LineFitLeastSquaresf(ypts,leftpara);
    datumpts[2]=cv::Point2f(3000*leftpara[0]+leftpara[1],3000);
    cv::Point2f topleft(0,0);
    if(fabs(toppara[0])<=0.00001){
        topleft.y=toppara[1];
        topleft.x=topleft.y*leftpara[0]+leftpara[1];
    }
    else if(fabs(leftpara[0])<=0.00001){
        topleft.x=leftpara[1];
        topleft.y=topleft.x*toppara[0]+toppara[1];
    }
    else if(toppara[0]-1/leftpara[0]<-0.000001||toppara[0]-1/leftpara[0]>0.000001){
        topleft.x=(-leftpara[1]/leftpara[0]-toppara[1])/(toppara[0]-1/leftpara[0]);
        topleft.y=topleft.x*toppara[0]+toppara[1];
    }
    datumpts[0]=topleft;
    return 0;
}
