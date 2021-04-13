#include "interpolation.h"

interpolation::interpolation()
{

}

int interpolation::imageRotation(cv::Mat srcimg, cv::Mat& dstimg, float rad){
    dstimg=cv::Mat(srcimg.rows,srcimg.cols,CV_8UC1);
    dstimg=0;
    cv::Mat rotateMatrix=(cv::Mat_<double>(2,2)<<cos(rad),-sin(rad),sin(rad),cos(rad));
    std::cout<<rotateMatrix<<std::endl;
    cv::Point rotateCenter=cv::Point(srcimg.cols/2,srcimg.rows/2);
    for(int i=0;i<dstimg.rows;++i){
        for(int j=0;j<dstimg.cols;++j){
            cv::Point post=cv::Point(j-rotateCenter.x,i-rotateCenter.y);
            cv::Point2f origin=cv::Point2f(post.x*rotateMatrix.at<double>(0,0)+post.y*rotateMatrix.at<double>(0,1),post.x*rotateMatrix.at<double>(1,0)+post.y*rotateMatrix.at<double>(1,1));
            cv::Point originpt=nearestInterpolation(origin);
            if(originpt.x+rotateCenter.x<srcimg.cols&&originpt.y+rotateCenter.y<srcimg.rows&&originpt.x+rotateCenter.x>=0&&originpt.y+rotateCenter.y>=0){
                dstimg.at<uchar>(i,j)=srcimg.at<uchar>(originpt.y+rotateCenter.y,originpt.x+rotateCenter.x);
            }
        }
    }
    return 0;
}

cv::Point interpolation::nearestInterpolation(cv::Point2f srcpt){
    int a=(int)srcpt.x;
    int b=a+1;
    int c=(int)srcpt.y;
    int d=c+1;
    std::vector<double> dists(4,0);
    dists[0]=calcEdistance(a,c,srcpt.x,srcpt.y);
    dists[1]=calcEdistance(b,c,srcpt.x,srcpt.y);
    dists[2]=calcEdistance(a,d,srcpt.x,srcpt.y);
    dists[3]=calcEdistance(b,d,srcpt.x,srcpt.y);
    int nearest=0;
    for(int i=1;i<dists.size();++i){
        if(dists[nearest]>dists[i]){
            nearest=i;
        }
    }
    switch (nearest) {
    case 0:
        return cv::Point(a,c);
        break;
    case 1:
        return cv::Point(b,c);
        break;
    case 2:
        return cv::Point(a,d);
        break;
    case 3:
        return cv::Point(b,d);
        break;
    default:
        break;
    }
    return cv::Point(0,0);
}

template<class T1,class T2>
double interpolation::calcEdistance(T1 x1,T1 y1,T2 x2,T2 y2){
    return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}

int interpolation::imageRotationbl(cv::Mat srcimg, cv::Mat& dstimg, float rad){
    dstimg=cv::Mat(srcimg.rows,srcimg.cols,CV_8UC1);
    dstimg=0;
    cv::Mat rotateMatrix=(cv::Mat_<double>(2,2)<<cos(rad),-sin(rad),sin(rad),cos(rad));
    std::cout<<rotateMatrix<<std::endl;
    cv::Point rotateCenter=cv::Point(srcimg.cols/2,srcimg.rows/2);
    for(int i=0;i<dstimg.rows;++i){
        for(int j=0;j<dstimg.cols;++j){
            cv::Point post=cv::Point(j-rotateCenter.x,i-rotateCenter.y);
            cv::Point2f origin=cv::Point2f(post.x*rotateMatrix.at<double>(0,0)+post.y*rotateMatrix.at<double>(0,1),post.x*rotateMatrix.at<double>(1,0)+post.y*rotateMatrix.at<double>(1,1));
            if((int)origin.y+rotateCenter.y>=0&&(int)origin.x+rotateCenter.x>=0&&(int)origin.y+rotateCenter.y+1<srcimg.rows&&(int)origin.x+rotateCenter.x+1<srcimg.cols){
                dstimg.at<uchar>(i,j)=bilinearInterpolation(origin,srcimg,rotateCenter);
            }
        }
    }
    return 0;
}

int interpolation::bilinearInterpolation(cv::Point2f srcpt, cv::Mat srcimg, cv::Point rotateCenter){
    std::vector<cv::Point> neighbours(4);
    neighbours[0]=cv::Point((int)(srcpt.x+rotateCenter.x),(int)(srcpt.y+rotateCenter.y));
    neighbours[1]=cv::Point((int)(srcpt.x+1+rotateCenter.x),(int)(srcpt.y+rotateCenter.y));
    neighbours[2]=cv::Point((int)(srcpt.x+rotateCenter.x),(int)(srcpt.y+1+rotateCenter.y));
    neighbours[3]=cv::Point((int)(srcpt.x+1+rotateCenter.x),(int)(srcpt.y+1+rotateCenter.y));
    std::vector<int> neighboursdata(4);
    neighboursdata[0]=srcimg.at<uchar>(neighbours[0].y,neighbours[0].x);
    neighboursdata[1]=srcimg.at<uchar>(neighbours[1].y,neighbours[1].x);
    neighboursdata[2]=srcimg.at<uchar>(neighbours[2].y,neighbours[2].x);
    neighboursdata[3]=srcimg.at<uchar>(neighbours[3].y,neighbours[3].x);
    float x1=neighboursdata[0]+(neighboursdata[1]-neighboursdata[0])*(srcpt.x+rotateCenter.x-neighbours[0].x);
    float x2=neighboursdata[2]+(neighboursdata[3]-neighboursdata[2])*(srcpt.x+rotateCenter.x-neighbours[2].x);
    float val=x1+(x2-x1)*(srcpt.y+rotateCenter.y-neighbours[0].y);
    if(val>255){
        val=255;
    }
    if(val<0){
        val=0;
    }
    return (int)val;
}

int interpolation::imageRotationbcl(cv::Mat srcimg, cv::Mat& dstimg, float rad){
    dstimg=cv::Mat(srcimg.rows,srcimg.cols,CV_8UC1);
    dstimg=0;
    cv::Mat rotateMatrix=(cv::Mat_<double>(2,2)<<cos(rad),-sin(rad),sin(rad),cos(rad));
    std::cout<<rotateMatrix<<std::endl;
    cv::Point rotateCenter=cv::Point(srcimg.cols/2,srcimg.rows/2);
    for(int i=0;i<dstimg.rows;++i){
        for(int j=0;j<dstimg.cols;++j){
            cv::Point post=cv::Point(j-rotateCenter.x,i-rotateCenter.y);
            cv::Point2f origin=cv::Point2f(post.x*rotateMatrix.at<double>(0,0)+post.y*rotateMatrix.at<double>(0,1),post.x*rotateMatrix.at<double>(1,0)+post.y*rotateMatrix.at<double>(1,1));
            if((int)origin.y+rotateCenter.y>=0&&(int)origin.x+rotateCenter.x>=0&&(int)origin.y+rotateCenter.y+1<srcimg.rows&&(int)origin.x+rotateCenter.x+1<srcimg.cols){
                dstimg.at<uchar>(i,j)=bilinearInterpolation(origin,srcimg,rotateCenter);
            }
        }
    }
    return 0;
}

int interpolation::bicubicInterpolation(cv::Point2f srcpt, cv::Mat srcimg, cv::Point rotateCenter){
    std::vector<cv::Point> neighbours(16);
    neighbours[0]=cv::Point((int)(srcpt.x+rotateCenter.x-1),(int)(srcpt.y+rotateCenter.y-1));
    neighbours[1]=cv::Point((int)(srcpt.x+rotateCenter.x),(int)(srcpt.y+rotateCenter.y-1));
    neighbours[2]=cv::Point((int)(srcpt.x+rotateCenter.x+1),(int)(srcpt.y+rotateCenter.y-1));
    neighbours[3]=cv::Point((int)(srcpt.x+rotateCenter.x+2),(int)(srcpt.y+rotateCenter.y-1));
    neighbours[4]=cv::Point((int)(srcpt.x+rotateCenter.x-1),(int)(srcpt.y+rotateCenter.y));
    neighbours[5]=cv::Point((int)(srcpt.x+rotateCenter.x),(int)(srcpt.y+rotateCenter.y));
    neighbours[6]=cv::Point((int)(srcpt.x+rotateCenter.x+1),(int)(srcpt.y+rotateCenter.y));
    neighbours[7]=cv::Point((int)(srcpt.x+rotateCenter.x+2),(int)(srcpt.y+rotateCenter.y));
    neighbours[8]=cv::Point((int)(srcpt.x+rotateCenter.x-1),(int)(srcpt.y+rotateCenter.y+1));
    neighbours[9]=cv::Point((int)(srcpt.x+rotateCenter.x),(int)(srcpt.y+rotateCenter.y+1));
    neighbours[10]=cv::Point((int)(srcpt.x+rotateCenter.x+1),(int)(srcpt.y+rotateCenter.y+1));
    neighbours[11]=cv::Point((int)(srcpt.x+rotateCenter.x+2),(int)(srcpt.y+rotateCenter.y+1));
    neighbours[12]=cv::Point((int)(srcpt.x+rotateCenter.x-1),(int)(srcpt.y+rotateCenter.y+2));
    neighbours[13]=cv::Point((int)(srcpt.x+rotateCenter.x),(int)(srcpt.y+rotateCenter.y+2));
    neighbours[14]=cv::Point((int)(srcpt.x+rotateCenter.x+1),(int)(srcpt.y+rotateCenter.y+2));
    neighbours[15]=cv::Point((int)(srcpt.x+rotateCenter.x+2),(int)(srcpt.y+rotateCenter.y+2));
    std::vector<int> neighboursdata(16);
    for(int i=0;i<neighboursdata.size();++i){
        neighboursdata[i]=srcimg.at<uchar>(neighbours[i].y,neighbours[i].x);
    }
    std::vector<double> neighboursdist(16);
    float val=0;
    for(int i=0;i<neighboursdist.size();++i){
        neighboursdist[i]=calcEdistance(neighbours[i].x,neighbours[i].y,srcpt.x+rotateCenter.x,srcpt.y+rotateCenter.y);
        val+=bicubicWeight(neighboursdist[i],0.5)*neighboursdata[i];
    }
    return (int)val;
}

float interpolation::bicubicWeight(double dist,float a){
    if(dist<=1){
        return (a+2)*dist*dist*dist-(a+3)*dist*dist+1;
    }
    else if(dist>1&&dist<2){
        return a*dist*dist*dist-5*a*dist*dist+8*a*dist-4*a;
    }
    else{
        return 0;
    }
}
