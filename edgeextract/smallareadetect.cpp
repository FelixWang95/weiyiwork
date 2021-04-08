#include "smallareadetect.h"
#include "glcm.h"

SmallAreaDetect::SmallAreaDetect()
{

}

int SmallAreaDetect::KernelSelect(cv::Mat srcimg, int unitlength, int deltathresh, vector<cv::Point>& position){
    if(srcimg.empty()||unitlength<=0||deltathresh<=0||deltathresh>=255)
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    position.clear();
    for(int i=0;i<srcimg.rows-unitlength*3;++i){
        for(int j=0;j<srcimg.cols-unitlength*3;++j){
            cv::Mat roiimg=srcimg(cv::Rect(j,i,unitlength*3,unitlength*3));
            if(KernelJudgement(roiimg, deltathresh))
                position.push_back(cv::Point(j+unitlength,i+unitlength));
        }
    }
    return 0;
}

bool SmallAreaDetect::KernelJudgement(cv::Mat kernelimg, int deltathresh){
    vector<float> areadata(9);
    areadata[0]=cv::mean(kernelimg(cv::Rect(0,0,kernelimg.cols/3,kernelimg.rows/3)))[0];
    areadata[1]=cv::mean(kernelimg(cv::Rect(kernelimg.cols/3,0,kernelimg.cols/3,kernelimg.rows/3)))[0];
    areadata[2]=cv::mean(kernelimg(cv::Rect(kernelimg.cols/3*2,0,kernelimg.cols/3,kernelimg.rows/3)))[0];
    areadata[3]=cv::mean(kernelimg(cv::Rect(0,kernelimg.rows/3,kernelimg.cols/3,kernelimg.rows/3)))[0];
    areadata[4]=cv::mean(kernelimg(cv::Rect(kernelimg.cols/3,kernelimg.rows/3,kernelimg.cols/3,kernelimg.rows/3)))[0];
    areadata[5]=cv::mean(kernelimg(cv::Rect(kernelimg.cols/3*2,kernelimg.rows/3,kernelimg.cols/3,kernelimg.rows/3)))[0];
    areadata[6]=cv::mean(kernelimg(cv::Rect(0,kernelimg.rows/3*2,kernelimg.cols/3,kernelimg.rows/3)))[0];
    areadata[7]=cv::mean(kernelimg(cv::Rect(kernelimg.cols/3,kernelimg.rows/3*2,kernelimg.cols/3,kernelimg.rows/3)))[0];
    areadata[8]=cv::mean(kernelimg(cv::Rect(kernelimg.cols/3*2,kernelimg.rows/3*2,kernelimg.cols/3,kernelimg.rows/3)))[0];
    float mean8=(areadata[0]+areadata[1]+areadata[2]+areadata[3]+areadata[5]+areadata[6]+areadata[7]+areadata[8])/8;
    for(int i=0;i<9;++i){
        if(i!=4&&fabs(areadata[i]-mean8)>4)
            return false;
    }
    for(int i=0;i<9;++i){
        if(i!=4&&areadata[i]-areadata[4]<deltathresh)
            return false;
    }
    return true;
}

int SmallAreaDetect::KernelSelectglcm(cv::Mat srcimg, int unitlength, float deltathresh, vector<cv::Point>& position){
    if(srcimg.empty()||unitlength<=0)
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    position.clear();
    //cv::GaussianBlur(srcimg,srcimg,cv::Size(3,3),2,2);
    for(int i=0;i<srcimg.rows-unitlength*3;i=i+2){
        for(int j=0;j<srcimg.cols-unitlength*3;j=j+2){
            cv::Mat roiimg=srcimg(cv::Rect(j,i,unitlength*3,unitlength*3));
            if(KernelJudgementglcm(roiimg, deltathresh))
                position.push_back(cv::Point(j+unitlength,i+unitlength));
        }
    }
    return 0;
}

bool SmallAreaDetect::KernelJudgementglcm(cv::Mat kernelimg, float deltathresh){
    GLCM glcm;
    vector<vector<float>> areadata(9);
    cv::Mat kernelimg1=kernelimg(cv::Rect(0,0,kernelimg.cols/3,kernelimg.rows/3));
    cv::Mat kernelimg2=kernelimg(cv::Rect(kernelimg.cols/3,0,kernelimg.cols/3,kernelimg.rows/3));
    cv::Mat kernelimg3=kernelimg(cv::Rect(kernelimg.cols/3*2,0,kernelimg.cols/3,kernelimg.rows/3));
    cv::Mat kernelimg4=kernelimg(cv::Rect(0,kernelimg.rows/3,kernelimg.cols/3,kernelimg.rows/3));
    cv::Mat kernelimg5=kernelimg(cv::Rect(kernelimg.cols/3,kernelimg.rows/3,kernelimg.cols/3,kernelimg.rows/3));
    cv::Mat kernelimg6=kernelimg(cv::Rect(kernelimg.cols/3*2,kernelimg.rows/3,kernelimg.cols/3,kernelimg.rows/3));
    cv::Mat kernelimg7=kernelimg(cv::Rect(0,kernelimg.rows/3*2,kernelimg.cols/3,kernelimg.rows/3));
    cv::Mat kernelimg8=kernelimg(cv::Rect(kernelimg.cols/3,kernelimg.rows/3*2,kernelimg.cols/3,kernelimg.rows/3));   
    cv::Mat kernelimg9=kernelimg(cv::Rect(kernelimg.cols/3*2,kernelimg.rows/3*2,kernelimg.cols/3,kernelimg.rows/3));
    vector<float> areamean(9);
    areamean[0]=cv::mean(kernelimg1)[0];
    areamean[1]=cv::mean(kernelimg2)[0];
    areamean[2]=cv::mean(kernelimg3)[0];
    areamean[3]=cv::mean(kernelimg4)[0];
    areamean[4]=cv::mean(kernelimg5)[0];
    areamean[5]=cv::mean(kernelimg6)[0];
    areamean[6]=cv::mean(kernelimg7)[0];
    areamean[7]=cv::mean(kernelimg8)[0];
    areamean[8]=cv::mean(kernelimg9)[0];
    for(int i=0;i<areadata.size();++i){
        if(i!=4&&(areamean[i]<40||areamean[i]>100))
            return false;
        for(int j=i+1;j<areadata.size();++j){
            if(i!=4&&j!=4&&(areamean[i]-areamean[j])*(areamean[i]-areamean[j])>50)
                return false;
        }
    }
    if(areamean[4]-(areamean[0]+areamean[1]+areamean[2]+areamean[3]+areamean[5]+areamean[6]+areamean[7]+areamean[8])/8>-5)
        return false;
    glcm.getavgfeatures(kernelimg1,areadata[0]);
    glcm.getavgfeatures(kernelimg2,areadata[1]);
    glcm.getavgfeatures(kernelimg3,areadata[2]);
    glcm.getavgfeatures(kernelimg4,areadata[3]);
    glcm.getavgfeatures(kernelimg5,areadata[4]);
    glcm.getavgfeatures(kernelimg6,areadata[5]);
    glcm.getavgfeatures(kernelimg7,areadata[6]);
    glcm.getavgfeatures(kernelimg8,areadata[7]);
    glcm.getavgfeatures(kernelimg9,areadata[8]);

    vector<float> meanvec(4,0);
    for(int i=0;i<9;++i){
        if(i!=4){
            meanvec[0]+=areadata[i][0];
            meanvec[1]+=areadata[i][1];
            meanvec[2]+=areadata[i][2];
            meanvec[3]+=areadata[i][3];
        }
    }
    meanvec[0]/=8;
    meanvec[1]/=8;
    meanvec[2]/=8;
    meanvec[3]/=8;
    vector<float> distvec(9,0);
    for(int i=0;i<distvec.size();++i){
        distvec[i]=Cosdist(meanvec, areadata[i]);
    }
    sort(distvec.rbegin(), distvec.rend());
    if(distvec[0]-distvec[1]<-deltathresh)
        return false;
//    for(int i=0;i<areadata.size();++i){
//        if(i!=4&&DistCalc(areadata[i],areadata[4])<deltathresh)
//            return false;
//    }
//    for(int i=0;i<areadata.size()-1;++i){
//        for(int j=i+1;j<areadata.size();++j){
//            if(i!=4&&j!=4&&DistCalc(areadata[i],areadata[j])>3)
//                return false;
//        }
//    }
    return true;
}

float SmallAreaDetect::DistCalc(vector<float> vec1, vector<float> vec2){
    if(vec1.size()!=vec2.size())
        return 0;
    float dist=0,sum=0;
    for(int i=0;i<vec1.size();++i){
        sum+=(vec1[i]-vec2[i])*(vec1[i]-vec2[i]);
    }
    dist=sqrt(sum);
    return dist;
}

float SmallAreaDetect::Cosdist(vector<float> vec1, vector<float> vec2){
    if(vec1.size()!=vec2.size())
        return 0;
    float dist=0;
    float vec1m=0,vec2m=0;
    for(int i=0;i<vec1.size();++i){
        vec1m+=vec1[i]*vec1[i];
        vec2m+=vec2[i]*vec2[i];
        dist+=vec1[i]*vec2[i];
    }
    dist/=(vec1m*vec2m);
    return dist;
}
