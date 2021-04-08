#include "calcfunc.h"

using namespace std;

calcfunc::calcfunc()
{

}


float calcfunc::gamma(float x){
     return x>0.04045?powf((x+0.055f)/1.055f,2.4f):(x/12.92);
}

void calcfunc::RGB2XYZ(float R,float G,float B, float& X, float& Y, float& Z){
    float RR = gamma(R/255.0);
    float GG = gamma(G/255.0);
    float BB = gamma(B/255.0);

    X = 0.4124564f * RR + 0.3575761f * GG + 0.1804375f * BB;
    Y = 0.2126729f * RR + 0.7151522f * GG + 0.0721750f * BB;
    Z = 0.0193339f * RR + 0.1191920f * GG + 0.9503041f * BB;
}

void calcfunc::XYZ2Lab(float X,float Y,float Z, float& L){
    float fX, fY, fZ;

    X /= (Xn);
    Y /= (Yn);
    Z /= (Zn);

    if (Y > 0.008856f)
    fY = pow(Y, param_13);
    else
    fY = 7.787f * Y + param_16116;

    if (X > 0.008856f)
        fX = pow(X, param_13);
    else
        fX = 7.787f * X + param_16116;

    if (Z > 0.008856)
        fZ = pow(Z, param_13);
    else
        fZ = 7.787f * Z + param_16116;

    L = 116.0f * fY - 16.0f;
    L = L > 0.0f ? L : 0.0f;
    float a = 500.0f * (fX - fY);
    float b = 200.0f * (fY - fZ);
}

float calcfunc::BrightnessLevel(vector<cv::Point> area,cv::Mat srcimg){
    if(area.size()==0||srcimg.empty()||srcimg.type()!=CV_8UC1)
        return -1;
    vector<int> ROI;
    for(int i=0;i<area.size();++i){
        ROI.push_back(srcimg.at<uchar>(area[i].y,area[i].x));
    }
    int length=ROI.size();
    int tail=length%3;
    for(int i=0;i<tail;++i){
        ROI.pop_back();
    }
    float R,G,B,X,Y,Z,L;
    R=accumulate(ROI.begin(),ROI.begin()+ROI.size()/3,0)/ROI.size()*3;
    G=accumulate(ROI.begin()+ROI.size()/3,ROI.begin()+ROI.size()*2/3,0)/ROI.size()*3;
    B=accumulate(ROI.begin()+ROI.size()*2/3,ROI.begin()+ROI.size(),0)/ROI.size()*3;
    RGB2XYZ(R,G,B,X,Y,Z);
    XYZ2Lab(X,Y,Z,L);
    return L;
}

int calcfunc::OpenCVLab(int R,int G,int B){
    cv::Mat BGRimg(1,1,CV_8UC3);
    cv::Mat Labimg;
    BGRimg.at<cv::Vec3b>(0,0)[2]=R;
    BGRimg.at<cv::Vec3b>(0,0)[1]=G;
    BGRimg.at<cv::Vec3b>(0,0)[0]=B;
    cv::cvtColor(BGRimg,Labimg,CV_BGR2Lab);
    return Labimg.at<cv::Vec3b>(0,0)[0];
}

int calcfunc::ReadPointNum(std::string filename,vector<int>& pointnum){
    cv::String pattern_txt;
    std::vector<cv::String> txt_files;
    pattern_txt = filename+"/*val0.txt";
    cv::glob(pattern_txt, txt_files);
    for(int num=0;num<txt_files.size();++num)//read all txts in the folder
    {
        cout << txt_files[num] << endl;
        FILE *pFile;
        int c;
        int n=0;
        const char* c_s = txt_files[num].c_str();
        pFile=fopen(c_s,"r");
        if (pFile==NULL){
        return -1;
        }
        else{
        do{
        c=fgetc(pFile);
        if (c==' ')
        n++;
        }while(c!=EOF);
        }
        pointnum.push_back(n);
    }
    ofstream outfile("pointnum.txt", ios::trunc);
    for(int i=0;i<pointnum.size();++i){
        outfile<<pointnum[i]<<endl;
    }
    outfile.close();
    return 0;
}

int calcfunc::DilateThresh(cv::Mat srcimg, vector<cv::Point> inputarea, vector<cv::Point>& outputarea){
    if(srcimg.empty()||inputarea.size()==0||srcimg.type()!=CV_8UC1)
        return -1;
    outputarea.clear();
    while(inputarea.size()<0.7*sqrt(srcimg.cols*srcimg.cols+srcimg.rows*srcimg.rows)){
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
        cv::dilate(srcimg,srcimg,kernel);
        for(int i=0;i<srcimg.rows;++i){
            for(int j=0;j<srcimg.cols;++j){
                if(srcimg.at<uchar>(i,j)==255){
                    outputarea.push_back(cv::Point(j,i));
                }
            }
        }
        if(outputarea.size()>=0.7*sqrt(srcimg.cols*srcimg.cols+srcimg.rows*srcimg.rows))
            break;
    }
    if(outputarea.size()==0){
        outputarea=inputarea;
    }
    return 0;
}
