#include<opencv2/opencv.hpp>
#include<iostream>
#include<omp.h>
#include "rectextract.h"
#include <time.h>
#include "findlines.h"
#include "imagecheck.h"
#include "findoutercontour.h"
#include "ResultEvaluation.h"
#include "knntraining.h"
#include "physicalquantity.h"
#include "smallareadetect.h"
#include "movingleastsquare.h"
#include "lof.h"
#include "edgedetect.h"
#include "curvefitting.h"

using namespace std;
using namespace cv;

//计算灰度直方图
Mat calcGrayHist(const Mat & image)
{
    //存储 256 个灰度级的像素数
    Mat histogram = Mat::zeros(Size(256, 1), CV_32SC1);
    int rows = image.rows;
    int cols = image.cols;
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            int index = int( image.at<uchar>(r, c));
            histogram.at<int>(0, index) += 1;
        }
    }
    return histogram;
}
//Otsu 阈值处理：得到阈值处理后的二值图 OtsuThreshImage，并返回分割阈值
int otsu(const Mat & image, Mat &OtsuThreshImage)
{
    //计算灰度直方图
    Mat histogram = calcGrayHist(image);
    //归一化灰度直方图
    Mat normHist;
    histogram.convertTo(normHist, CV_32FC1, 1.0 / (image.rows*image.cols), 0.0);
    //计算累加直方图(零阶累加矩)和一阶累加矩
    Mat zeroCumuMoment = Mat::zeros(Size(256, 1), CV_32FC1);
    Mat oneCumuMoment = Mat::zeros(Size(256, 1), CV_32FC1);
    for (int i = 0; i < 256; i++)
    {
        if (i == 0)
        {
            zeroCumuMoment.at<float>(0, i) = normHist.at<float>(0, i);
            oneCumuMoment.at<float>(0, i) = i*normHist.at<float>(0, i);
        }
        else
        {
            zeroCumuMoment.at<float>(0, i) = zeroCumuMoment.at<float>(0, i-1)+ normHist.at<float>(0, i);
            oneCumuMoment.at<float>(0, i) = oneCumuMoment.at<float>(0,i-1) + i*normHist.at<float>(0, i);
        }
    }
    //计算类间方差
    Mat variance = Mat::zeros(Size(256, 1), CV_32FC1);
    //总平均值
    float mean = oneCumuMoment.at<float>(0, 255);
    for (int i = 0; i < 255; i++)
    {
        if (zeroCumuMoment.at<float>(0, i) == 0 || zeroCumuMoment.at<float>(0, i) == 1)
            variance.at<float>(0, i) = 0;
        else
        {
            float cofficient = zeroCumuMoment.at<float>(0, i)*(1.0 - zeroCumuMoment.at<float>(0, i));
            variance.at<float>(0, i) = pow((double)mean*zeroCumuMoment.at<float>(0, i) - oneCumuMoment.at<float>(0, i), 2.0) / cofficient;
        }
    }
    //找到阈值
    Point maxLoc;
    minMaxLoc(variance, NULL, NULL, NULL, &maxLoc);
    int otsuThresh = maxLoc.x;
    //阈值处理
    threshold(image, OtsuThreshImage, otsuThresh, 255, THRESH_BINARY);
    return otsuThresh;
}

void txt_to_vectorint(vector<vector<int>>& res, string pathname)
{
    ifstream infile;
    infile.open(pathname.data());   //将文件流对象与文件连接起来
    assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行
    vector<int> suanz;
    string s;
    while (getline(infile, s)) {
        istringstream is(s); //将读出的一行转成数据流进行操作
        int d;
        while (!is.eof()) {
            is >> d;
            suanz.push_back(d);
        }
        res.push_back(suanz);
        suanz.clear();
        s.clear();
    }
    infile.close();             //关闭文件输入流
}

void txt_to_vectorfloat(vector<vector<float>>& res, string pathname)
{
    ifstream infile;
    infile.open(pathname.data());   //将文件流对象与文件连接起来
    assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行
    vector<float> suanz;
    string s;
    while (getline(infile, s)) {
        istringstream is(s); //将读出的一行转成数据流进行操作
        float d;
        while (!is.eof()) {
            is >> d;
            suanz.push_back(d);
        }
        res.push_back(suanz);
        suanz.clear();
        s.clear();
    }
    infile.close();             //关闭文件输入流
}

int getleftcircle(){
    vector<vector<int>> templatepara;
    vector<int> temppara1{195,984,619,350,145,920,1110};
    templatepara.push_back(temppara1);
    vector<int> temppara2{180,1240,767,315,115,920,1110};
    templatepara.push_back(temppara2);
    vector<int> temppara3{210,1238,1270,260,50,920,1110};
    templatepara.push_back(temppara3);
    vector<int> temppara4{190,988,1493,220,10,920,1110};
    templatepara.push_back(temppara4);
    vector<int> temppara5{210,590,1493,175,320,920,1110};
    templatepara.push_back(temppara5);
    vector<int> temppara6{190,405,1264,130,290,920,1110};
    templatepara.push_back(temppara6);
    vector<int> temppara7{200,409,764,70,225,920,1110};
    templatepara.push_back(temppara7);
    vector<int> temppara8{175,585,620,40,185,920,1110};
    templatepara.push_back(temppara8);
    clock_t start, finish;
    double  duration;
    start = clock();
    RectExtract *rectex= new RectExtract();
    vector<cv::Mat> srcimgs;
    cv::Mat src1=cv::imread("/mnt/hgfs/linuxsharefiles/capimg/classifiedimg/0_L/0_L1/classify3/2/35588-1-1-11.jpg",0);
    srcimgs.push_back(src1);
    cv::Mat src2=cv::imread("/mnt/hgfs/linuxsharefiles/capimg/classifiedimg/0_L/0_L2/classify4/0/35588-1-1-11.jpg",0);
    srcimgs.push_back(src2);
    cv::Mat src3=cv::imread("/mnt/hgfs/linuxsharefiles/capimg/classifiedimg/0_L/0_L3/classify3/0/35588-1-1-11.jpg",0);
    srcimgs.push_back(src3);
    cv::Mat src4=cv::imread("/mnt/hgfs/linuxsharefiles/capimg/classifiedimg/0_L/0_L4/classify3/2/35588-1-1-11.jpg",0);
    srcimgs.push_back(src4);
    cv::Mat src5=cv::imread("/mnt/hgfs/linuxsharefiles/capimg/classifiedimg/0_L/0_L5/classify3/1/35588-1-1-11.jpg",0);
    srcimgs.push_back(src5);
    cv::Mat src6=cv::imread("/mnt/hgfs/linuxsharefiles/capimg/classifiedimg/0_L/0_L6/classify6/1/35588-1-1-11.jpg",0);
    srcimgs.push_back(src6);
    cv::Mat src7=cv::imread("/mnt/hgfs/linuxsharefiles/capimg/classifiedimg/0_L/0_L7/classify3/0/35588-1-1-11.jpg",0);
    srcimgs.push_back(src7);
    cv::Mat src8=cv::imread("/mnt/hgfs/linuxsharefiles/capimg/classifiedimg/0_L/0_L8/classify5/4/35588-1-1-11.jpg",0);
    srcimgs.push_back(src8);
    vector<cv::Point> contours;
    vector<cv::Point> leftcircle;
    for(int i=0;i<templatepara.size();++i){
        contours.clear();
        rectex->KirschEdgeCircle(srcimgs[i],templatepara[i][0], templatepara[i][3],templatepara[i][4],contours);
        for(int j=0;j<contours.size();++j){
            leftcircle.push_back(cv::Point(contours[j].x+templatepara[i][1],contours[j].y+templatepara[i][2]));
        }
    }
    vector<float> leftcirclepara;
    leftcirclepara=rectex->FitCircle(leftcircle);
    cv::Mat wholeimg=cv::imread("/mnt/hgfs/linuxsharefiles/newimages/totalimage/35588-1-1-11.jpg",0);
    cv::circle(wholeimg,cv::Point(leftcirclepara[0],leftcirclepara[1]),5,255,-1);
    for(int i=0;i<leftcircle.size();++i){
        cv::circle(wholeimg,leftcircle[i],1,255,-1);
    }
    cv::imwrite("circle1.jpg",wholeimg);
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout<<duration<<endl;
    cv::waitKey(0);
    return 0;
}

int getrectlines(){
    clock_t start, finish;
    double  duration;
    start = clock();
    RectExtract *rectex= new RectExtract();
    std::vector<cv::String> image_files;
    std::string filepath="/mnt/hgfs/linuxsharefiles/testImages/smallPartsMatch/s_matchResult/S_Rect/s_S2/*.jpg";
    cv::glob(filepath, image_files);
    for(int i=0;i<image_files.size();++i){
        cv::Mat rect1=cv::imread(image_files[i],0);
        vector<float> dists;
        rectex->GetRectEdge(rect1,120,2,dists);
        for(int j=0;j<dists.size();++j){
            cout<<dists[j]<<endl;
        }
    }
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
//    cout<<duration<<endl;
    cv::waitKey(0);
    return 0;
}

int getrightcircle(){
    vector<vector<int>> templatepara;
    vector<int> temppara1{200,90,45,60,225,240,1,2468,594};
    templatepara.push_back(temppara1);
    vector<int> temppara2{180,130,50,65,230,245,2,2887,954};
    templatepara.push_back(temppara2);
    vector<int> temppara3{200,115,60,75,245,260,3,2458,1377};
    templatepara.push_back(temppara3);
    vector<int> temppara4{210,125,60,75,240,255,4,2088,947};
    templatepara.push_back(temppara4);
    clock_t start, finish;
    double  duration;
    start = clock();
    RectExtract *rectex= new RectExtract();
    vector<cv::Mat> srcimgs;
    cv::Mat src1=cv::imread("/mnt/hgfs/linuxsharefiles/capimg/classifiedimg/0_R/0_R1/classify5/1/35588-1-1-11.jpg",0);
    srcimgs.push_back(src1);
    cv::Mat src2=cv::imread("/mnt/hgfs/linuxsharefiles/capimg/classifiedimg/0_R/0_R2/classify6/1/35588-1-1-11.jpg",0);
    srcimgs.push_back(src2);
    cv::Mat src3=cv::imread("/mnt/hgfs/linuxsharefiles/capimg/classifiedimg/0_R/0_R3/classify3/1/35588-1-1-11.jpg",0);
    srcimgs.push_back(src3);
    cv::Mat src4=cv::imread("/mnt/hgfs/linuxsharefiles/capimg/classifiedimg/0_R/0_R4/classify3/2/35588-1-1-11.jpg",0);
    srcimgs.push_back(src4);
    vector<cv::Point> contours;
    vector<cv::Point> rightcirclecontours;
    for(int i=0;i<templatepara.size();++i){
        contours.clear();
        std::vector<int> vec(templatepara[i].begin()+2,templatepara[i].begin()+6);
        rectex->KirschEdgeLine(srcimgs[i],templatepara[i][0],templatepara[i][6],contours);
        cv::Mat top=srcimgs[i].clone();
        for(int j=0;j<contours.size();++j){
            cv::circle(top,contours[j],1,255,-1);
        }
        cv::imwrite("right"+std::to_string(i)+".jpg",top);
        for(int j=0;j<contours.size();++j){
            rightcirclecontours.push_back(cv::Point(contours[j].x+templatepara[i][7],contours[j].y+templatepara[i][8]));
        }
    }
    vector<float> rightcirclepara;
    rightcirclepara=rectex->FitCircle(rightcirclecontours);
    cv::Mat wholeimg=cv::imread("/mnt/hgfs/linuxsharefiles/newimages/totalimage/35588-1-1-11.jpg",0);
    cv::circle(wholeimg,cv::Point(rightcirclepara[0],rightcirclepara[1]),5,255,-1);
    for(int i=0;i<rightcirclecontours.size();++i){
        cv::circle(wholeimg,rightcirclecontours[i],1,255,-1);
    }
    cv::imwrite("rightcircle1.jpg",wholeimg);
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout<<duration<<endl;
    cv::waitKey(0);
    return 0;
}

int getsmallcircle(){
    ofstream circlecenter;
    circlecenter.open("circlecenter160SS7.txt");
    clock_t start, finish;
    double  duration;
    start = clock();
    RectExtract *rectex= new RectExtract();
    std::vector<cv::String> image_files;
    std::string filepath="/mnt/hgfs/linuxsharefiles/testImages/smallPartsMatch/s_matchResult/s_SS7/*.jpg";
    cv::glob(filepath, image_files);
    for(int i=0;i<image_files.size();++i){
        cv::Mat leftcircle1=cv::imread(image_files[i],0);
        cv::Mat testarea=leftcircle1.clone();
//        vector<cv::Point> contours;
//        rectex->KirschEdgeSmallCircle(testarea,160,7,contours);
        vector<float> circlepara;
//        circlepara=rectex->FitCircle(contours);
        rectex->GetSmallCircle(testarea,160,7,circlepara);
        circlecenter<<circlepara[0]<<"  "<<circlepara[1]<<endl;
        cout<<circlepara[0]<<"  "<<circlepara[1]<<endl;
//        if(i==0){
//            cv::Mat top=leftcircle1.clone();
//            for(int j=0;j<contours.size();++j){
//                cv::circle(top,contours[j],1,255,-1);
//            }
//            cv::circle(top,cv::Point(circlepara[0],circlepara[1]),3,255,-1);
//            cv::imshow("top",top);
//        }
    }
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout<<duration<<endl;
    //cv::imwrite("smallcircle.jpg",top);
    circlecenter.close();
    cv::waitKey(0);
    return 0;
}

int getdatums(){
    RectExtract *rectex= new RectExtract();
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/35588-1-5-13.jpg",0);
    vector<cv::Point2f> datumpoints;
    rectex->GetDatum(srcimg,datumpoints);
    for(int i=0;i<datumpoints.size();++i){
        int x=datumpoints[i].x;
        int y=datumpoints[i].y;
        cout<<x<<" "<<y<<endl;
        cv::circle(srcimg,cv::Point(x,y),10,255,-1);
    }
    cv::line(srcimg, datumpoints[0], datumpoints[2], 255, 2);
    cv::line(srcimg, datumpoints[1], datumpoints[2], 255, 2);
    cout<<rectex->Point2Line(cv::Point(1000,1000),datumpoints,0)<<endl;
    cout<<rectex->Point2Line(cv::Point(1000,1000),datumpoints,1)<<endl;
    cv::imwrite("datumimg.jpg",srcimg);
    cv::waitKey(0);
    return 0;
}

int gettopline(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/001.jpg",0);
    cv::Mat testarea=srcimg(cv::Rect(cv::Point(1806,56),cv::Point(2702,309)));
    RectExtract *rectex= new RectExtract();
    vector<cv::Point> contours;
    vector<int> segments{50,2702-1806-50};
    rectex->KirschEdgeInnerLine(testarea,200,3,138,segments,contours);
    cv::Mat lines=testarea.clone();
    for(int i=0;i<contours.size();++i){
        cv::circle(testarea,contours[i],1,255,-1);
    }
    vector<vector<cv::Point>> edgecontours;
    edgecontours.push_back(contours);
    rectex->FitLineAndDraw(lines,lines,edgecontours);
    cv::imshow("lines",lines);
    cv::imshow("testarea",testarea);
    cv::waitKey(0);
    return 0;
}

int getInnerLine(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/001.jpg",0);
    cv::Mat testarea=srcimg(cv::Rect(cv::Point(2330,2498),cv::Point(2399,2751)));
    RectExtract *rectex= new RectExtract();
    vector<cv::Point> contours;
    vector<int> segments{10,2751-2498-10};
    rectex->KirschEdgeInnerLine(testarea,200,2,0,segments,contours);
    for(int i=0;i<contours.size();++i){
        cv::circle(testarea,contours[i],1,255,-1);
    }
    cv::imshow("testarea",testarea);
    cv::waitKey(0);
    return 0;
}

int getGapPoint(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/0.png",0);
    RectExtract *rectex= new RectExtract();
    vector<cv::Point> seedpoints;
    rectex->GetDarkGap(srcimg,seedpoints);
    for(int i=0;i<seedpoints.size();++i){
        cv::circle(srcimg,seedpoints[i],1,255,-1);
    }
    cv::imshow("dst",srcimg);
    cv::waitKey(0);
    return 0;
}

int GetEdgeOuter(){
    RectExtract *rectex= new RectExtract();
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/smallPartsMatch/s_matchResult/SE/s_SE4/35588-1-1-13.jpg",0);
//    std::vector<cv::String> image_files;
//    std::string filepath="/mnt/hgfs/linuxsharefiles/smallPartsMatch/s_matchResult/SE/s_SE3/*.jpg";
//    cv::glob(filepath, image_files);
//    for(int i=0;i<image_files.size();++i){
//        cv::Mat srcimg=cv::imread(image_files[i],0);
        //cout<<image_files[i]<<endl;
        float dist;
        rectex->KirschEdgeOuter(srcimg,180,4,dist);
        cout<<dist<<endl;
//    }
    cv::waitKey(0);
    return 0;
}

int GetLines(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/model.png",0);
    FindLines fl;
    vector<cv::Point> contours;
    fl.GetLines(srcimg,contours,0);
    cv::waitKey(0);
}

int ImgCheck(){
    clock_t start, finish;
    double  duration;
    start = clock();
    ImageCheck IC;
    std::vector<cv::String> image_files;
    std::string filepath="/mnt/hgfs/linuxsharefiles/testImages/images/1-11/*.jpg";
    cv::glob(filepath, image_files);
    for(int i=0;i<image_files.size();++i){
        cv::Mat srcimg=cv::imread(image_files[i],0);
        cout<<IC.GetBlackArea(srcimg,40)<<endl;
    }
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout<<duration<<endl;
    cv::waitKey(0);
    return 0;
}

int FindOuterContours(){
    clock_t start, finish;
    double  duration;
    start = clock();
    ImageCheck IC;
    std::vector<cv::String> image_files;
    std::string filepath="/mnt/hgfs/linuxsharefiles/testImages/images/totalimage/*.jpg";
    cv::glob(filepath, image_files);
    cv::Mat histMat,angleMat;
    for(int i=0;i<image_files.size();++i){
        cv::Mat srcimg=cv::imread(image_files[i],0);
        vector<cv::Point> contour;
        IC.GetContour(srcimg,contour);
        vector<float> angles;
        IC.Contour2Angle(contour,6,6,angles);
        if(angles.size()>58){
            angles.erase(angles.begin()+58,angles.end());
        }
        else if(angles.size()<58){
            int as=angles.size();
            for(int j=0;j<58-as;++j){
                angles.push_back(0.0);
            }
        }
        cv::Mat contourangles;
        IC.Vecf2Mat32F(angles,contourangles);
        cv::Mat hist;
        IC.CalcHistogram(srcimg,0,255,hist);
        if(i==0){
            histMat=hist;
            angleMat=contourangles;
        }
        else{
            cv::vconcat(histMat,hist,histMat);
            cv::vconcat(angleMat,contourangles,angleMat);
        }
    }

    cout<<histMat.size()<<endl;
    cout<<angleMat.size()<<endl;
    //IC.OutputMat("angleMat.txt",angleMat);
    cv::Mat histoutput,angleoutput;
    IC.PCADecreaseDim(histMat,histoutput, 0.95);
    cv::Ptr<cv::ml::SVM> svm=IC.OneClassSVMmodel(histoutput,0.3,0.1,"hist");
//    cv::Mat histreslabels;
//    IC.TestSVMmodel(svm,histoutput,histreslabels);
//    cout<<histreslabels<<endl;
    IC.PCADecreaseDim(angleMat,angleoutput, 0.95);
    cv::Ptr<cv::ml::SVM> anglesvm=IC.OneClassSVMmodel(angleoutput,0.9,0.1,"angle");
//    cv::Mat anglereslabels;
//    IC.TestSVMmodel(anglesvm,angleoutput,anglereslabels);
//    cout<<anglereslabels<<endl;
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout<<duration<<endl;
    //cv::waitKey(0);
    return 0;
}

int FindhistSVMpara(){
    clock_t start, finish;
    double  duration;
    start = clock();
    ImageCheck IC;
    std::vector<cv::String> image_files;
    std::string filepath="/mnt/hgfs/linuxsharefiles/testImages/images/totalimage/*.jpg";
    cv::glob(filepath, image_files);
    cv::Mat histMat;
    for(int i=0;i<image_files.size();++i){
        cv::Mat srcimg=cv::imread(image_files[i],0);
        cv::Mat hist;
        IC.CalcHistogram(srcimg,0,255,hist);
        if(i==0){
            histMat=hist;
        }
        else{
            cv::vconcat(histMat,hist,histMat);
        }
    }
    cout<<histMat.size()<<endl;
    float fp,fn;
    IC.FindHistSVM(histMat,fp,fn);
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout<<duration<<endl;
    cv::waitKey(0);
    return 0;
}

int FindangleSVMpara(){
    clock_t start, finish;
    double  duration;
    start = clock();
    ImageCheck IC;
    std::vector<cv::String> image_files;
    std::string filepath="/mnt/hgfs/linuxsharefiles/testImages/images/totalimage/*.jpg";
    cv::glob(filepath, image_files);
    cv::Mat angleMat;
    for(int i=0;i<image_files.size();++i){
        cv::Mat srcimg=cv::imread(image_files[i],0);
        vector<cv::Point> contour;
        IC.GetContour(srcimg,contour);
        vector<float> angles;
        IC.Contour2Angle(contour,6,6,angles);
        if(angles.size()>58){
            angles.erase(angles.begin()+58,angles.end());
        }
        else if(angles.size()<58){
            int as=angles.size();
            for(int j=0;j<58-as;++j){
                angles.push_back(0.0);
            }
        }
        cv::Mat contourangles;
        IC.Vecf2Mat32F(angles,contourangles);
        if(i==0){
            angleMat=contourangles;
        }
        else{
            cv::vconcat(angleMat,contourangles,angleMat);
        }
    }
    cout<<angleMat.size()<<endl;
    float fp,fn;
    IC.FindEdgeSVM(angleMat,fp,fn);
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout<<duration<<endl;
    cv::waitKey(0);
    return 0;
}

int generatewrongimg(){
    clock_t start, finish;
    double  duration;
    start = clock();
    ImageCheck IC;
    std::vector<cv::String> image_files;
    std::string filepath="/mnt/hgfs/linuxsharefiles/testImages/images/1-11/*.jpg";
    cv::glob(filepath, image_files);
    for(int i=0;i<1;++i){
        cv::Mat srcimg=cv::imread(image_files[i],0);
        cv::imwrite("/mnt/hgfs/linuxsharefiles/testImages/images/1-11w/"+std::to_string(i)+".jpg",IC.GenerateWrongImage(srcimg,0.5));
    }
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout<<duration<<endl;
    cv::waitKey(0);
    return 0;
}

int testimage(){
    clock_t start, finish;
    double  duration;
    start = clock();
    ImageCheck IC;
    std::vector<cv::String> image_files;
    std::string filepath="/mnt/hgfs/linuxsharefiles/testImages/images/1-11w/*.jpg";
    cv::glob(filepath, image_files);
    cv::Mat histMat,angleMat;
    for(int i=0;i<image_files.size();++i){
        cv::Mat srcimg=cv::imread(image_files[i],0);
        vector<cv::Point> contour;
        IC.GetContour(srcimg,contour);
        vector<float> angles;
        IC.Contour2Angle(contour,6,6,angles);
        if(angles.size()>58){
            angles.erase(angles.begin()+58,angles.end());
        }
        else if(angles.size()<58){
            int as=angles.size();
            for(int j=0;j<58-as;++j){
                angles.push_back(0.0);
            }
        }
        cv::Mat contourangles;
        IC.Vecf2Mat32F(angles,contourangles);
        cv::Mat hist;
        IC.CalcHistogram(srcimg,0,255,hist);
        if(i==0){
            histMat=hist;
            angleMat=contourangles;
        }
        else{
            cv::vconcat(histMat,hist,histMat);
            cv::vconcat(angleMat,contourangles,angleMat);
        }
    }
    cout<<histMat.size()<<endl;
    cout<<angleMat.size()<<endl;
    cv::Mat histoutput,angleoutput;
    histoutput=IC.PCAdecrease(IC.ReadPCAFromXML("192255pca.xml"),histMat);
    cout<<histoutput<<endl;
    cv::Ptr<cv::ml::SVM> svm=IC.ReadSVMFromXML("hist_svm.xml");
    cv::Mat histreslabels;
    IC.TestSVMmodel(svm,histoutput,histreslabels);
    cout<<histreslabels<<endl;
    angleoutput=IC.PCAdecrease(IC.ReadPCAFromXML("19258pca.xml"),angleMat);
    cout<<angleoutput<<endl;
    cv::Ptr<cv::ml::SVM> anglesvm=IC.ReadSVMFromXML("angle_svm.xml");
    cv::Mat anglereslabels;
    IC.TestSVMmodel(anglesvm,angleoutput,anglereslabels);
    cout<<anglereslabels<<endl;
    std::vector<int> realresult(80),preresult(80);
    for(int i=0;i<realresult.size();++i){
        if(i<40){
            realresult[i]=-1;
        }
        else{
            realresult[i]=1;
        }
    }
    for(int i=0;i<histreslabels.rows;++i){
        if(histreslabels.at<float>(i,0)>0.0001&&anglereslabels.at<float>(i,0)>0.0001){
            preresult[i]=1;
        }
        else{
            preresult[i]=-1;
        }
    }
    ResultEvaluation reseva;
    CalcResult res=reseva(realresult,preresult);
    cout<<res.F1_p<<"    "<<res.F1_n<<endl;
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout<<duration<<endl;
    cv::waitKey(0);
    return 0;
}

int GetOptimalSVM(){
    FindOuterContours();
    return 0;
}

int HistKnn(){
    KNNtraining knn;
    ImageCheck IC;
    cv::String trainpath="/mnt/hgfs/linuxsharefiles/testImages/images/totalimage/*.jpg";
    cv::String testgoodpath="/mnt/hgfs/linuxsharefiles/testImages/images/good/*.jpg";
    cv::String testbadpath="/mnt/hgfs/linuxsharefiles/testImages/images/bad/*.jpg";
    cv::Mat traingMat=IC.HistExtract(trainpath,5);
    cv::Mat testgoodMat=IC.HistExtract(testgoodpath,5);
    cv::Mat testbadMat=IC.HistExtract(testbadpath,5);
    //cout<<knn.disttest(traingMat)<<endl;
    vector<float> res;
    knn.KNNfunc(traingMat,testgoodMat,testbadMat,0.0278, res);
    for(int i=0;i<res.size();++i)
        cout<<res[i]<<endl;
    cv::waitKey(0);
    return 0;
}

int EdgeKnn(){
    KNNtraining knn;
    ImageCheck IC;
    cv::String trainpath="/mnt/hgfs/linuxsharefiles/testImages/images/totalimage/*.jpg";
    cv::String testgoodpath="/mnt/hgfs/linuxsharefiles/testImages/images/good/*.jpg";
    cv::String testbadpath="/mnt/hgfs/linuxsharefiles/testImages/images/bad/*.jpg";
    cv::Mat traingMat=IC.EdgeExtract(trainpath);
    cv::Mat testgoodMat=IC.EdgeExtract(testgoodpath);
    cv::Mat testbadMat=IC.EdgeExtract(testbadpath);
    //cout<<knn.disttest(traingMat)<<endl;
    vector<float> res;
    knn.KNNfunc(traingMat,testgoodMat,testbadMat,1.533,res);
    for(int i=0;i<res.size();++i)
        cout<<res[i]<<endl;
    cv::waitKey(0);
    return 0;
}

int GenerateHistFeature(){
    ImageCheck IC;
    cv::String trainpath="/mnt/hgfs/linuxsharefiles/testImages/images/totalimage/*.jpg";
    cv::Mat traingMat=IC.HistExtract(trainpath,5);
    IC.SaveXml("/home/adt/QTcode/edgeextract/KnnTrainingData.xml",traingMat);
    return 0;
}

int TestKnnHist(){
    cv::Mat testimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/bad/1896-0001-11.jpg",0);
    cout<<testimg.size()<<endl;
    clock_t start, finish;
    double  duration;
    start = clock();
    KNNtraining knn;
    cout<<knn.ImageCheck(testimg)<<endl;
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout<<duration<<endl;
    cv::waitKey(0);
    return 0;
}

int GetDefectOuterContour(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/pq/defect5.png",0);
    FindOuterContour FC;
    vector<float> contourx,contoury;
    FC.PeripheralTraversa(srcimg,contourx,contoury);
    vector<cv::Point> contour(contourx.size());
    for(int i=0;i<contourx.size();++i){
        cv::Point temp;
        temp.x=contourx[i]*2;
        temp.y=contoury[i]*2;
        contour[i]=temp;
    }
    cv::Mat pointimg=srcimg.clone();
    int border=10;
    cv::copyMakeBorder(pointimg,pointimg,border,border,border,border,cv::BORDER_CONSTANT,0);
    cv::imwrite("expandedimg.png",pointimg);
    ofstream pointfile;
    pointfile.open("contourpoints.txt");
    for(int i=0;i<contour.size();++i){
        cv::circle(pointimg,cv::Point(contour[i].x+border,contour[i].y+border),1,255,-1);
        pointfile<<contour[i].x+border<<" "<<contour[i].y+border<<endl;
    }
    cv::imshow("contourpoints",pointimg);
    cv::waitKey(0);
    return 0;
}

int DrawDefectOuterContour(){
    cv::Mat srcimg=cv::imread("/home/adt/edgeextract/expandedimg.png",0);
    cv::Mat ployimg=srcimg.clone();
    vector<vector<int>> contourpoints;
    txt_to_vectorint(contourpoints,"/home/adt/edgeextract/contourpoints.txt");
    vector<cv::Point> contour(contourpoints.size());
    for(int i=0;i<contourpoints.size();++i){
        cv::Point temp(contourpoints[i][0],contourpoints[i][1]);
        contour[i]=temp;
        cv::circle(srcimg,temp,1,255,-1);
    }
    vector<cv::Point> contourploy;
    cv::approxPolyDP(contour, contourploy, 10, true);
    for(int i=0;i<contourploy.size();++i){
        cv::circle(ployimg,contourploy[i],1,255,-1);
        cv::putText(ployimg, std::to_string(i), contourploy[i], cv::FONT_HERSHEY_TRIPLEX, 0.4, cv::Scalar(255, 200, 200), 1, CV_AA);
    }
    cv::imshow("srcimg",srcimg);
    cv::imshow("ployimg",ployimg);
    cv::waitKey(0);
    return 0;
}

int openmptest(){
    vector<int> vecInt(100);
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < vecInt.size(); ++i)
    {
        vecInt[i] = i*i;
        cout<<vecInt[i]<<endl;
    }
    int mpenable=1;
    #pragma omp parallel num_threads(2) if(mpenable)
    {
        cout << "parallel run!!!\n";
    }
    return 0;
}

int pockmarksdetect(){
    SmallAreaDetect sad;
    int length=6;
    std::vector<cv::String> image_files;
    std::string filepath="/mnt/hgfs/linuxsharefiles/testImages/images/pockmarksok/*.jpg";
    cv::glob(filepath, image_files);
    for(int index=0;index<image_files.size();++index){
        cv::Mat srcimg=cv::imread(image_files[index],0);
        vector<cv::Point> position;
        clock_t start, finish;
        double  duration;
        start = clock();
        sad.KernelSelectglcm(srcimg,length,0.3,position);
        finish = clock();
        duration = (double)(finish - start) / CLOCKS_PER_SEC;
        cout<<duration<<endl;
        for(int i=0;i<position.size();++i){
            //cout<<position[i]<<endl;
            cv::circle(srcimg,cv::Point(position[i].x+length/2,position[i].y+length/2),1,255,-1);
        }
        cv::imwrite("/mnt/hgfs/linuxsharefiles/testImages/images/pockmarksokresult/resmadian"+std::to_string(index)+".jpg",srcimg);
//        cv::waitKey(0);
    }
    return 0;
}

int patent(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/patent/0948-0011-14.jpg",0);
    FindOuterContour foc;
    cv::Mat dstimg;
    foc.PerspectiveTransformation(srcimg,dstimg);
    cv::imwrite("PTdstimgg.jpg",dstimg);
    ImageCheck ic;
    vector<float> curve;
    ic.KernelCurve(dstimg,3,360,curve);
    ofstream filetxt;
    filetxt.open("curvebrightnessg.txt");
    for(int i=0;i<curve.size();++i){
        filetxt<<curve[i]<<endl;
    }
//    RectExtract er;
//    FindLines fl;
//    cv::Mat edgeup=srcimg(cv::Rect(cv::Point(1018,137),cv::Point(3550,391))).clone();
//    cv::Mat edgeleft=srcimg(cv::Rect(cv::Point(143,909),cv::Point(809,5121))).clone();
//    cv::Mat edgedown=srcimg(cv::Rect(cv::Point(1145,5521),cv::Point(3609,5889))).clone();
//    cv::Mat edgeright=srcimg(cv::Rect(cv::Point(3921,865),cv::Point(4377,5073))).clone();
//    vector<cv::Point> contour;
//    er.KirschEdgeInnerLine(edgeup,50,3,0,vector<int>{10,edgeup.cols-10},contour);
//    for(int i=0;i<contour.size();++i){
//        cv::Point temp(contour[i].x+1018,contour[i].y+137);
//        contour[i]=temp;
//    }
//    vector<float> topline;
//    fl.LineFitLeastSquares(contour,topline);
//    cv::Point startpt(100,100*topline[0]+topline[1]);
//    cv::Point endpt(srcimg.cols-100,(srcimg.cols-100)*topline[0]+topline[1]);
//    cv::line(srcimg,startpt,endpt,255,3);
//    contour.clear();

//    er.KirschEdgeInnerLine(edgeleft,50,2,0,vector<int>{10,edgeleft.rows-10},contour);
//    for(int i=0;i<contour.size();++i){
//        cv::Point temp(contour[i].y+909,contour[i].x+143);
//        contour[i]=temp;
//    }
//    vector<float> leftline;
//    fl.LineFitLeastSquares(contour,leftline);
//    startpt.x=100*leftline[0]+leftline[1];
//    startpt.y=100;
//    endpt.x=(srcimg.rows-100)*leftline[0]+leftline[1];
//    endpt.y=srcimg.rows-100;
//    cv::line(srcimg,startpt,endpt,255,3);
//    contour.clear();

//    er.KirschEdgeInnerLine(edgedown,50,1,edgedown.rows,vector<int>{10,edgedown.cols-10},contour);
//    for(int i=0;i<contour.size();++i){
//        cv::Point temp(contour[i].x+1145,contour[i].y+5521);
//        contour[i]=temp;
//    }
//    vector<float> bottomline;
//    fl.LineFitLeastSquares(contour,bottomline);
//    startpt.x=100;
//    startpt.y=100*bottomline[0]+bottomline[1];
//    endpt.x=srcimg.cols-100;
//    endpt.y=(srcimg.cols-100)*bottomline[0]+bottomline[1];
//    cv::line(srcimg,startpt,endpt,255,3);
//    contour.clear();

//    er.KirschEdgeInnerLine(edgeright,50,4,edgeright.cols,vector<int>{10,edgeright.rows-10},contour);
//    for(int i=0;i<contour.size();++i){
//        cv::Point temp(contour[i].y+865,contour[i].x+3921);
//        contour[i]=temp;
//    }
//    vector<float> rightline;
//    fl.LineFitLeastSquares(contour,rightline);
//    startpt.x=100*rightline[0]+rightline[1];
//    startpt.y=100;
//    endpt.x=(srcimg.rows-100)*rightline[0]+rightline[1];
//    endpt.y=srcimg.rows-100;
//    cv::line(srcimg,startpt,endpt,255,3);
//    cv::imwrite("patent2.jpg",srcimg);
}

int circleparttest(){
    RectExtract re;
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/20201227/left/37-8+4000.jpg",0);
    //cv::Mat srcimg=totalimg(cv::Rect(cv::Point(1169,1516),cv::Point(1479,1708))).clone();//zuo left
    //cv::Mat srcimg=totalimg(cv::Rect(cv::Point(1683,1536),cv::Point(2000,1746))).clone();//zuo right
    //cv::Mat srcimg=totalimg(cv::Rect(cv::Point(2703,2749),cv::Point(2871,3025))).clone();//xia left
    //cv::Mat srcimg=totalimg(cv::Rect(cv::Point(3128,1579),cv::Point(3369,1707))).clone();//you left
    //cv::Mat srcimg=totalimg(cv::Rect(cv::Point(3100,2354),cv::Point(4130,3028))).clone();//s1 left
    //cv::Mat srcimg=totalimg(cv::Rect(cv::Point(646,1180),cv::Point(782,1906)));//se9 13 left
    //cv::Mat srcimg=totalimg(cv::Rect(cv::Point(1572,2937),cv::Point(1796,3156)));//ss1 13 left
    //cv::Mat srcimg=totalimg(cv::Rect(cv::Point(2942,2155),cv::Point(3128,2348)));//ss4 13 left
    //cv::Mat srcimg=totalimg(cv::Rect(cv::Point(2849,3191),cv::Point(3033,3371)));//ss3 13 left
    //cv::Mat srcimg=totalimg(cv::Rect(cv::Point(764,1230),cv::Point(936,1945)));//se9 13 right
    //cv::Mat srcimg=totalimg(cv::Rect(cv::Point(1667,2102),cv::Point(1847,2276)));//ss2 13 right
    //cv::Mat srcimg=totalimg(cv::Rect(cv::Point(3079,2195),cv::Point(3260,2373)));//ss4 13 right
    //cv::Mat srcimg=totalimg(cv::Rect(cv::Point(2980,3226),cv::Point(3167,3401)));//ss3 13 right
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat edge = re.krisch(srcimg);
    int radius=5;
    float percent=0.8;
    int radthresh=1;
    cv::threshold(edge, edge, 160, 255, cv::THRESH_BINARY);
    //cv::imshow("edge",edge);
    cv::imwrite("newedge.jpg",edge);
    cv::Point center(edge.cols/2,edge.rows/2);
    edge.at<uchar>(edge.rows/2,edge.cols/2)=0;
    vector<cv::Point> contours;
//    re.FindEdge(edge, center, contours, radius,percent,radthresh);
//    contours=re.ContoursCut(contours, 240, 10);
//    re.KirschEdgeLine(srcimg,160,3,contours);
//    vector<float> dist;
//    re.GetRectEdge(srcimg,160,1,dist);
//    float dist;
//    re.KirschEdgeOuter(srcimg,160,9,dist);
//    re.KirschEdgeSmallCircle(srcimg,160,1,contours);
//    for(int i=0;i<contours.size();++i){
//        cv::circle(srcimg,contours[i],1,255,-1);
//    }
//    cv::imshow("edgeres",srcimg);
    cv::waitKey(0);
    return 0;
}

int GetNewDatumleft(){
    RectExtract rectex;
    cv::Mat dstimg;
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/452-7-2/36/0110-0054-36.jpg",0);
    cv::threshold(srcimg, dstimg, 160, 255, cv::THRESH_BINARY_INV);
    cv::Mat TopRight=dstimg(cv::Rect(cv::Point(4220,170),cv::Point(4690,220)));
    cv::Mat TopLeft=dstimg(cv::Rect(cv::Point(150,170),cv::Point(290,210)));
    cv::Mat Bottom=dstimg(cv::Rect(cv::Point(370,2220),cv::Point(420,2730)));
    vector<cv::Point> contourstr;
    cv::imshow("111",TopRight);
    rectex.ScanLine(TopRight,1,contourstr,TopRight.rows-1,5,0.5);
    for(int i=0;i<contourstr.size();++i){
        contourstr[i].x+=4220;
        contourstr[i].y+=170;
    }
    vector<cv::Point> contourstl;
    rectex.ScanLine(TopLeft,1,contourstl,TopLeft.rows-1,5,0.5);
    for(int i=0;i<contourstl.size();++i){
        contourstl[i].x+=150;
        contourstl[i].y+=170;
    }
    vector<float> top;
    contourstr.insert(contourstr.end(),contourstl.begin(),contourstl.end());
    rectex.LineFitLeastSquares(contourstr,top);
    vector<cv::Point> contoursl;
    rectex.ScanLine(Bottom,4,contoursl,Bottom.cols-1,5,0.5);
    for(int i=0;i<contoursl.size();++i){
        cv::Point temp=contoursl[i];
        contoursl[i].x=temp.y+2220;
        contoursl[i].y=temp.x+370;
    }
    vector<float> left;
    rectex.LineFitLeastSquares(contoursl,left);
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
    cv::circle(srcimg,topleft,3,255,-1);
    cv::circle(srcimg,toppoint,3,255,-1);
    cv::circle(srcimg,leftpoint,3,255,-1);
    for(int i=0;i<srcimg.rows;++i){
        cv::circle(srcimg,cv::Point(i*left[0]+left[1],i),3,255,-1);
    }
    for(int i=0;i<srcimg.cols;++i){
        cv::circle(srcimg,cv::Point(i,i*top[0]+top[1]),3,255,-1);
    }
    cv::imwrite("res.jpg",srcimg);
    return 0;
}

int GetNewDatumright(){
    RectExtract re;
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/20201227/right/37+8+2000.jpg",0);
    cv::Mat srcimg1=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/20201227/right/36-3000.jpg",0);
    cv::threshold(srcimg1, srcimg1, 160, 255, cv::THRESH_BINARY);
    for(int i=0;i<srcimg.rows;++i){
        for(int j=0;j<srcimg.cols;++j){
            if(srcimg1.at<uchar>(i,j)==0){
                srcimg1.at<uchar>(i,j)=srcimg.at<uchar>(i,j);
            }
        }
    }
    cv::imwrite("merge.jpg",srcimg1);
    FindLines fl;
    cv::Mat TopRight=srcimg1(cv::Rect(cv::Point(4641,222),cv::Point(5060,319)));
    cv::Mat TopLeft=srcimg1(cv::Rect(cv::Point(570,253),cv::Point(690,293)));
    cv::Mat Bottom=srcimg1(cv::Rect(cv::Point(777,2313),cv::Point(858,2662)));
    vector<cv::Point> contourstr;
    re.ScanLine(TopRight,1,contourstr,TopRight.rows-1,5,0.5);
    for(int i=0;i<contourstr.size();++i){
        contourstr[i].x+=4641;
        contourstr[i].y+=222;
    }
    vector<cv::Point> contourstl;
    re.ScanLine(TopLeft,1,contourstl,TopLeft.rows-1,5,0.5);
    for(int i=0;i<contourstl.size();++i){
        contourstl[i].x+=570;
        contourstl[i].y+=253;
    }
    vector<float> top;
    contourstr.insert(contourstr.end(),contourstl.begin(),contourstl.end());
    fl.LineFitLeastSquares(contourstr,top);
    for(int i=0;i<srcimg1.cols;++i){
        cv::circle(srcimg1,cv::Point(i,i*top[0]+top[1]),3,255,-1);
    }
    vector<cv::Point> contoursl;
    re.ScanLine(Bottom,4,contoursl,Bottom.cols-1,5,0.5);
    for(int i=0;i<contoursl.size();++i){
        cv::Point temp=contoursl[i];
        contoursl[i].x=temp.y+2313;
        contoursl[i].y=temp.x+777;
    }
    vector<float> left;
    fl.LineFitLeastSquares(contoursl,left);
    for(int i=0;i<srcimg1.rows;++i){
        cv::circle(srcimg1,cv::Point(i*left[0]+left[1],i),3,255,-1);
    }
    cv::imwrite("mergetop.jpg",srcimg1);
    return 0;
}

int LofTest(){
    LOF lof;
    vector<vector<float>> feadata;
    lof.ReadTxt("/mnt/hgfs/linuxsharefiles/txtfiles/histfeature.txt",feadata);
    vector<vector<float>> testfeadata;
    lof.ReadTxt("/mnt/hgfs/linuxsharefiles/txtfiles/testhist.txt",testfeadata);
    feadata.insert(feadata.end(),testfeadata.begin(),testfeadata.end());
    vector<int> labels;
    lof.LOFclassification(feadata,5,0.90,labels);
    return 0;
}

int NewKirschEdgeOuter5751(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/20200121/001.jpg",0);
    cv::Mat ori=srcimg(cv::Rect(cv::Point(1526,3156),cv::Point(1748,3368)));
    RectExtract re;
    vector<int> paras={1,200,0,60,4,4,4,80,0,1,100,120};
    vector<cv::Point2f> dist;
    re.NewKirschEdgeOuter(ori,paras,dist);
    return 0;
}

int NewKirschEdgeOuter5752(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/20200121/001.jpg",0);
    cv::Mat ori=srcimg(cv::Rect(cv::Point(2740,914),cv::Point(3040,1094)));
    RectExtract re;
    vector<int> paras={2,160,1,60,SCAN_DIRECTION_DOWN,SCAN_DIRECTION_UP,4,80,0,1,100,120};
    vector<cv::Point2f> dist;
    re.NewKirschEdgeOuter(ori,paras,dist);
    return 0;
}

int NewCircleEdge575(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/20200121/000.jpg");
    RectExtract re;
    cv::Mat ori=srcimg(cv::Rect(cv::Point(942,1882),cv::Point(1128,2110)));
    vector<int> paras={3,160,1,ori.cols/2,ori.rows/2,4,50,0,150,210};
    vector<cv::Point> contours;
    re.NewCircleEdge(ori,paras,contours);
    return 0;
}

int NewSmallCircle575ss1(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/20210123/000.jpg");
    RectExtract re;
    cv::Mat ori=srcimg(cv::Rect(cv::Point(1602,2513),cv::Point(1808,2725)));
    cv::imshow("ori",ori);
    vector<int> paras={4,120,1,ori.cols/2,ori.rows/2,4,50,0,0,360};
    vector<float> res;
    re.NewSmallCircle(ori,paras,res);
    return 0;
}

int NewSmallCircle575ss3(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/20210123/001.jpg");
    RectExtract re;
    cv::Mat ori=srcimg(cv::Rect(cv::Point(3190,4048),cv::Point(3400,4260)));
    vector<int> paras={4,160,0,ori.cols/2,ori.rows/2,4,50,55,0,360};
//    cv::Mat ori=srcimg(cv::Rect(cv::Point(4302,3938),cv::Point(4513,4146)));
//    vector<int> paras={5,160,0,ori.cols/2+2,ori.rows/2+2,4,50,52,0,360};
    vector<float> res;
    re.NewSmallCircle(ori,paras,res);
    return 0;
}

int NewSmallCircle575sshigh(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/20210123/002.jpg");
    RectExtract re;
    cv::Mat ori=srcimg(cv::Rect(cv::Point(3681,2768),cv::Point(3867,2968)));
    vector<int> paras={7,120,2,ori.cols/2,ori.rows/2+6,4,50,10,0,360};
    vector<float> res;
    re.NewSmallCircle(ori,paras,res);
    return 0;
}

int NewRect1(){
    //cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/20210123/000.jpg",0);
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/57530pcs图片数据/1/1-1.jpg",0);
    RectExtract re;
    //cv::Mat ori=srcimg(cv::Rect(cv::Point(3537,3075),cv::Point(4577,3729)));
    cv::Mat ori=srcimg(cv::Rect(cv::Point(3551,3103),cv::Point(4565,3748)));
    cv::imshow("ori",ori);
    vector<int> paras{1,250,1,4,50,350,675,595,875,125,525,130,300,730,900,140,530};
    vector<float> dists;
    vector<vector<cv::Point2f>> contoursres;
    re.NewGetRectEdge(ori,paras,dists,contoursres);
    for(int i=0;i<contoursres.size();++i){
        for(int j=0;j<contoursres[i].size();++j){
            cv::circle(ori,contoursres[i][j],2,160,-1);
        }
    }
    cv::imshow("orires",ori);
    return 0;
}

int NewRect2(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/20210123/000.jpg",0);
    RectExtract re;
    cv::Mat ori=srcimg(cv::Rect(cv::Point(3457,3736),cv::Point(4344,4234)));
    cv::imshow("ori",ori);
    vector<int> paras{2,250,1,4,50,120,420};
    vector<float> dists;
    vector<vector<cv::Point2f>> contoursres;
    re.NewGetRectEdge(ori,paras,dists,contoursres);
    for(int i=0;i<contoursres.size();++i){
        for(int j=0;j<contoursres[i].size();++j){
            cv::circle(ori,contoursres[i][j],2,160,-1);
        }
    }
    cv::imshow("orires",ori);
    return 0;
}

int LDarea(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/20210123/000.jpg",0);
    RectExtract re;
    cv::Mat ori=srcimg(cv::Rect(cv::Point(651,2340),cv::Point(1304,3439)));
    cv::imshow("ori",ori);
    vector<cv::Point2f> respts;
    vector<int> paras{1,230,6,4,50,130,120,125,2,10,1015,1020,2,1080,80,85,1,200,735,740,4,680,460,465,1,640,425,430,4};
    re.LDmeasure(ori,paras,respts);
    for(int i=0;i<respts.size();++i){
        cv::circle(ori,cv::Point((int)respts[i].x,(int)respts[i].y),2,160,-1);
    }
    cv::imwrite("orires.jpg",ori);
    return 0;
}

int Datum2(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/0210127/12-15000.jpg",0);
    RectExtract re;
//    cv::Mat roi=srcimg(cv::Rect(cv::Point(3937,573),cv::Point(4579,757)));
//    vector<int> paras={1,200,2,90,1,5,roi.rows-1,3,60,66,220,226,380,386};
//    cv::Mat roi=srcimg(cv::Rect(cv::Point(1109,785),cv::Point(1410,956)));
//    vector<int> paras={1,200,2,90,1,5,roi.rows-1,3,20,26,100,106,230,236};
    cv::Mat roi=srcimg(cv::Rect(cv::Point(743,1028),cv::Point(976,2061)));
    vector<int> paras={1,200,2,90,4,2,roi.cols-1,4,150,156,548,554,820,826,930,936};
    vector<cv::Point2f> dist;
    re.NewKirschEdgeOuter(roi,paras,dist);
    return 0;
}

int Datum6161(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/616_templates/side1/SE11/1-1-8.png",0);
    edgedetect ed;
    vector<int> paras={1,230,1,4,50,4,135,80,320};
    vector<cv::Point> dists;
    ed.GetLineContours(srcimg,paras,dists);
    vector<int> paras1={2,160,0,4,50,3,40,450,790};
    vector<cv::Point> dists1;
    cv::Mat srcimg1=cv::imread("/mnt/hgfs/linuxsharefiles/616_templates/side1/SE12/1-1-8.png",0);
    ed.GetLineContours(srcimg1,paras1,dists1);
    vector<cv::Point2f> datum;
    ed.GetSideDatum616(dists,dists1,datum);
    return 0;
}

int mea452(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/E452_temp/back/Setup3/1015-0010-36.png",0);
    edgedetect ed;
    vector<int> paras={13,160,1,4,50,4,85,85,420};
    vector<cv::Point> dist;
    ed.GetLineContours(srcimg,paras,dist);
    return 0;
}

int cutimage(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/images/20210315/1-1.jpg",0);
    cv::Mat roi=srcimg(cv::Rect(2788,1782,1382,323));
    cv::imwrite("616roi.png",roi);
    return 0;
}

int fai575new(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/codes/ALgMeasurement575-Added/575MeasureData/template_images/front/SE4/best_temp.png",0);
    vector<int> paras1={21,2,90,5,4,50,135,355,360,4,135,655,660,4,130,1060,1065,4,135,1555,1560,4,140,1915,1920,4};
    edgedetect ed;
    vector<cv::Point2f> dist;
    ed.LDmeasure(srcimg,paras1,dist);
    return 0;
}

int fai618(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/616_temp/back/SetUp1/0020-0001-03.png",0);
    vector<int> paras1={1,2,160,4,50,2,30,135,585,2,30,4040,4330,1,730,130,300};
    edgedetect ed;
    vector<cv::Point2f> dist;
    ed.GetDatum616(srcimg,paras1,dist);
    for(int i=0;i<dist.size();++i){
        cv::circle(srcimg,cv::Point((int)dist[i].x,(int)dist[i].y),3,160,-1);
    }
    cv::imwrite("se777.jpg",srcimg);
    return 0;
}

int edgese(){
    vector<vector<float>> curves;
    std::string path="/mnt/hgfs/linuxsharefiles/txtfiles/curve.txt";
    txt_to_vectorfloat(curves,path);
    vector<float> curve=curves[40];
    vector<float> curveinv;
    for(int i=0;i<curve.size();++i){
        curveinv.push_back(curve[curve.size()-1-i]);
    }
    vector<int> change;
    float minpos=256;
    for(int i=0;i<curve.size();++i){
        if(minpos>curve[i]){
            minpos=curve[i];
            change.push_back(1);
        }
        else{
            change.push_back(0);
        }
    }
    vector<int> changeinv;
    float minposinv=256;
    for(int i=0;i<curveinv.size();++i){
        if(minposinv>curveinv[i]){
            minposinv=curveinv[i];
            changeinv.push_back(1);
        }
        else{
            changeinv.push_back(0);
        }
    }
    vector<int> stpoint;
    int zeronum=0;
    for(int i=0;i<changeinv.size();++i){
        if(changeinv[i]==0){
            zeronum++;
        }
        else{
            stpoint.push_back(zeronum);
            zeronum=0;
        }
    }
    int pos=0;
    for(int i=0;i<stpoint.size();++i){
        pos+=stpoint[i];
        if(stpoint[i]>20&&pos+i-stpoint[i]>30){
            cout<<pos+i-stpoint[i]<<endl;
        }
    }
    return 0;
}

int gencurveparas(){
    vector<vector<float>> curves;
    std::string path="/mnt/hgfs/linuxsharefiles/txtfiles/curve.txt";
    txt_to_vectorfloat(curves,path);
    vector<float> curve=curves[23];
    vector<float> curveinv;
    for(int i=0;i<curve.size();++i){
        curveinv.push_back(curve[curve.size()-1-i]);
    }
    int length=20;
    curvefitting cf;
    vector<vector<cv::Point2d>> curvespoint(curveinv.size()-length+1);
    for(int i=0;i<curveinv.size()-length;++i){
        vector<cv::Point2d> curvepoint(length);
        for(int j=0+i;j<length+i;++j){
            curvepoint[j-i].x=j-i;
            curvepoint[j-i].y=curveinv[j];
        }
        curvespoint[i]=curvepoint;
    }
    vector<vector<float>> res;
    cf(curvespoint,res);
    res.pop_back();
    double minerror=res[0][3];
    int minerrpos=0;
    for(int i=0;i<res.size();++i){
        if(minerror>res[i][3]){
            minerror=res[i][3];
            minerrpos=i;
        }
    }
    return 0;
}

int a616fai3(){
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/testImages/616_temp/front/SE2/0020-0001-04.png",0);
    vector<int> paras1={1,100,2,10,2,0,107,3,1065,1070,1280,1285,1580,1585};
    edgedetect ed;
    vector<cv::Point2f> dist;
    ed.NewKirschEdgeOuter(srcimg,paras1,dist);
    for(int i=0;i<dist.size();++i){
        cv::circle(srcimg,cv::Point((int)dist[i].x,(int)dist[i].y),3,160,-1);
    }
    cv::imwrite("se2.jpg",srcimg);
    return 0;
}

int getcurvestest(){
    std::string path="/mnt/hgfs/linuxsharefiles/codes/618Project/MeasureData618/template_images/front/SE1/";
    vector<cv::String> imgnames;
    glob(path,imgnames,false);
    vector<vector<float>> totalcurves;
    for(int i=0;i<imgnames.size();++i){
        cv::Mat singleimg=cv::imread(imgnames[i],0);
        edgedetect ed;
        vector<vector<float>> curves;
        ed.Pixel2Curve(singleimg,170,1965,3,curves);
        for(int j=0;j<curves.size();++j){
            curves[j].erase(curves[j].begin(),curves[j].begin()+112);
        }
        if(i==0){
            totalcurves=curves;
        }
        else{
            totalcurves.insert(totalcurves.end(),curves.begin(),curves.end());
        }
    }
    ofstream outputfile;
    outputfile.open("edgecurves.txt");
    for(int i=0;i<totalcurves.size();++i){
        for(int j=0;j<totalcurves[i].size();++j){
            outputfile<<totalcurves[i][j]<<" ";
        }
        outputfile<<endl;
    }
    outputfile.close();
    return 0;
}

int main()
{
    vector<cv::Point2f> xpts,ypts,datum;
    xpts.push_back(cv::Point2f(30,200));
    xpts.push_back(cv::Point2f(3000,205));
    xpts.push_back(cv::Point2f(1500,203));
    ypts.push_back(cv::Point2f(500,300));
    ypts.push_back(cv::Point2f(503,1600));
    ypts.push_back(cv::Point2f(502,3000));
    RectExtract re;
    re.getdatum001(xpts,ypts,datum);
    cv::waitKey(0);
    return 0;
}
