#include <iostream>
#include "preprocess.h"
#include "imagescreen.h"

using namespace std;

int prepro(){
    int rtn=0;
    int task=0;//0:feature extraction  1:classification
    std::vector<cv::String> image_files;
    std::vector<cv::String> image_files_1;
    preprocess *prepro=new preprocess();
    std::string filepath="/mnt/hgfs/linuxsharefiles/capimg/classifiedimg/0_L/0_L1/*.jpg";
    std::string savepath=filepath.substr(0,filepath.size()-5);
    prepro->ReadImagePath(filepath,image_files);
    //cout<<image_files[0].size()<<endl;
    cv::Mat labelspy;
    if(task==1)
    {
        prepro->ReadTxt("labels.txt",labelspy);
    }
    cv::Mat dataMat;
    int dim=256;
    for(int num=0;num<image_files.size();++num)//read all images in the folder
    {
        cout << image_files[num] << endl;
        image_files_1.push_back(image_files[num]);
        if(task==0){
            cv::Mat srcimg=cv::imread(image_files[num],0);
            cv::Mat hist;
            rtn=prepro->CalcHistogram(srcimg,20,dim,hist);
            if(image_files_1.size()==1){
                dataMat=hist;
            }
            else{
                cv::vconcat(dataMat,hist,dataMat);
            }
        }
//        if(image_files[num].size()>=63&&image_files[num][53]=='1'&&image_files[num][image_files[num].size()-5]=='1'){
//            cout << image_files[num] << endl;
//            image_files_1.push_back(image_files[num]);
//            cv::Mat srcimg=cv::imread(image_files[num],0);
//            cv::Mat lineimg=srcimg(cv::Rect(cv::Point(1806,56),cv::Point(2702,309)));
//            float angle=prepro->CalibrationImgOri(lineimg,3,180);
//            cout<<angle<<endl;
//            cv::Mat dstimg;
//            prepro->RotateImg(srcimg,dstimg,angle);
//            cv::imwrite(image_files[num],dstimg);
//            cv::Mat hist;
//            rtn=prepro->CalcHistogram(srcimg,20,dim,hist);
//            if(image_files_1.size()==1){
//                dataMat=hist;
//            }
//            else{
//                cv::vconcat(dataMat,hist,dataMat);
//            }
//        }
    }
    if(task==0){
        cout<<dataMat.size()<<endl;
        cv::Mat output;
        prepro->PCADecreaseDim(dataMat,output, 0.1);
        cout<<output<<endl;
        rtn=prepro->OutputMat("/home/adt/PycharmProjects/pythonProject/feaMat256.txt",dataMat);
    }
    else{
        prepro->ClassifyImages(image_files_1,labelspy,savepath+"classify");
    }
    //cv::waitKey(0);
    return 0;
}

int classifyandgenerate(){
    int task=0;
    std::string filepath="/mnt/hgfs/linuxsharefiles/s_data_file/s_SS/s_SS7/s_SS7_features.xml";
    std::string namepath="/mnt/hgfs/linuxsharefiles/s_data_file/s_D/s_D1/name.txt";
    preprocess *prepro=new preprocess();
    cv::Mat feaMat,labels;
    vector<cv::String> namevec;
    prepro->ReadXml(filepath, feaMat);
    //cout<<feaMat<<endl;
    prepro->OutputMat("/home/adt/PycharmProjects/pythonProject/feaMat256.txt",feaMat);
    if(task==1){
        prepro->ReadName(namepath,namevec);
        cv::String firstpart;
        for(int i=filepath.size()-1;i>=0;--i){
            if(filepath[i]=='/'){
                firstpart=filepath.substr(0,i+1);
                break;
            }
        }
        cv::String secpart;
        for(int i=0;i<namevec.size();++i){
            for(int j=namevec[i].size()-1;j>=0;--j){
                if(namevec[i][j]=='/'){
                    secpart=namevec[i].substr(j+1,namevec[i].size()-j);
                    break;
                }
            }
            namevec[i]=firstpart+secpart;
        }
        prepro->ReadTxt("labels.txt",labels);
        labels.convertTo(labels,CV_32SC1);
        feaMat.convertTo(feaMat,CV_32FC1);
        //cout<<labels<<endl;
        //cout<<feaMat<<endl;
        prepro->SVMmodel(feaMat,labels,filepath.substr(0,filepath.size()-13));
        prepro->ClassifyImages(namevec,labels,firstpart+"classify");
    }
//    cout<<feaMat<<endl;
//    for(int i=0;i<namevec.size();++i){
//        cout<<namevec[i]<<endl;
//    }
    cv::waitKey(0);
    return 0;
}

int generatedatums(){
    preprocess *prepro=new preprocess();
    std::vector<cv::String> image_files;
    std::string filepath="/mnt/hgfs/linuxsharefiles/newimages/totalimage/1-11/*.jpg";
    prepro->ReadImagePath(filepath,image_files);
    ofstream datumfile;
    datumfile.open("datums.txt");
    for(int i=0;i<1;++i){
        cv::Mat srcimg=cv::imread(image_files[i],0);
        datumfile<<image_files[i];
        vector<cv::Point2f> datumpoints;
        prepro->GetDatum(srcimg,datumpoints);
        for(int i=0;i<datumpoints.size();++i){
            int x=datumpoints[i].x;
            int y=datumpoints[i].y;
            cout<<x<<" "<<y<<endl;
            datumfile<<" "<<x<<" "<<y;
            //cv::circle(srcimg,cv::Point(x,y),10,255,-1);
        }
        //cv::line(srcimg, datumpoints[0], datumpoints[2], 255, 2);
        //cv::line(srcimg, datumpoints[1], datumpoints[2], 255, 2);
        cv::imwrite("/mnt/hgfs/linuxsharefiles/newimages/totalimage/1-11datum/"+std::to_string(i)+"datumimg.jpg",srcimg);
        datumfile<<endl;
    }
    datumfile.close();
    cv::waitKey(0);
    return 0;
}

void txt_to_vectordouble(vector<vector<double>>& res, string pathname)
{
    ifstream infile;
    infile.open(pathname.data());   //将文件流对象与文件连接起来
    assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行
    vector<double> suanz;
    string s;
    while (getline(infile, s)) {
        istringstream is(s); //将读出的一行转成数据流进行操作
        double d;
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

//计算数据偏差
int getoptimaloffset(){
    vector<float> datavec0{-21.7588  ,
                           -21.752   ,
                           -21.7603  ,
                           -21.7505  ,
                           -21.751   ,
                           -21.7539  ,
                           -21.7461  ,
                           -21.752   ,
                           -21.7559  ,
                           -21.7446  ,
                           -21.7573  ,
                           -21.7227  ,
                           -21.748   ,
                           -21.7603  ,
                           -21.7603  ,
                           -21.7251  ,
                           -21.7324  ,
                           -21.7402  ,
                           -21.7593  ,

};
    vector<float> datavec2{-20.416 ,
                           -20.424 ,
                           -20.379 ,
                           -20.372 ,
                           -20.37  ,
                           -20.427 ,
                           -20.419 ,
                           -20.364 ,
                           -20.372 ,
                           -20.41  ,
                           -20.378 ,
                           -20.347 ,
                           -20.377 ,
                           -20.414 ,
                           -20.379 ,
                           -20.417 ,
                           -20.418 ,
                           -20.346 ,
                           -20.366 ,
};
    float th=-5,step=0.001;
    vector<float> distvec;
    for(float k=th;k<5;k=k+step){
        vector<float> datavec1;
        for(int i=0;i<datavec0.size();++i){
            datavec1.push_back(datavec0[i]+k);
        }
        float sum=0;
        for(int i=0;i<datavec1.size();++i){
            sum+=fabs(datavec1[i]-datavec2[i]);
        }
        float b=sum/datavec1.size();
        float sumsd=0;
        for(int i=0;i<datavec1.size();++i){
            sumsd+=(fabs(datavec1[i]-datavec2[i])-b)*(fabs(datavec1[i]-datavec2[i])-b);
        }
        float c=sumsd/datavec1.size();
        distvec.push_back(sqrt(b*b+c*c));
    }
    auto smallest = std::min_element(std::begin(distvec), std::end(distvec));
    float r=th+std::distance(std::begin(distvec), smallest)*step;
    cout<<r<<endl;
    cv::waitKey(0);
    return 0;
}

int main()
{
    getoptimaloffset();
    cv::waitKey(0);
    return 0;
}
