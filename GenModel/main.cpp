#include <iostream>
#include <QString>
#include <QStringList>
#include "preprocess.h"
#include "interpolation.h"

using namespace std;

int GenSVMmodel(){
    int task=1;
    std::string strDirName = "S4";
    std::string filepath= "/mnt/hgfs/linuxsharefiles/testImages/s_data_file/S/" + strDirName + "/s_"+strDirName +"_features.xml";
    std::string namepath= "/mnt/hgfs/linuxsharefiles/testImages/s_data_file/S/"+strDirName +"/Name.txt";
    preprocess *prepro=new preprocess();
    cv::Mat feaMat,labels;
    vector<cv::String> namevec;
    prepro->ReadXml(filepath, feaMat);
    //cout<<feaMat<<endl;
    prepro->OutputMat("/home/adt/PycharmProjects/pythonProject/feaMat256.txt",feaMat);
    if(task==1){
        prepro->ReadName(namepath,namevec);
        //namevec.pop_back();
//        for(int i=0;i<namevec.size();++i){
//            QString strString = QString::fromStdString(namevec.at(i));
//            QStringList strList = strString.split(QChar('.'));
//            QString strFirst = strList.at(0);
//            strFirst.replace(strFirst.length()-2, 2, QString::fromStdString(strDirName));
//            QString strResult = QString("%1.%2").arg(strFirst).arg(strList.at(1));
//            std::string s = strResult.toStdString();
//            namevec[i] = s;
//        }
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
        prepro->ReadTxt("/home/adt/PycharmProjects/pythonProject/labels.txt",labels);
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

int main()
{
    interpolation ip;
    cv::Mat srcimg=cv::imread("/mnt/hgfs/linuxsharefiles/small144.jpg",0);
    cv::imshow("src",srcimg);
    cv::Mat dstimg;
    float rad=30.0/180.0*CV_PI;
    ip.imageRotationbcl(srcimg,dstimg,rad);
    cv::imshow("dst",dstimg);
    cv::waitKey(0);
    return 0;
}
