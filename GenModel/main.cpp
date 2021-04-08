#include <iostream>
#include <QString>
#include <QStringList>
#include "preprocess.h"

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
    preprocess pp;
    vector<vector<string>> csvdata;
    pp.ReadCSVfile("/mnt/hgfs/linuxsharefiles/txtfiles/measureDistance_7_383.csv",csvdata);
    vector<vector<float>> stddata;
    vector<string> stdname;
    pp.ReadTxttoString("/mnt/hgfs/linuxsharefiles/txtfiles/45.txt",stdname);
    pp.ReadTxttoVec("/mnt/hgfs/linuxsharefiles/txtfiles/b1.txt",stddata);
    ofstream outfile("out.txt",ios::app); //ios::app指追加写入
    for(int i=0;i<stdname.size();++i){
        vector<float> datavec0;
        vector<float> datavec2;
        string name=stdname[i].substr(0,stdname[i].size()-1);
        pp.GetColdata(csvdata,datavec0,2,3,name);
        pp.GetColstddata(stddata,datavec2,i);
        float c1=pp.getoptimaloffset(datavec0,datavec2,30,-30);
        string temp=std::to_string(c1);
        outfile<<temp;//写文件
        outfile<<endl;
    }
    outfile.close();
    return 0;
}
