#include <iostream>
#include "dbscan.h"

using namespace std;

int main()
{
    ifstream infile;
    infile.open("/home/adt/PycharmProjects/pythonProject/feaMat25699.txt");   //将文件流对象与文件连接起来
    assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行
    string tmp;
    vector<vector<float>> data;
    while(getline(infile,tmp,'\n'))
    {
        float s;
        vector<float> linedata;
        int start=0,end=0;
        for(int i=0;i<tmp.size();++i){
            if(tmp[i]==' '||i==tmp.size()-1){
                end=i;
                s = atof(tmp.substr(start,i-start).c_str());
                start=i;
                linedata.push_back(s);
            }
        }
        data.push_back(linedata);
    }
    infile.close();
    vector<cv::Point2d> features(data.size());
    for(int i=0;i<data.size();++i){
        cv::Point2d temp(data[i][0]*1000,data[i][1]*1000);
        features[i]=temp;
    }
    DBSCAN db(32,10);
    std::vector<resdata> s;
    db.res(features,s);
    return 0;
}
