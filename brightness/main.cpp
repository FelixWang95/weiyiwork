#include <iostream>
#include "calcfunc.h"

using namespace std;

int main()
{
    calcfunc *func=new calcfunc();
    std::string fn="/mnt/hgfs/linuxsharefiles/Result";
    vector<int> ptnum;
    func->ReadPointNum(fn,ptnum);
    for(int i=0;i<ptnum.size();++i)
    {cout<<ptnum[i]<<endl;}
    cout<<ptnum.size()<<endl;
    vector<cv::Point> pts;
    for(int i=50;i<60;++i){
        for(int j=50;j<60;++j){
            cv::Point pt(i,j);
            pts.push_back(pt);
        }
    }
    cv::Mat src=cv::imread("/mnt/hgfs/linuxsharefiles/darksurface/kmeansClasses4/3/35588-4-5-11.jpg",0);
    float brightness=func->BrightnessLevel(pts,src);
    cout<<brightness<<endl;
    vector<cv::Point> oppts;
    func->DilateThresh(src,pts,oppts);
    cout<<pts.size()<<endl;
    cout<<oppts.size()<<endl;
    return 0;
}
