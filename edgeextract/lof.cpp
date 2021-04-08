#include "lof.h"

LOF::LOF()
{

}

float LOF::calcdist(vector<float> vec1, vector<float> vec2){
    if(vec1.size()!=vec2.size())
        return -1;
    double dist=0;
    for(int i=0;i<vec1.size();++i){
        dist+=(vec1[i]-vec2[i])*(vec1[i]-vec2[i]);
    }
    return sqrt(dist);
}

float LOF::getkdistpoints(vector<float> basevec, vector<vector<float>> vecgroup, int k, vector<vector<float>>& kdistpoints){
    vector<float> dists(vecgroup.size());
    vector<float> sortdists(vecgroup.size());
    for(int i=0;i<vecgroup.size();++i){
        dists[i]=calcdist(basevec,vecgroup[i]);
        sortdists[i]=dists[i];
    }
    sort(sortdists.begin(),sortdists.end());
    float kdis=sortdists[k];
    kdistpoints.clear();
    for(int i=0;i<vecgroup.size();++i){
        if(dists[i]<=kdis){
            kdistpoints.push_back(vecgroup[i]);
        }
    }
    return kdis;
}

float LOF::reachdist(vector<float> vec1, vector<float> vec2, vector<vector<float>> vecgroup, int k){
    vector<vector<float>> kdistpoints;
    float kdist=getkdistpoints(vec2, vecgroup, k, kdistpoints);
    float dist=calcdist(vec1,vec2);
    if(kdist<dist){
        return dist;
    }
    else{
        return kdist;
    }
}

float LOF::localreachdense(vector<float> vec1,vector<vector<float>> vecgroup, int k){
    vector<vector<float>> kdistpoints;
    float kdist=getkdistpoints(vec1, vecgroup, k, kdistpoints);
    float sumreachdist=0;
    for(int i=0;i<kdistpoints.size();++i){
        sumreachdist+=reachdist(vec1,kdistpoints[i],vecgroup,k);
    }
    float lrd=(float)kdistpoints.size()/sumreachdist;
    return lrd;
}

float LOF::LocalOutlierFactor(vector<float> vec1,vector<vector<float>> vecgroup, int k){
    vector<vector<float>> kdistpoints;
    float kdist=getkdistpoints(vec1, vecgroup, k, kdistpoints);
    float sumdiv=0;
    for(int i=0;i<kdistpoints.size();++i){
        sumdiv+=(localreachdense(kdistpoints[i],vecgroup, k)/localreachdense(vec1,vecgroup, k));
    }
    return sumdiv/(float)kdistpoints.size();
}

int LOF::LOFclassification(vector<vector<float>> vecgroup, int k, float percent, vector<int>& labels){
    if(vecgroup.size()<=k||k<1)
        return -1;
    vector<float> lofres(vecgroup.size());
    vector<float> sortlof(vecgroup.size());
    for(int i=0;i<vecgroup.size();++i){
        lofres[i]=LocalOutlierFactor(vecgroup[i],vecgroup,k);
        sortlof[i]=lofres[i];
    }
    sort(sortlof.begin(),sortlof.end());
    int divideindex=vecgroup.size()*percent;
    float divdist=sortlof[divideindex-1];
    labels.resize(vecgroup.size());
    for(int i=0;i<vecgroup.size();++i){
        if(lofres[i]<=divdist){
            labels[i]=1;
        }else{
            labels[i]=-1;
        }
    }
    return 0;
}

int LOF::ReadTxt(std::string txtname, vector<vector<float>>& txtdata){
    fstream file1;
    int n=0;
    string tmp;
    int colnum=0;
    file1.open(txtname);
    if(!file1.is_open())//文件打开失败:返回0
    {
        return -1;
    }
    else//文件存在
    {
        while(getline(file1,tmp,'\n'))
        {
            if(n==0){
                for(int i=0;i<tmp.size();++i){
                    if(tmp[i]==' ')
                        colnum++;
                }
            }
            n++;
        }
    }
    file1.close();
    fstream file2;
    file2.open(txtname);
    txtdata.resize(n);
    vector<float> linedata(colnum);
    //将txt文件数据写入到Data矩阵中
    for (int i = 0; i < txtdata.size(); ++i)
    {
        for (int j = 0; j < linedata.size(); ++j)
        {
            file2 >> linedata[j];
            //cout<<linedata[j]<<endl;
        }
        txtdata[i]=linedata;
    }
    file2.close();
    return 0;
}
