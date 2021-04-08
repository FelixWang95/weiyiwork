#ifndef LOF_H
#define LOF_H

#include <opencv2/opencv.hpp>

using namespace std;
//该算法会给数据集中的每个点计算一个离群因子LOF，通过判断LOF是否接近于1来判定是否是离群因子。若LOF远大于1，则认为是离群因子，接近于1，则是正常点。
class LOF
{
public:
    LOF();
    //欧氏距离
    float calcdist(vector<float> vec1, vector<float> vec2);
    //k距离，返回k距离内的点集和k距离的值
    float getkdistpoints(vector<float> basevec, vector<vector<float>> vecgroup, int k, vector<vector<float>>& kdistpoints);
    //vec1相对于vec2的可达距离
    float reachdist(vector<float> vec1, vector<float> vec2, vector<vector<float>> vecgroup, int k);
    //vec1局部可达密度
    float localreachdense(vector<float> vec1,vector<vector<float>> vecgroup, int k);
    //vec1的局部离群因子lof
    float LocalOutlierFactor(vector<float> vec1,vector<vector<float>> vecgroup, int k);
    //Lof算法，k为kdistance,percent为正常点的比例
    int LOFclassification(vector<vector<float>> vecgroup, int k, float percent, vector<int>& labels);
    //read txt to vector
    int ReadTxt(std::string txtname, vector<vector<float>>& txtdata);
};

#endif // LOF_H
