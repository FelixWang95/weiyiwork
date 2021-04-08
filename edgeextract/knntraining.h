#ifndef KNNTRAINING_H
#define KNNTRAINING_H

#include <opencv2/opencv.hpp>

using namespace std;

class KNNtraining
{
public:
    KNNtraining();
    int ImageCheck(cv::Mat srcimg);
    int KNNfunc(cv::Mat InputTrainingGood,cv::Mat InputGood, cv::Mat InputBad,float sigma, vector<float>& result);

private:
    float disttest(cv::Mat InputTrainingGood);
    float CalcDistance(cv::Mat rowvec1, cv::Mat rowvec2);
    vector<float> CalcDistVecForOne(cv::Mat rowvec, cv::Mat trainingdata);
    cv::Mat PCAdecrease(cv::PCA pca, cv::Mat inputMat);
    cv::PCA ReadPCAFromXML(cv::String path);
    int KNNtest(cv::Mat InputTraining, cv::Mat InputTest, float sigma, int k);
    void ReadXml(std::string xmlname, cv::Mat& xmldata);
    int CalcHistogram(cv::Mat srcimg, int thresh,int bin,cv::Mat& hist);
};

#endif // KNNTRAINING_H
