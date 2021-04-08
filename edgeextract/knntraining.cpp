#include "knntraining.h"
#include "ResultEvaluation.h"

KNNtraining::KNNtraining()
{

}

int KNNtraining::ImageCheck(cv::Mat srcimg){
    cv::resize(srcimg,srcimg,cv::Size(srcimg.cols/5,srcimg.rows/5));
    cv::String trainpath="/home/adt/QTcode/edgeextract/KnnTrainingData.xml";
    cv::Mat trainingMat;
    cv::Mat testimgfea;
    ReadXml(trainpath,trainingMat);
    CalcHistogram(srcimg,0,255,testimgfea);
    return KNNtest(trainingMat,testimgfea,0.0278, 130);
}

int KNNtraining::KNNfunc(cv::Mat InputTrainingGood,cv::Mat InputGood, cv::Mat InputBad,float sigma, vector<float>& result){
    if(InputGood.empty()||InputBad.empty()||InputTrainingGood.empty())
        return -1;
    if(InputGood.type()!=5)
        InputGood.convertTo(InputGood,CV_32FC1);
    if(InputBad.type()!=5)
        InputBad.convertTo(InputBad,CV_32FC1);
    if(InputTrainingGood.type()!=5)
        InputTrainingGood.convertTo(InputTrainingGood,CV_32FC1);
    vector<int> testlabels(InputGood.rows+InputBad.rows,0);
    int startk=InputTrainingGood.rows/10;
    int endk=InputTrainingGood.rows*9/10;
    vector<float> f1pnvec(endk-startk);
    vector<float> f1pvec(endk-startk);
    vector<float> f1nvec(endk-startk);
    for(int k=startk;k<endk;++k){
        for(int i=0;i<InputGood.rows;++i){
            vector<float> dists=CalcDistVecForOne(InputGood.rowRange(i,i+1),InputTrainingGood);
            int knum=0;
            for(int j=0;j<dists.size();++j){
                if(dists[j]<sigma)
                    knum++;
            }
            if(knum>k)
                testlabels[i]=1;
            else
                testlabels[i]=-1;
        }
        for(int i=0;i<InputBad.rows;++i){
            vector<float> dists=CalcDistVecForOne(InputBad.rowRange(i,i+1),InputTrainingGood);
            int knum=0;
            for(int j=0;j<dists.size();++j){
                if(dists[j]<sigma)
                    knum++;
            }
            if(knum>k)
                testlabels[i+InputGood.rows]=1;
            else
                testlabels[i+InputGood.rows]=-1;
        }
        vector<int> reallabels(InputGood.rows+InputBad.rows);
        for(int i=0;i<InputGood.rows;++i)
            reallabels[i]=1;
        for(int i=InputGood.rows;i<InputGood.rows+InputBad.rows;++i)
            reallabels[i]=-1;
        ResultEvaluation reseva;
        CalcResult res=reseva(reallabels,testlabels);
        cout<<res.F1_p<<"    "<<res.F1_n<<endl;
        float F1pn=(res.F1_p-1.0)*(res.F1_p-1.0)+(res.F1_n-1.0)*(res.F1_n-1.0);
        f1pnvec[k-startk]=F1pn;
        f1pvec[k-startk]=res.F1_p;
        f1nvec[k-startk]=res.F1_n;
    }
    result.clear();
    auto smallest = std::min_element(std::begin(f1pnvec), std::end(f1pnvec));
    int pos=std::distance(std::begin(f1pnvec), smallest);
    cout<<f1pnvec[pos]<<"  "<<sigma<<"  "<<startk+pos<<endl;
    result.push_back(f1pvec[pos]);
    result.push_back(f1nvec[pos]);
    result.push_back(f1pnvec[pos]);
    result.push_back(sigma);
    result.push_back(startk+pos);
    return 0;
}

int KNNtraining::KNNtest(cv::Mat InputTraining, cv::Mat InputTest, float sigma, int k){
    if(InputTraining.empty()||InputTest.empty()||InputTest.rows!=1)
        return -1;
    if(InputTraining.type()!=5)
        InputTraining.convertTo(InputTraining,CV_32FC1);
    if(InputTest.type()!=5)
        InputTest.convertTo(InputTest,CV_32FC1);
    vector<float> dists=CalcDistVecForOne(InputTest,InputTraining);
    int knum=0;
    for(int j=0;j<dists.size();++j){
        if(dists[j]<sigma)
            knum++;
    }
    if(knum>k)
        return 1;
    else
        return -1;
}

float KNNtraining::CalcDistance(cv::Mat rowvec1, cv::Mat rowvec2){
    if(rowvec1.rows!=1||rowvec2.rows!=1||rowvec1.cols!=rowvec2.cols)
        return -1;
    double distsum=0;
    for(int i=0;i<rowvec1.cols;++i){
        distsum+=(rowvec1.at<float>(0,i)-rowvec2.at<float>(0,i))*(rowvec1.at<float>(0,i)-rowvec2.at<float>(0,i));
    }
    float dist=sqrt(distsum);
    return dist;
}

vector<float> KNNtraining::CalcDistVecForOne(cv::Mat rowvec, cv::Mat trainingdata){
    vector<float> dists(trainingdata.rows,-1);
    if(rowvec.rows!=1||rowvec.cols!=trainingdata.cols)
        return dists;
    for(int i=0;i<trainingdata.rows;++i){
        dists[i]=CalcDistance(trainingdata.rowRange(i,i+1),rowvec);
    }
    return dists;
}

float KNNtraining::disttest(cv::Mat InputTrainingGood){
    if(InputTrainingGood.empty())
        return -1;
    if(InputTrainingGood.type()!=5)
        InputTrainingGood.convertTo(InputTrainingGood,CV_32FC1);
    vector<float> dists(InputTrainingGood.rows,-1);
    for(int i=0;i<InputTrainingGood.rows;++i){
        dists[i]=CalcDistance(InputTrainingGood.rowRange(i,i+1),InputTrainingGood.rowRange(0,1));
    }
    double sum = std::accumulate(std::begin(dists), std::end(dists), 0.0);
    std::vector<float>::iterator biggest = std::max_element(std::begin(dists), std::end(dists));
    return  *biggest;
}

void KNNtraining::ReadXml(std::string xmlname, cv::Mat& xmldata){
    cv::FileStorage fs(xmlname, cv::FileStorage::READ);
    fs["ex"] >> xmldata;
}

int KNNtraining::CalcHistogram(cv::Mat srcimg, int thresh,int bin,cv::Mat& hist){
    if(srcimg.empty()||bin>256||bin<1)
    {return -1;}
    int histSize = bin;//直方图分成多少个区间
    float range[] = { 0,256 };
    const float *histRanges = { range };//统计像素值的区间
    cv::calcHist(&srcimg,1,0,cv::Mat(),hist,1,&histSize,&histRanges,true,false);
    hist=hist.rowRange(thresh,hist.rows);
    float sum=0;
    for(int i=0;i<hist.rows;++i){
        sum+=hist.at<float>(i,0);
    }
    for(int i=0;i<hist.rows;++i){
        hist.at<float>(i,0)/=sum;
    }
    hist=hist.t();
    hist=PCAdecrease(ReadPCAFromXML("192255pca.xml"),hist);
    return 0;
}

cv::Mat KNNtraining::PCAdecrease(cv::PCA pca, cv::Mat inputMat){
    cv::Mat outputMat;
    for(int i=0;i<inputMat.rows;++i){
        cv::Mat point=pca.project(inputMat.row(i));
        //cout<<point.size()<<endl;
        if(i==0){
            outputMat=point;
        }
        else{
            cv::vconcat(outputMat,point,outputMat);
        }
    }
    return outputMat;
}

cv::PCA KNNtraining::ReadPCAFromXML(cv::String path){
    cv::FileStorage fs(path,cv::FileStorage::READ);
    cv::PCA pca;
    pca.read(fs.root());
    fs.release();
    return pca;
}
