#include "imagecheck.h"
#include "findoutercontour.h"
#include "ResultEvaluation.h"

ImageCheck::ImageCheck()
{

}

bool ImageCheck::IsBlackAround(cv::Mat srcimg){
    if(srcimg.empty())
        return false;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    for(int i=0;i<srcimg.cols;++i){
        if(srcimg.at<uchar>(0,i)>=40){
            return false;
        }
        if(srcimg.at<uchar>(srcimg.rows-1,i)>=40){
            return false;
        }
    }
    for(int i=0;i<srcimg.rows;++i){
        if(srcimg.at<uchar>(i,0)>=40){
            return false;
        }
        if(srcimg.at<uchar>(i,srcimg.cols)>=40){
            return false;
        }
    }
    return true;
}

bool ImageCheck::IsBlackAreaSatisfied(cv::Mat srcimg, int thresh, float lower, float upper){
    if(srcimg.empty())
        return false;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    int blacknum=0;
    for(int i=0;i<srcimg.rows;++i){
        for(int j=0;j<srcimg.cols;++j){
            if(srcimg.at<uchar>(i,j)<thresh){
                blacknum++;
            }
        }
    }
    float blackpercent=(float)blacknum/(srcimg.rows*srcimg.cols);
    if(blackpercent>lower&&blackpercent<upper){
        return true;
    }
    else{
        return false;
    }
}

float ImageCheck::GetBlackArea(cv::Mat srcimg, int thresh){
    if(srcimg.empty())
        return false;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    int blacknum=0;
    for(int i=0;i<srcimg.rows;++i){
        for(int j=0;j<srcimg.cols;++j){
            if(srcimg.at<uchar>(i,j)<thresh){
                blacknum++;
            }
        }
    }
    float blackpercent=(float)blacknum/(srcimg.rows*srcimg.cols);
    return blackpercent;
}

int ImageCheck::Contour2Angle(vector<cv::Point> contour, int step, int length, vector<float>& angles){
    if(contour.size()<10){
        return -1;
    }
    angles.clear();
    contour.insert(contour.end(),contour.begin(),contour.begin()+step+length);
    for(int i=0;i<contour.size()-(step+length);i=i+6){
        float angle=GetAngle(contour[i], contour[i+length], contour[i+step], contour[i+step+length]);
        angles.push_back(angle);
    }
    return 0;
}

float ImageCheck::GetAngle(cv::Point A, cv::Point B, cv::Point C, cv::Point D){
    float ABx=B.x-A.x;
    float ABy=B.y-A.y;
    float CDx=D.x-C.x;
    float CDy=D.y-C.y;
    float cosa=(ABx*CDx+ABy*CDy)/(sqrt(ABx*ABx+ABy*ABy)*sqrt(CDx*CDx+CDy*CDy));
    if(cosa>=1)
        cosa=0.999999;
    float angle=acos(cosa)*180/PI;
    if(isnan(angle))
        cout<<cosa<<endl;
    return angle;
}

int ImageCheck::GetContour(cv::Mat srcimg, vector<cv::Point>& contour){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    FindOuterContour fb;
    std::vector<float> source_x;
    std::vector<float> source_y;
    fb.PeripheralTraversa(srcimg,source_x,source_y);
    contour.resize(source_x.size());
    for(int j=0;j<source_x.size();++j){
        contour[j].x=source_x[j]*2;
        contour[j].y=source_y[j]*2;
    }
    contour=SortContourCenter(contour,cv::Point(3000,1500));
//    ofstream txtfile;
//    txtfile.open("contourposition.txt");
//    for(int i=0;i<contour.size();++i){
//        txtfile<<contour[i].x<<" "<<contour[i].y<<endl;
//    }
//    txtfile.close();
    cv::Mat dstimg=srcimg.clone();
    cv::resize(dstimg,dstimg,cv::Size(srcimg.cols,srcimg.rows));
    for(int j=0;j<contour.size();++j){
        cv::circle(dstimg,contour[j],1,255,-1);
        cv::putText(dstimg, std::to_string(j), contour[j], cv::FONT_HERSHEY_TRIPLEX, 0.4, cv::Scalar(255, 200, 200), 1, CV_AA);
    }
    cv::imshow("contours",dstimg);
    cv::imwrite("contours.jpg",dstimg);
    return 0;
}

int ImageCheck::CalcHistogram(cv::Mat srcimg, int thresh,int bin,cv::Mat& hist){
    if(srcimg.empty()||bin>256||bin<1)
    {return -1;}
    //cv::resize(srcimg,srcimg,cv::Size(srcimg.cols/2,srcimg.rows/2));
    int histSize = bin;//直方图分成多少个区间
    float range[] = { 0,256 };
    const float *histRanges = { range };//统计像素值的区间
    cv::calcHist(&srcimg,1,0,cv::Mat(),hist,1,&histSize,&histRanges,true,false);
    hist=hist.rowRange(thresh,hist.rows);
//    cout<<hist.type()<<endl;
//    cout<<hist.rows<<endl;
//    cout<<hist<<endl;
    float sum=0;
    for(int i=0;i<hist.rows;++i){
        sum+=hist.at<float>(i,0);
    }
    for(int i=0;i<hist.rows;++i){
        hist.at<float>(i,0)/=sum;
    }
    hist=hist.t();
    return 0;
}

cv::PCA ImageCheck::PCADecreaseDim(cv::Mat iuputMat,cv::Mat& outputMat, float percentage){
    cv::PCA pca(iuputMat,cv::Mat(),cv::PCA::DATA_AS_ROW,percentage);
    for(int i=0;i<iuputMat.rows;++i){
        cv::Mat point=pca.project(iuputMat.row(i));
        //cout<<point.size()<<endl;
        if(i==0){
            outputMat=point;
        }
        else{
            cv::vconcat(outputMat,point,outputMat);
        }
    }
//    cout<<pca.eigenvectors.size()<<endl;
//    cout<<pca.eigenvalues<<endl;
    int rows=iuputMat.rows;
    int cols=iuputMat.cols;
    FileStorage fs(std::to_string(rows)+std::to_string(cols)+"pca.xml",FileStorage::WRITE);
    pca.write(fs);
    fs.release();
    return pca;
}

void ImageCheck::Vecf2Mat32F(vector<float> vec, cv::Mat& outputMat){
    cv::Mat temp=cv::Mat::zeros(1,vec.size(),CV_32FC1);
    for(int i=0;i<vec.size();++i){
        temp.at<float>(0,i)= vec[i]/180.0;
        if(isnan(temp.at<float>(0,i)))
            cout<<i<<endl;
    }
    outputMat=temp.clone();
}

int ImageCheck::OutputMat(std::string filename,cv::Mat outputMat){
    ofstream out;
    out.open(filename);
    if(!out.is_open()){
         cout << "open dst File  Error opening file" << endl;
         return -1;
     }
    if(outputMat.type()==5){
        for(int i=0;i<outputMat.rows;++i){
            for(int j=0;j<outputMat.cols;++j){
                out<<outputMat.at<float>(i,j)<<" ";
            }
            out<<endl;
        }
    }
    else if(outputMat.type()==6){
        for(int i=0;i<outputMat.rows;++i){
            for(int j=0;j<outputMat.cols;++j){
                out<<outputMat.at<double>(i,j)<<" ";
            }
            out<<endl;
        }
    }
    else{
        for(int i=0;i<outputMat.rows;++i){
            for(int j=0;j<outputMat.cols;++j){
                out<<outputMat.at<int>(i,j)<<" ";
            }
            out<<endl;
        }
    }
    out.close();
    return 0;
}

cv::Ptr<cv::ml::SVM> ImageCheck::OneClassSVMmodel(cv::Mat traindata,float gamma, float nu, cv::String path){
    int labels[traindata.rows] = {0};
    cv::Mat labelsMat(traindata.rows,1,CV_32SC1,labels);
    //建立模型
    Ptr<cv::ml::SVM> model = cv::ml::SVM::create();
    model->setType(cv::ml::SVM::ONE_CLASS);
    model->setKernel(cv::ml::SVM::RBF);
    //model->setDegree(degree);
    model->setGamma(gamma);
    model->setC(1);
    //model->setP(0);
    model->setNu(nu);
    model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER,100,1e-6));
    model->train(traindata,cv::ml::ROW_SAMPLE,labelsMat);
    model->save(path+"_svm.xml");
    return model;
}

void ImageCheck::TestSVMmodel(cv::Ptr<cv::ml::SVM> svm,cv::Mat testdata,cv::Mat& resultlabel){
    svm->predict(testdata,resultlabel);
}

cv::Mat ImageCheck::GenerateWrongImage(cv::Mat srcimg, float hor){
    cv::Mat temp1=srcimg(Range::all(), Range(0,srcimg.cols*hor));
    cv::Mat temp2=srcimg(Range::all(), Range(srcimg.cols*hor,srcimg.cols));
    cv::Mat res;
    cv::hconcat(temp2,temp1,res);
    return res;
}

cv::Ptr<cv::ml::SVM> ImageCheck::ReadSVMFromXML(cv::String path){
    return Algorithm::load<cv::ml::SVM>(path);
}

cv::PCA ImageCheck::ReadPCAFromXML(cv::String path){
    FileStorage fs(path,FileStorage::READ);
    cv::PCA pca;
    pca.read(fs.root());
    fs.release();
    return pca;
}

cv::Mat ImageCheck::PCAdecrease(cv::PCA pca, cv::Mat inputMat){
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

vector<cv::Point> ImageCheck::SortContour(vector<cv::Point> contour){
    cv::Point2f center(0,0);
    for(int i=0;i<contour.size();++i){
        center.x+=contour[i].x;
        center.y+=contour[i].y;
    }
    center.x/=contour.size();
    center.y/=contour.size();
    multimap<float,cv::Point,greater<float>> sortedcontours;
    for(int i=0;i<contour.size();++i){
        float theta=GetAngle(center, cv::Point(center.x+10,center.y), center, contour[i]);
        if(contour[i].y>center.y)
            theta=360.0-theta;
        pair<float,cv::Point> fp(theta,contour[i]);
        sortedcontours.insert(fp);
    }
    vector<cv::Point> sortcontour(contour.size());
    multimap<float,cv::Point>::iterator i,iend;
    iend=sortedcontours.end();
    int count=0;
    for (i=sortedcontours.begin();i!=iend;i++){
       //cout<<(*i).second<< "dian "<<(*i).first<< "jiaodu"<<endl;
       sortcontour[count]=(*i).second;
       count++;
    }
    return sortcontour;
}

vector<cv::Point> ImageCheck::SortContourCenter(vector<cv::Point> contour, cv::Point center){
    multimap<float,cv::Point,greater<float>> sortedcontours;
    for(int i=0;i<contour.size();++i){
        float theta=GetAngle(center, cv::Point(center.x+10,center.y), center, contour[i]);
        if(contour[i].y>center.y)
            theta=360.0-theta;
        pair<float,cv::Point> fp(theta,contour[i]);
        sortedcontours.insert(fp);
    }
    vector<cv::Point> sortcontour(contour.size());
    multimap<float,cv::Point>::iterator i,iend;
    iend=sortedcontours.end();
    int count=0;
    for (i=sortedcontours.begin();i!=iend;i++){
       //cout<<(*i).second<< "dian "<<(*i).first<< "jiaodu"<<endl;
       sortcontour[count]=(*i).second;
       count++;
    }
    return sortcontour;
}

int ImageCheck::FindHistSVM(cv::Mat histMat, float& fp, float& fn){
    ofstream para;
    para.open("paras.txt");
    cv::Mat histoutput;
    cv::PCA pca=PCADecreaseDim(histMat,histoutput, 0.95);
    cout<<histoutput<<endl;
    for(float i=0.1;i<1.0;i=i+0.1){
        for(float j=0.1;j<1.0;j=j+0.1){
            cv::Ptr<cv::ml::SVM> svm=OneClassSVMmodel(histoutput,i,j,"hist");
            std::vector<cv::String> image_files;
            std::string filepath="/mnt/hgfs/linuxsharefiles/testImages/images/1-11w/*.jpg";
            cv::glob(filepath, image_files);
            cv::Mat histMat;
            for(int ii=0;ii<image_files.size();++ii){
                cv::Mat srcimg=cv::imread(image_files[ii],0);
                cv::Mat hist;
                CalcHistogram(srcimg,0,255,hist);
                if(ii==0){
                    histMat=hist;
                }
                else{
                    cv::vconcat(histMat,hist,histMat);
                }
            }
            cout<<histMat.size()<<endl;
            cv::Mat histoutput;
            histoutput=PCAdecrease(pca,histMat);
            //cout<<histoutput<<endl;
            cv::Mat histreslabels;
            TestSVMmodel(svm,histoutput,histreslabels);
            cout<<histreslabels<<endl;
            std::vector<int> realresult(80),preresult(80);
            for(int ii=0;ii<realresult.size();++ii){
                if(ii<40){
                    realresult[ii]=-1;
                }
                else{
                    realresult[ii]=1;
                }
            }
            for(int ii=0;ii<histreslabels.rows;++ii){
                if(histreslabels.at<float>(ii,0)<0.0001){
                    preresult[ii]=-1;
                }
                else{
                    preresult[ii]=1;
                }
            }
            ResultEvaluation reseva;
            CalcResult res=reseva(realresult,preresult);
            fp=res.F1_p;
            fn=res.F1_n;
            cout<<res.F1_p<<"    "<<res.F1_n<<endl;
            para<<i<<"  "<<j<<"  "<<fp<<"  "<<fn<<endl;
        }
    }
    para.close();
    return 0;
}

int ImageCheck::FindEdgeSVM(cv::Mat angleMat, float& fp, float& fn){
    ofstream para;
    para.open("parasedge.txt");
    cv::Mat angleoutput;
    cv::PCA pca=PCADecreaseDim(angleMat,angleoutput, 0.95);
    //cout<<angleoutput<<endl;
    for(float i=0.1;i<1.0;i=i+0.1){
        for(float j=0.1;j<1.0;j=j+0.1){
            cv::Ptr<cv::ml::SVM> anglesvm=OneClassSVMmodel(angleoutput,i,j,"angle");
            std::vector<cv::String> image_files;
            std::string filepath="/mnt/hgfs/linuxsharefiles/testImages/images/1-11w/*.jpg";
            cv::glob(filepath, image_files);
            cv::Mat angleMat;
            for(int ii=0;ii<image_files.size();++ii){
                cv::Mat srcimg=cv::imread(image_files[ii],0);
                vector<cv::Point> contour;
                GetContour(srcimg,contour);
                vector<float> angles;
                Contour2Angle(contour,6,6,angles);
                if(angles.size()>58){
                    angles.erase(angles.begin()+58,angles.end());
                }
                else if(angles.size()<58){
                    int as=angles.size();
                    for(int ij=0;ij<58-as;++ij){
                        angles.push_back(0.0);
                    }
                }
                cv::Mat contourangles;
                Vecf2Mat32F(angles,contourangles);
                if(ii==0){
                    angleMat=contourangles;
                }
                else{
                    cv::vconcat(angleMat,contourangles,angleMat);
                }
            }
            cout<<angleMat.size()<<endl;
            cv::Mat angleoutput;
            angleoutput=PCAdecrease(pca,angleMat);
            //cout<<angleoutput<<endl;
            cv::Mat anglereslabels;
            TestSVMmodel(anglesvm,angleoutput,anglereslabels);
            cout<<anglereslabels<<endl;
            std::vector<int> realresult(80),preresult(80);
            for(int ii=0;ii<realresult.size();++ii){
                if(ii<40){
                    realresult[ii]=-1;
                }
                else{
                    realresult[ii]=1;
                }
            }
            for(int ii=0;ii<anglereslabels.rows;++ii){
                if(anglereslabels.at<float>(ii,0)<0.0001){
                    preresult[ii]=-1;
                }
                else{
                    preresult[ii]=1;
                }
            }
            ResultEvaluation reseva;
            CalcResult res=reseva(realresult,preresult);
            fp=res.F1_p;
            fn=res.F1_n;
            cout<<res.F1_p<<"    "<<res.F1_n<<endl;
            para<<i<<"  "<<j<<"  "<<fp<<"  "<<fn<<endl;
        }
    }
    para.close();
    return 0;
}

cv::Mat ImageCheck::HistExtract(cv::String path, int resizenum){
    std::vector<cv::String> image_files;
    cv::glob(path, image_files);
    cv::Mat histMat;
    for(int i=0;i<image_files.size();++i){
        cv::Mat srcimg=cv::imread(image_files[i],0);
        cv::resize(srcimg,srcimg,cv::Size(srcimg.cols/resizenum,srcimg.rows/resizenum));
        cv::Mat hist;
        CalcHistogram(srcimg,0,255,hist);
        if(i==0){
            histMat=hist;
        }
        else{
            cv::vconcat(histMat,hist,histMat);
        }
    }
    cv::Mat histoutput;
    histoutput=PCAdecrease(ReadPCAFromXML("192255pca.xml"),histMat);
    return histoutput;
}

void ImageCheck::SaveXml(std::string xmlname, cv::Mat xmldata){
    FileStorage fs(xmlname, FileStorage::WRITE);
    fs << "ex" << xmldata;
    fs.release();
}

void ImageCheck::ReadXml(std::string xmlname, cv::Mat& xmldata){
    FileStorage fs(xmlname, FileStorage::READ);
    fs["ex"] >> xmldata;
}

cv::Mat ImageCheck::EdgeExtract(cv::String path){
    std::vector<cv::String> image_files;
    cv::glob(path, image_files);
    cv::Mat angleMat;
    for(int i=0;i<image_files.size();++i){
        cv::Mat srcimg=cv::imread(image_files[i],0);
        vector<cv::Point> contour;
        GetContour(srcimg,contour);
        vector<float> angles;
        Contour2Angle(contour,6,6,angles);
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
        Vecf2Mat32F(angles,contourangles);
        if(i==0){
            angleMat=contourangles;
        }
        else{
            cv::vconcat(angleMat,contourangles,angleMat);
        }
    }
    cv::Mat angleoutput;
    angleoutput=PCAdecrease(ReadPCAFromXML("19258pca.xml"),angleMat);
    return angleoutput;
}

int ImageCheck::KernelCurve(cv::Mat srcimg, int kernelsize, int row, vector<float>& curve){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    if(row<kernelsize/2||(srcimg.rows-row)<kernelsize/2||srcimg.rows<kernelsize||srcimg.cols<kernelsize)
        return -1;
    for(int i=0;i<srcimg.cols-kernelsize;++i){
        float aver=0;
        for(int m=0;m<kernelsize;++m){
            for(int n=0;n<kernelsize;++n){
                aver+=srcimg.at<uchar>(row-kernelsize/2+m,i+n);
            }
        }
        aver/=kernelsize*kernelsize;
        curve.push_back(aver);
    }
    return 0;
}
