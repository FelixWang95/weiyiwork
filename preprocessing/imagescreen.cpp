#include "imagescreen.h"
#include "NumeralCalculations.h"

ImageScreen::ImageScreen()
{

}

int ImageScreen::ImageCut(cv::String path, vector<vector<cv::Mat>>& partimgs){
    cv::String pattern_jpg;
    std::vector<cv::String> image_files;
    pattern_jpg = path;
    cv::glob(pattern_jpg, image_files);
    partimgs.resize(24);
    for(int i=0;i<partimgs.size();++i){
        partimgs[i].resize(image_files.size());
    }
    vector<cv::Rect> sections(4);
    sections[0]=cv::Rect(1,1,1,1);
    sections[1]=cv::Rect(1,1,1,1);
    sections[2]=cv::Rect(1,1,1,1);
    sections[3]=cv::Rect(1,1,1,1);
    vector<cv::Rect> sectionparts(24);
    for(int i=0; i<4; ++i){
        sectionparts.insert(sectionparts.begin()+i*6,RectCut(sections[i]).begin(),RectCut(sections[i]).end());
    }
    for(int i=0;i<image_files.size();++i){
        cv::Mat srcimg=cv::imread(image_files[i],0);
        for(int j=0;j<partimgs.size();++j){
            partimgs[j][i]=srcimg(sectionparts[j]);
        }
    }
    return 0;
}

int ImageScreen::ImageCuttest(cv::String path, vector<vector<cv::Mat>>& partimgs){
    cv::String pattern_jpg;
    std::vector<cv::String> image_files;
    pattern_jpg = path;
    cv::glob(pattern_jpg, image_files);
    partimgs.resize(6);
    for(int i=0;i<partimgs.size();++i){
        partimgs[i].resize(image_files.size());
    }
    vector<cv::Rect> sections(1);
    sections[0]=cv::Rect(cv::Point(1592,430),cv::Point(2045,685));
    vector<cv::Rect> sectionparts(6);
    for(int i=0; i<sections.size(); ++i){
        for(int j=0;j<6;++j){
            sectionparts[i*6+j]=RectCut(sections[i])[j];
        }
    }
    for(int i=0;i<image_files.size();++i){
        cv::Mat srcimg=cv::imread(image_files[i],0);
        for(int j=0;j<partimgs.size();++j){
            partimgs[j][i]=srcimg(sectionparts[j]);
        }
    }
    return 0;
}

vector<cv::Rect> ImageScreen::RectCut(cv::Rect section){
    vector<cv::Rect> sectionparts(6);
    sectionparts[0]=cv::Rect(section.x,section.y,section.width/3,section.height/2);
    sectionparts[1]=cv::Rect(section.x+section.width/3,section.y,section.width/3,section.height/2);
    sectionparts[2]=cv::Rect(section.x+section.width*2/3,section.y,section.width/3,section.height/2);
    sectionparts[3]=cv::Rect(section.x,section.y+section.height/2,section.width/3,section.height/2);
    sectionparts[4]=cv::Rect(section.x+section.width/3,section.y+section.height/2,section.width/3,section.height/2);
    sectionparts[5]=cv::Rect(section.x+section.width*2/3,section.y+section.height/2,section.width/3,section.height/2);
    return sectionparts;
}

int ImageScreen::PCAgeneration(vector<vector<cv::Mat>> partimgs,vector<cv::Mat>& traindata){
    traindata.resize(partimgs.size());
    for(int i=0;i<partimgs.size();++i){
        cv::Mat dataMat;
        for(int j=0;j<partimgs[i].size();++j){
            if(j==0){
                dataMat=Histgeneration(partimgs[i][j]);
            }
            else{
                cv::vconcat(dataMat,Histgeneration(partimgs[i][j]),dataMat);
            }
        }
        cv::PCA pca(dataMat,cv::Mat(),cv::PCA::DATA_AS_ROW,0.99);
        cv::Mat outputMat;
        for(int i=0;i<dataMat.rows;++i){
            cv::Mat point=pca.project(dataMat.row(i));
            //cout<<point.size()<<endl;
            if(i==0){
                outputMat=point;
            }
            else{
                cv::vconcat(outputMat,point,outputMat);
            }
        }
        traindata[i]=outputMat;
//        if(i==0){
//            cout<<outputMat.size()<<endl;
//            ofstream txtfile;
//            txtfile.open("histfeature.txt");
//            for(int i=0;i<outputMat.rows;++i){
//                for(int j=0;j<outputMat.cols;++j){
//                    txtfile<<outputMat.at<float>(i,j)<<" ";
//                }
//                txtfile<<endl;
//            }
//        }
        //cout<<pca.eigenvectors.size()<<endl;
        //cout<<pca.eigenvalues<<endl;
        cv::FileStorage fs("partindex"+std::to_string(i)+"pca.xml",cv::FileStorage::WRITE);
        pca.write(fs);
        fs.release();
    }
    return 0;
}

cv::Mat ImageScreen::Histgeneration(cv::Mat srcimg){
    cv::Mat hist;
    int histSize = 256;//直方图分成多少个区间
    float range[] = { 0,256 };
    const float *histRanges = { range };//统计像素值的区间
    cv::calcHist(&srcimg,1,0,cv::Mat(),hist,1,&histSize,&histRanges,true,false);
    //cout<<hist.type()<<endl;
    //cout<<hist.rows<<endl;
    //cout<<hist<<endl;
    float sum=0;
    for(int i=0;i<hist.rows;++i){
        sum+=hist.at<float>(i,0);
    }
    for(int i=0;i<hist.rows;++i){
        hist.at<float>(i,0)/=sum;
    }
    hist=hist.t();
    //cout<<hist<<endl;
    return hist;
}

cv::PCA ImageScreen::ReadPCAFromXML(cv::String path){
    cv::FileStorage fs(path,cv::FileStorage::READ);
    cv::PCA pca;
    pca.read(fs.root());
    fs.release();
    return pca;
}

cv::Ptr<cv::ml::SVM> ImageScreen::ReadSVMFromXML(cv::String path){
    return cv::Algorithm::load<cv::ml::SVM>(path);
}

cv::Mat ImageScreen::PCAdecrease(cv::PCA pca, cv::Mat inputMat){
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

int ImageScreen::SectionBrightness(vector<cv::Mat> partimgs, vector<cv::Mat>& PCAfeatures){
    PCAfeatures.resize(partimgs.size());
    ofstream txtfile;
    txtfile.open("testhist.txt");
    for(int i=0;i<partimgs.size();++i){
        cv::Mat dataMat=Histgeneration(partimgs[i]);
        PCAfeatures[i]=PCAdecrease(ReadPCAFromXML("partindex"+std::to_string(i)+"pca.xml"),dataMat);
        for(int j=0;j<PCAfeatures[i].cols;++j){
            txtfile<<PCAfeatures[i].at<float>(0,j)<<" ";
        }
        txtfile<<endl;
    }
    return 0;
}

void ImageScreen::WriteCSV(string filename, cv::Mat m)
{
   ofstream myfile;
   myfile.open(filename.c_str());
   myfile<< cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
   myfile.close();
}

int ImageScreen::OutputMat(std::string filename,cv::Mat outputMat){
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

int ImageScreen::FeatureGeneration(cv::Mat srcimg, vector<float>& labels){
    vector<cv::Rect> sections(4);
    sections[0]=cv::Rect(1,1,1,1);
    sections[1]=cv::Rect(1,1,1,1);
    sections[2]=cv::Rect(1,1,1,1);
    sections[3]=cv::Rect(1,1,1,1);
    vector<cv::Rect> sectionparts(24);
    for(int i=0; i<sections.size(); ++i){
        for(int j=0;j<6;++j){
            sectionparts[i*6+j]=RectCut(sections[i])[j];
        }
    }
    vector<cv::Mat> partimgs(sectionparts.size());
    for(int i=0;i<partimgs.size();++i){
        partimgs[i]=srcimg(sectionparts[i]);
    }
    vector<cv::Mat> PCAfeatures;
    SectionBrightness(partimgs,PCAfeatures);
    labels.resize(PCAfeatures.size());
    for(int i=0;i<PCAfeatures.size();++i){
        cv::Ptr<cv::ml::SVM> histsvm=ReadSVMFromXML("hist"+std::to_string(i)+"_svm.xml");
        cv::Mat resultlabel;
        histsvm->predict(PCAfeatures[i],resultlabel);
        labels[i]=resultlabel.at<float>(0,0);
        cout<<labels[i]<<endl;
    }
    return 0;
}

int ImageScreen::FeatureGenerationtest(cv::Mat srcimg){
    vector<cv::Rect> sections(1);
    sections[0]=cv::Rect(cv::Point(1592,430),cv::Point(2045,685));
    cv::imshow("sample",srcimg(sections[0]));
    vector<cv::Rect> sectionparts(6);
    for(int i=0; i<sections.size(); ++i){
        for(int j=0;j<6;++j){
            sectionparts[i*6+j]=RectCut(sections[i])[j];
        }
    }
    vector<cv::Mat> partimgs(sectionparts.size());
    for(int i=0;i<partimgs.size();++i){
        partimgs[i]=srcimg(sectionparts[i]);
    }
    vector<cv::Mat> PCAfeatures;
    SectionBrightness(partimgs,PCAfeatures);
    vector<float> labels(PCAfeatures.size());
    for(int i=0;i<PCAfeatures.size();++i){
        cv::Ptr<cv::ml::SVM> histsvm=ReadSVMFromXML("hist"+std::to_string(i)+"_svm.xml");
        cv::Mat resultlabel;
        //cout<<PCAfeatures[i]<<endl;
        histsvm->predict(PCAfeatures[i],resultlabel);
        labels[i]=resultlabel.at<float>(0,0);
        cout<<resultlabel<<endl;
    }
    return 0;
}

int ImageScreen::PathToModelHist(cv::String path){
    vector<vector<cv::Mat>> partimgs;
    ImageCuttest(path, partimgs);
    vector<cv::Mat> traindata;
    PCAgeneration(partimgs,traindata);
    for(int index=0;index<traindata.size();++index){
        vector<int> sumvec(90);
        int cnt=0;
        for(float i=0.1;i<1.1;i=i+0.1){
            for(float j=0.1;j<1;j=j+0.1){
                cv::Ptr<cv::ml::SVM> histsvm=OneClassSVMmodel(traindata[index],i, j, "hist"+std::to_string(index));
                cv::Mat resultlabel;
                histsvm->predict(traindata[index],resultlabel);
                //cout<<resultlabel<<endl;
                int sum=0;
                for(int i=0;i<resultlabel.rows;++i){
                    sum+=resultlabel.at<float>(i,0);
                }
                //cout<<i<<" "<<j<<endl;
                //cout<<sum<<endl;
                sumvec[cnt]=sum;
                cnt++;
            }
        }
        auto max = std::max_element(std::begin(sumvec), std::end(sumvec));
        int positionmax = std::distance(std::begin(sumvec),max);
        float gamma=positionmax/9*0.1+0.1;
        float nu=positionmax%9*0.1+0.1;
        cout<<gamma<<" "<<nu<<endl;
        OneClassSVMmodel(traindata[index],gamma, nu, "hist"+std::to_string(index));
    }
    return 0;
}






int ImageScreen::EdgeSelect(cv::String path, vector<vector<vector<float>>>& partcurves){
    cv::String pattern_jpg;
    std::vector<cv::String> image_files;
    pattern_jpg = path;
    cv::glob(pattern_jpg, image_files);
    partcurves.resize(50);
    for(int i=0;i<partcurves.size();++i){
        partcurves[i].resize(image_files.size());
    }
    vector<cv::Rect> sections(5);
    sections[0]=cv::Rect(1,1,1,1);
    sections[1]=cv::Rect(1,1,1,1);
    sections[2]=cv::Rect(1,1,1,1);
    sections[3]=cv::Rect(1,1,1,1);
    sections[4]=cv::Rect(1,1,1,1);
    for(int i=0;i<image_files.size();++i){
        cv::Mat srcimg=cv::imread(image_files[i],0);
        for(int j=0;j<sections.size();++j){
            cv::Mat sectionimg=srcimg(sections[j]);
            for(int k=0;k<10;++k){
                partcurves[j*10+k][i]=GetCurve(sectionimg,j+1)[k];
            }
        }
    }
    return 0;
}

int ImageScreen::EdgeSelectTest(cv::String path, vector<vector<vector<float>>>& partcurves){
    cv::String pattern_jpg;
    std::vector<cv::String> image_files;
    pattern_jpg = path;
    cv::glob(pattern_jpg, image_files);
    partcurves.resize(10);
    for(int i=0;i<partcurves.size();++i){
        partcurves[i].resize(image_files.size());
    }
    vector<cv::Rect> sections(1);
    sections[0]=cv::Rect(1876,147,847,94);
//    sections[1]=cv::Rect(42,1120,128,704);
//    sections[2]=cv::Rect(3605,789,96,690);
    for(int i=0;i<image_files.size();++i){
        cv::Mat srcimg=cv::imread(image_files[i],0);
        for(int j=0;j<sections.size();++j){
            cv::Mat sectionimg=srcimg(sections[j]);
            for(int k=0;k<10;++k){
                partcurves[j*10+k][i]=GetCurve(sectionimg,j+1)[k];
            }
        }
    }
    return 0;
}

vector<vector<float>> ImageScreen::GetCurve(cv::Mat srcimg, int num){
    vector<vector<float>> sectioncurves;
    if(num==1){
        vector<int> startpos{1,85,170,250,335,415,500,585,665,750};
        sectioncurves=SectionCurve(srcimg,startpos,SCAN_DIRECTION_DOWN);
    }
    else if(num==2){
        vector<int> startpos{1,75,150,225,300,375,450,525,600,675};
        sectioncurves=SectionCurve(srcimg,startpos,SCAN_DIRECTION_RIGHT);
    }
    else if(num==3){
        vector<int> startpos{1,70,140,210,280,350,420,490,560,630};
        sectioncurves=SectionCurve(srcimg,startpos,SCAN_DIRECTION_LEFT);
    }
    else if(num==4){
        vector<int> startpos{1,1,1,1,1,1,1,1,1,1};
        sectioncurves=SectionCurve(srcimg,startpos,0);
    }
    else if(num==5){
        vector<int> startpos{1,1,1,1,1,1,1,1,1,1};
        sectioncurves=SectionCurve(srcimg,startpos,0);
    }
    return sectioncurves;
}

vector<vector<float>> ImageScreen::SectionCurve(cv::Mat srcimg, vector<int> startpos, int orientation){
    vector<vector<float>> sectioncurves(startpos.size());
    int kernel=2;
    if(orientation==SCAN_DIRECTION_RIGHT){
        vector<float> curve(srcimg.cols-kernel);
        for(int i=0;i<startpos.size();++i){
            for(int j=0;j<srcimg.cols-kernel;++j){
                curve[j]=(float)(srcimg.at<uchar>(startpos[i],j)+srcimg.at<uchar>(startpos[i],j+1)+srcimg.at<uchar>(startpos[i]+1,j)+srcimg.at<uchar>(startpos[i]+1,j+1))/(kernel*kernel);
            }
            sectioncurves[i]=curve;
        }
    }
    else if(orientation==SCAN_DIRECTION_LEFT){
        vector<float> curve(srcimg.cols-kernel);
        for(int i=0;i<startpos.size();++i){
            for(int j=0;j<srcimg.cols-kernel;++j){
                curve[j]=(float)(srcimg.at<uchar>(startpos[i],srcimg.cols-kernel-j)+srcimg.at<uchar>(startpos[i],srcimg.cols-kernel-j+1)+srcimg.at<uchar>(startpos[i]+1,srcimg.cols-kernel-j)+srcimg.at<uchar>(startpos[i]+1,srcimg.cols-kernel-j+1))/(kernel*kernel);
            }
            sectioncurves[i]=curve;
        }
    }
    else if(orientation==SCAN_DIRECTION_UP){
        vector<float> curve(srcimg.rows-kernel);
        for(int i=0;i<startpos.size();++i){
            for(int j=0;j<srcimg.rows-kernel;++j){
                curve[j]=(float)(srcimg.at<uchar>(srcimg.rows-kernel-j,startpos[i])+srcimg.at<uchar>(srcimg.rows-kernel-j+1,startpos[i])+srcimg.at<uchar>(srcimg.rows-kernel-j,startpos[i]+1)+srcimg.at<uchar>(srcimg.rows-kernel-j+1,startpos[i]+1))/(kernel*kernel);
            }
            sectioncurves[i]=curve;
        }
    }
    else if(orientation==SCAN_DIRECTION_DOWN){
        vector<float> curve(srcimg.rows-kernel);
        for(int i=0;i<startpos.size();++i){
            for(int j=0;j<srcimg.rows-kernel;++j){
                curve[j]=(float)(srcimg.at<uchar>(j,startpos[i])+srcimg.at<uchar>(j+1,startpos[i])+srcimg.at<uchar>(j,startpos[i]+1)+srcimg.at<uchar>(j+1,startpos[i]+1))/(kernel*kernel);
            }
            sectioncurves[i]=curve;
        }
    }
    return sectioncurves;
}

vector<float> ImageScreen::FitCurves(vector<float> curves, int times){

}

int ImageScreen::EdgeCurves(cv::Mat srcimg){

}

int ImageScreen::ReadTxt(std::string txtname, cv::Mat& txtdata){
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
                    if(tmp[i]=='\t'){
                        colnum++;
                    }
                }
            }
            n++;
        }
    }
    file1.close();
    fstream file2;
    file2.open(txtname);
    txtdata=cv::Mat::zeros(n,colnum+1,CV_32FC1);
    //将txt文件数据写入到Data矩阵中
    for (int i = 0; i < txtdata.rows; ++i)
    {
        for (int j = 0; j < txtdata.cols; ++j)
        {
            file2 >> txtdata.at<float>(i, j);
            //cout<<txtdata.at<float>(i, j)<<endl;
        }
    }
    file2.close();
    return 0;
}

cv::Ptr<cv::ml::SVM> ImageScreen::OneClassSVMmodel(cv::Mat traindata,float gamma, float nu, std::string path){
    int labels[traindata.rows] = {0};
    cv::Mat labelsMat(traindata.rows,1,CV_32FC1,labels);
    //建立模型
    cv::Ptr<cv::ml::SVM> model = cv::ml::SVM::create();
    model->setType(cv::ml::SVM::ONE_CLASS);
    model->setKernel(cv::ml::SVM::RBF);
    //model->setDegree(degree);
    model->setGamma(gamma);
    //model->setC(1);
    //model->setP(0);
    model->setNu(nu);
    model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER,100,1e-6));
    model->train(traindata,cv::ml::ROW_SAMPLE,labelsMat);
    model->save(path+"_svm.xml");
    return model;
}

int ImageScreen::ReadandGen(std::string txtname){
    cv::Mat traindata;
    ReadTxt(txtname, traindata);
    vector<int> sumvec(90);
    int cnt=0;
    for(float i=0.1;i<1.1;i=i+0.1){
        for(float j=0.1;j<1;j=j+0.1){
            cv::Ptr<cv::ml::SVM> edgesvm=OneClassSVMmodel(traindata,i, j, "edge9");
            cv::Mat resultlabel;
            edgesvm->predict(traindata,resultlabel);
            //cout<<resultlabel<<endl;
            int sum=0;
            for(int i=0;i<resultlabel.rows;++i){
                sum+=resultlabel.at<float>(i,0);
            }
            cout<<i<<" "<<j<<endl;
            cout<<sum<<endl;
            sumvec[cnt]=sum;
            cnt++;
        }
    }
    auto min = std::min_element(std::begin(sumvec), std::end(sumvec));
    int positionmin = std::distance(std::begin(sumvec),min);
    float gamma=positionmin/9*0.1+0.1;
    float nu=positionmin%9*0.1+0.1;
    OneClassSVMmodel(traindata,gamma, nu, "edge9");
    return 0;
}

int ImageScreen::GenSVM(vector<vector<float>> feadata, int index){
    cv::Mat traindata(0, feadata[0].size(), cv::DataType<float>::type);
    for (unsigned int i = 0; i < feadata.size(); ++i)
    {
      cv::Mat Sample(1, feadata[0].size(), cv::DataType<float>::type, feadata[i].data());
      traindata.push_back(Sample);
    }
    vector<int> sumvec(90);
    int cnt=0;
    for(float i=0.1;i<1.1;i=i+0.1){
        for(float j=0.1;j<1;j=j+0.1){
            cv::Ptr<cv::ml::SVM> edgesvm=OneClassSVMmodel(traindata,i, j, "edge"+std::to_string(index));
            cv::Mat resultlabel;
            edgesvm->predict(traindata,resultlabel);
            //cout<<resultlabel<<endl;
            int sum=0;
            for(int i=0;i<resultlabel.rows;++i){
                sum+=resultlabel.at<float>(i,0);
            }
            //cout<<i<<" "<<j<<endl;
            //cout<<sum<<endl;
            sumvec[cnt]=sum;
            cnt++;
        }
    }
    auto min = std::min_element(std::begin(sumvec), std::end(sumvec));
    int positionmin = std::distance(std::begin(sumvec),min);
    float gamma=positionmin/9*0.1+0.1;
    float nu=positionmin%9*0.1+0.1;
    OneClassSVMmodel(traindata,gamma, nu, "edge"+std::to_string(index));
    return 0;
}

int ImageScreen::CurveFeatureGeneration(cv::Mat srcimg, vector<float>& labels){
    vector<cv::Rect> sections(5);
    sections[0]=cv::Rect(1876,147,847,94);
    sections[1]=cv::Rect(42,1120,128,704);
    sections[2]=cv::Rect(3605,789,96,690);
    sections[3]=cv::Rect(1,1,1,1);
    sections[4]=cv::Rect(1,1,1,1);
    vector<vector<float>> partcurves(sections.size()*10);
    for(int i=0;i<sections.size();++i){
        cv::Mat sectionimg=srcimg(sections[i]);
        for(int j=0;j<10;++j){
            partcurves[i*10+j]=GetCurve(sectionimg,i+1)[j];
        }
    }
    labels.resize(partcurves.size());
    for(int i=0;i<partcurves.size();++i){
        vector<float> edgefea=FitCurves(partcurves[i],4);
        cv::Ptr<cv::ml::SVM> histsvm=ReadSVMFromXML("edge"+std::to_string(i)+"_svm.xml");
        cv::Mat resultlabel;
        histsvm->predict(edgefea,resultlabel);
        labels[i]=resultlabel.at<float>(0,0);
        cout<<labels[i]<<endl;
    }
    return 0;
}

int ImageScreen::CurveFeatureGenerationtest(cv::Mat srcimg){
    vector<cv::Rect> sections(1);
    sections[0]=cv::Rect(1876,147,847,94);
//    sections[1]=cv::Rect(42,1120,128,704);
//    sections[2]=cv::Rect(3605,789,96,690);
    vector<vector<float>> partcurves(sections.size()*10);
    for(int i=0;i<sections.size();++i){
        cv::Mat sectionimg=srcimg(sections[i]);
        for(int j=0;j<10;++j){
            partcurves[i*10+j]=GetCurve(sectionimg,i+1)[j];
        }
    }
    //1
//    ofstream edgetxt;
//    edgetxt.open("edgecurve.txt");
//    for(int i=0;i<partcurves.size();++i){
//        for(int j=0;j<partcurves[i].size();++j){
//            edgetxt<<partcurves[i][j]<<" ";
//        }
//        edgetxt<<endl;
//    }
//    edgetxt.close();
    //2
//    vector<float> labels(partcurves.size());
    ofstream txtfile;
    txtfile.open("testcurvesample.txt");
    for(int i=0;i<partcurves.size();++i){
        NumeralCalculations nc;
        vector<float> ptx(partcurves[i].size());
        for(int j=0;j<partcurves[i].size();++j){
            ptx[j]=j;
        }
        vector<float> edgefea;
        nc.CubicSplineTrain(ptx,partcurves[i],edgefea);
        for(int j=0;j<edgefea.size();++j){
            txtfile<<edgefea[j]<<" ";
        }
        txtfile<<endl;
        cv::Mat pcaedgefea = PCAdecrease(ReadPCAFromXML("edgeindex"+std::to_string(i)+"pca.xml"), cv::Mat(edgefea).t());
        if(i==0){
            OutputMat("edgefeatest.txt",pcaedgefea);
        }
//        cv::Ptr<cv::ml::SVM> histsvm=ReadSVMFromXML("edge"+std::to_string(i)+"_svm.xml");
//        cv::Mat resultlabel;
//        histsvm->predict(edgefea,resultlabel);
//        labels[i]=resultlabel.at<float>(0,0);
//        cout<<labels[i]<<endl;
    }
    txtfile.close();
    return 0;
}

int ImageScreen::ClassifyFeature(std::string txtname){
    cv::Mat testdata;
    ReadTxt(txtname, testdata);
    vector<float> labels(testdata.rows);
    for(int i=0;i<testdata.rows;++i){
        cv::Mat single = testdata.rowRange(0+i, 1+i);
        cout<<single<<endl;
        cv::Ptr<cv::ml::SVM> histsvm=ReadSVMFromXML("edge"+std::to_string(i)+"_svm.xml");
        cv::Mat resultlabel;
        histsvm->predict(single,resultlabel);
        labels[i]=resultlabel.at<float>(0,0);
        cout<<resultlabel<<endl;
    }
    return 0;
}

int ImageScreen::PathToModelEdge(cv::String path){
    vector<vector<vector<float>>> partcurves;
    EdgeSelectTest(path,partcurves);
    for(int i=0;i<partcurves.size();++i){
        cv::Mat edgefeaimgs;
        for(int j=0;j<partcurves[i].size();++j){
            NumeralCalculations nc;
            vector<float> ptx(partcurves[i][j].size());
            for(int k=0;k<partcurves[i][j].size();++k){
                ptx[k]=k;
            }
            vector<float> edgefea;
            nc.CubicSplineTrain(ptx,partcurves[i][j],edgefea);
            cv::Mat edgefeaM=cv::Mat(edgefea).t();
            if(j==0){
                edgefeaimgs=edgefeaM;
            }
            else{
                cv::vconcat(edgefeaimgs,edgefeaM,edgefeaimgs);
            }
        }
        if(i==0){
            OutputMat("edgefea.txt",edgefeaimgs);
        }
        cv::PCA pca(edgefeaimgs,cv::Mat(),cv::PCA::DATA_AS_ROW,0.99);
        cv::Mat outputMat;
        for(int i=0;i<edgefeaimgs.rows;++i){
            cv::Mat point=pca.project(edgefeaimgs.row(i));
            //cout<<point.size()<<endl;
            if(i==0){
                outputMat=point;
            }
            else{
                cv::vconcat(outputMat,point,outputMat);
            }
        }
        cv::FileStorage fs("edgeindex"+std::to_string(i)+"pca.xml",cv::FileStorage::WRITE);
        pca.write(fs);
        fs.release();
        //GenSVM(outputMat, i);
    }
    return 0;
}

bool ImageScreen::ScoreEvaluation(vector<float> histlabels, vector<float> edgelabels){
    int histscore=20;
    int edgescore=10;
    int threshold=100;
    int totalsocre=0;
    for(int i=0;i<histlabels.size();++i){
        totalsocre+=(histscore*histlabels[i]);
    }
    for(int i=0;i<edgelabels.size();++i){
        totalsocre+=(edgescore*edgelabels[i]);
    }
    if(totalsocre>threshold)
        return false;
    else
        return true;
}

int ImageScreen::Pixel2Curve(cv::Mat srcimg, vector<int> testpos, int orientation, vector<vector<float>>& curves){
    curves.clear();
    if(orientation==SCAN_DIRECTION_RIGHT){//从左往右
        for(int i=0;i<testpos.size();++i){
            vector<float> curve;
            for(int j=5;j<srcimg.cols-1;++j){
                float kerneldata=(srcimg.at<uchar>(testpos[i],j)+srcimg.at<uchar>(testpos[i],j+1)+srcimg.at<uchar>(testpos[i]+1,j)+srcimg.at<uchar>(testpos[i]+1,j+1))/4;
                curve.push_back(kerneldata);
            }
            curves.push_back(curve);
        }
    }
    else if(orientation==SCAN_DIRECTION_DOWN){//从上往下
        for(int i=0;i<testpos.size();++i){
            vector<float> curve;
            for(int j=5;j<srcimg.rows-1;++j){
                float kerneldata=(srcimg.at<uchar>(j,testpos[i])+srcimg.at<uchar>(j,testpos[i]+1)+srcimg.at<uchar>(j+1,testpos[i])+srcimg.at<uchar>(j+1,testpos[i]+1))/4;
                curve.push_back(kerneldata);
            }
            curves.push_back(curve);
        }
    }
    else if(orientation==SCAN_DIRECTION_LEFT){//从右往左
        for(int i=0;i<testpos.size();++i){
            vector<float> curve;
            for(int j=srcimg.cols-1-5;j>0;--j){
                float kerneldata=(srcimg.at<uchar>(testpos[i],j)+srcimg.at<uchar>(testpos[i],j-1)+srcimg.at<uchar>(testpos[i]+1,j)+srcimg.at<uchar>(testpos[i]+1,j-1))/4;
                curve.push_back(kerneldata);
            }
            curves.push_back(curve);
        }
    }
    else if(orientation==SCAN_DIRECTION_UP){//从下往上
        for(int i=0;i<testpos.size();++i){
            vector<float> curve;
            for(int j=srcimg.rows-1-5;j>0;--j){
                float kerneldata=(srcimg.at<uchar>(j,testpos[i])+srcimg.at<uchar>(j,testpos[i]+1)+srcimg.at<uchar>(j-1,testpos[i])+srcimg.at<uchar>(j-1,testpos[i]+1))/4;
                curve.push_back(kerneldata);
            }
            curves.push_back(curve);
        }
    }
    return 0;
}

int ImageScreen::GetTransPos(vector<vector<float>> curves, vector<vector<int>>& pos, int thresh){
    if(curves.size()==0||thresh<=0||thresh>=255)
        return -1;
    pos.resize(curves.size());
    for(int i=0;i<curves.size();++i){
        vector<int> linepos;
        if(curves[i].size()<=6)
            return -1;
        for(int j=2;j<curves[i].size()-3;++j){
            float firstpoint=(curves[i][j-2]+curves[i][j-1]+curves[i][j]+curves[i][j+1]+curves[i][j+2])/5;
            float secondpoint=(curves[i][j-1]+curves[i][j]+curves[i][j+1]+curves[i][j+2]+curves[i][j+3])/5;
            if((firstpoint>thresh&&secondpoint<=thresh)||(firstpoint<=thresh&&secondpoint>thresh))
                linepos.push_back(j);
        }
        if(linepos.size()==0)
            linepos.push_back(-1);
        pos[i]=linepos;
    }
    return 0;
}











