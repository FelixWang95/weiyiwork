#include "preprocess.h"

preprocess::preprocess()
{

}

void preprocess::SVMTest(){
//    cv::RNG rng = theRNG();

//    cv::Mat tp1(20,3,CV_32FC1);
//    cv::Mat tp2(20,3,CV_32FC1);
//    cv::Mat tp3(20,3,CV_32FC1);

//    rng.fill(tp1,RNG::UNIFORM,0.f,1.f);
//    rng.fill(tp2,RNG::UNIFORM,2.f,4.f);
//    rng.fill(tp3,RNG::UNIFORM,5.f,6.f);

//    cv::Mat tq1=cv::Mat::ones(20,1,CV_32SC1);
//    cv::Mat tq2=cv::Mat::ones(20,1,CV_32SC1)*2;
//    cv::Mat tq3=cv::Mat::ones(20,1,CV_32SC1)*3;

//    cv::Mat tt1(5,3,CV_32FC1);
//    cv::Mat tt2(5,3,CV_32FC1);
//    cv::Mat tt3(5,3,CV_32FC1);

//    rng.fill(tt1,RNG::UNIFORM,0.f,1.f);
//    rng.fill(tt2,RNG::UNIFORM,2.f,4.f);
//    rng.fill(tt3,RNG::UNIFORM,5.f,6.f);

//    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
//    svm->setType(cv::ml::SVM::C_SVC);//可以处理非线性分割的问题
//    svm->setKernel(cv::ml::SVM::LINEAR);//径向基函数

//    cv::Mat trainData,testData; // 每行为一个样本
//    cv::Mat labels,result;
//    vconcat(tp1,tp2,trainData);
//    vconcat(trainData,tp3,trainData);
//    vconcat(tq1,tq2,labels);
//    vconcat(labels,tq3,labels);

//    vconcat(tt1,tt2,testData);
//    vconcat(testData,tt3,testData);

//    svm->train( trainData , cv::ml::ROW_SAMPLE , labels );

//    svm->predict(testData,result);
//    std::cout<<result<<endl;
//    svm->save("/home/qzs/DataFile/WyData/Tmp/xxsvm.xml");
}

float preprocess::Point2Line(cv::Point Point, int flag){
    float A,B,C;
    if(flag==1){
        A=top.y-cross.y;
        B=cross.x-top.x;
        C=(top.x-cross.x)*cross.y-cross.x*(top.y-cross.y);
    }
    else if(flag==0){
        A=left.y-cross.y;
        B=cross.x-left.x;
        C=(left.x-cross.x)*cross.y-cross.x*(left.y-cross.y);
    }
    else{
        return -1;
    }
    float dist;
    dist=fabs((A*Point.x+B*Point.y+C)/sqrt(A*A+B*B));
    return dist;
}

int preprocess::GetDatum(cv::Mat srcimg,vector<cv::Point2f>& datum){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat toplineimg=srcimg(cv::Rect(cv::Point(1806,56),cv::Point(2702,309)));
    cv::Mat topedge = krisch(toplineimg);
    cv::threshold(topedge, topedge, 150, 255, cv::THRESH_BINARY);
    cv::imshow("top",topedge);
    cv::imwrite("topedge.jpg",topedge);
    vector<cv::Point> toplinecontour;
    ScanRect(topedge,3,toplinecontour,5,0.8);
    for(int i=0;i<toplinecontour.size();++i){
        toplinecontour[i].x+=1806;
        toplinecontour[i].y+=56;
    }
    cv::Vec2f toplinepara;
    int toprtn=LineFit(toplinecontour,toplinepara);
    cv::Point2f topline;
    if(toprtn==0&&fabs(toplinepara[0])<=10){
        topline.x=(2702+1806)/2;
        topline.y=toplinepara[0]*topline.x+toplinepara[1];
    }
    else if(toprtn==0&&fabs(toplinepara[0])>10){
        topline.y=(56+309)/2;
        topline.x=(topline.y-toplinepara[1])/toplinepara[0];
    }
    cout<<topline<<endl;
    cv::Mat leftlineimg=srcimg(cv::Rect(cv::Point(32,1120),cv::Point(192,1760)));
    cv::Mat leftedge = krisch(leftlineimg);
    cv::threshold(leftedge, leftedge, 150, 255, cv::THRESH_BINARY);
    cv::imshow("left",leftedge);
    vector<cv::Point> leftlinecontour;
    ScanRect(leftedge,2,leftlinecontour,3,0.8);
    for(int i=0;i<leftlinecontour.size();++i){
        leftlinecontour[i].x+=32;
        leftlinecontour[i].y+=1120;
    }
    cv::Vec2f leftlinepara;
    int leftrtn=LineFit(leftlinecontour,leftlinepara);
    cv::Point2f leftline;
    if(leftrtn==0&&fabs(leftlinepara[0])<=10){
        leftline.x=(32+192)/2;
        leftline.y=leftlinepara[0]*leftline.x+leftlinepara[1];
    }
    else if(leftrtn==0&&fabs(leftlinepara[0])>10){
        leftline.y=(1120+1760)/2;
        leftline.x=(leftline.y-leftlinepara[1])/leftlinepara[0];
    }
    else{
        leftline.y=(1120+1760)/2;
        leftline.x=leftlinecontour[0].x;
    }
    cout<<leftline<<endl;
    cv::Point2f crosspoint;
    if(toprtn==0&&leftrtn==0){
        crosspoint.x=(leftlinepara[1]-toplinepara[1])/(toplinepara[0]-leftlinepara[0]);
        crosspoint.y=crosspoint.x*toplinepara[0]+toplinepara[1];
    }
    cout<<crosspoint<<endl;
    datum.clear();
    datum.push_back(topline);
    datum.push_back(leftline);
    datum.push_back(crosspoint);
    return 0;
}

void preprocess::SVMmodel(cv::Mat traindata,cv::Mat labels,cv::String path){
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);//可以处理非线性分割的问题
    svm->setKernel(cv::ml::SVM::LINEAR);//径向基函数
    svm->train( traindata , cv::ml::ROW_SAMPLE , labels );
//    svm->predict(testData,result);
//    std::cout<<result<<endl;
    cv::String pathname=path+"_svm.xml";
    svm->save(pathname);
}

int preprocess::PCADecreaseDim(cv::Mat iuputMat,cv::Mat& outputMat, float percentage){
    if(iuputMat.empty()){
        return -1;
    }
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
    cout<<pca.eigenvectors.size()<<endl;
    cout<<pca.eigenvalues<<endl;
    return 0;
}

int preprocess::ReadImagePath(std::string filename, std::vector<cv::String>& image_files){
    cv::String pattern_jpg;
    pattern_jpg = filename;
    cv::glob(pattern_jpg, image_files);
    return 0;
}

int preprocess::CalcHistogram(cv::Mat srcimg, int thresh,int bin,cv::Mat& hist){
    if(srcimg.empty()||bin>256||bin<1)
    {return -1;}
    //cv::resize(srcimg,srcimg,cv::Size(srcimg.cols/2,srcimg.rows/2));
    int histSize = bin;//直方图分成多少个区间
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
    hist=hist.rowRange(thresh,hist.rows);
    hist=hist.t();
    //cv::imshow("item",srcimg);
    return 0;
}

int preprocess::ReadTxt(std::string txtname, cv::Mat& txtdata){
    fstream file1;
    int n=0;
    string tmp;
    int colnum;
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
                colnum=tmp.size();
            }
            n++;
        }
    }
    file1.close();
    fstream file2;
    file2.open(txtname);
    txtdata=cv::Mat::zeros(n,colnum,CV_32S);
    //将txt文件数据写入到Data矩阵中
    for (int i = 0; i < txtdata.rows; ++i)
    {
        for (int j = 0; j < txtdata.cols; ++j)
        {
            file2 >> txtdata.at<int>(i, j);
            //cout<<txtdata.at<int>(i, j)<<endl;
        }
    }
    file2.close();
    return 0;
}

void preprocess::ReadXml(std::string xmlname, cv::Mat& xmldata){//read xml to Mat
    cv::FileStorage fs(xmlname.c_str(),cv::FileStorage::READ);
    fs["voc"]>>xmldata;
    fs.release();
}

void preprocess::ReadName(std::string txtname, vector<cv::String>& txtdata){
    fstream file1;
    string tmp;
    file1.open(txtname);
    if(!file1.is_open())//文件打开失败:返回0
    {
        return;
    }
    else//文件存在
    {
        while(getline(file1,tmp,'\n'))
        {
            txtdata.push_back(tmp);
        }
    }
    file1.close();
}

int preprocess::OutputMat(std::string filename,cv::Mat outputMat){
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

int preprocess::ClassifyImages(std::vector<cv::String> image_files,cv::Mat labels,std::string filename){
    if(labels.empty()){
        return -1;
    }
    double minv = 0.0, maxv = 0.0;
    cv::minMaxIdx(labels,&minv,&maxv);
    string folderPath = filename+std::to_string((int)maxv+1)+"/";
    for(int i=0;i<=maxv;++i){
        string command;
        command = "mkdir -p " + folderPath +std::to_string(i);
        system(command.c_str());
        //rm -rf +dir
        command="rm -rf "+ folderPath + std::to_string(i)+"/*";
        system(command.c_str());
    }
    if(labels.rows!=image_files.size())
        return -1;
    for(int i=0;i<labels.rows;++i){
        std::string source = image_files[i];//源文件
        std::string imgname;
        for(int j=image_files[i].size()-1;j>=0;--j){
            if(image_files[i][j]=='/'){
                imgname=image_files[i].substr(j,image_files[i].size()-j);
                //cout<<imgname<<endl;
                break;
            }
        }
        std::string destination = folderPath + std::to_string(labels.at<int>(i,0))+imgname;//目标文件
        CopyFile(source, destination);
    }
    return 0;
}

float preprocess::CalibrationImgOri(cv::Mat srcimg, int orientation, int threshold, int radius, float percent){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat edge = krisch(srcimg);
    cv::threshold(edge, edge, threshold, 255, cv::THRESH_BINARY);
    vector<cv::Point> linecontour;
    ScanRect(edge,orientation,linecontour,radius,percent);
    cv::Vec2f linepara;
    int rtn=LineFit(linecontour,linepara);
    if(rtn==-1)
        return 90.0;
    //cout<<linepara<<endl;
    float angle=atan(linepara[0])*180.0/PI;
    return angle;
}

void preprocess::RotateImg(cv::Mat srcimg, cv::Mat& dstimg, float angle){
    cv::Point center = cv::Point(srcimg.cols / 2, srcimg.rows / 2);//旋转中心
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);//获得仿射变换矩阵
    cv::Size dst_sz(srcimg.cols, srcimg.rows);
    cv::warpAffine(srcimg, dstimg, rot_mat, dst_sz);
}

int preprocess::ScanRect(cv::Mat srcimg, int lineori, vector<cv::Point>& contours, int radius, float percent){
    if(srcimg.empty()||srcimg.cols<radius*2+2||srcimg.rows<radius*2+2)
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    if(lineori==1){
        for(int i=radius+1;i<srcimg.cols-radius-1;++i){
            cv::Point seed(i,srcimg.rows-1);
            cv::Point contourpoint(i,seed.y-1);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y-=1;
                if(contourpoint.y==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(lineori==2){
        for(int i=radius+1;i<srcimg.rows-radius-1;++i){
            cv::Point seed(0,i);
            cv::Point contourpoint(seed.x+1,i);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x+=1;
                if(contourpoint.x>=srcimg.cols)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(lineori==3){
        for(int i=radius+1;i<srcimg.cols-radius-1;++i){
            cv::Point seed(i,0);
            cv::Point contourpoint(i,seed.y+1);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y+=1;
                if(contourpoint.y>=srcimg.rows)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(lineori==4){
        for(int i=radius+1;i<srcimg.rows-radius-1;++i){
            cv::Point seed(srcimg.cols-1,i);
            cv::Point contourpoint(seed.x-1,i);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x-=1;
                if(contourpoint.x==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    return 0;
}

int preprocess::PointDense(cv::Mat srcimg, cv::Point center, int rad, float thresh){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat roiimg;
    if(center.x-rad-1<0||center.y-rad-1<0||center.x+rad+1>=srcimg.cols||center.y+rad+1>=srcimg.rows)
    {return 0;}
    else{
        roiimg=srcimg(cv::Rect(cv::Point(center.x-rad-1,center.y-rad-1),cv::Point(center.x+rad+1,center.y+rad+1)));
    }
    cv::Mat maskimg=roiimg.clone();
    cv::circle(maskimg,cv::Point(maskimg.cols/2,maskimg.rows/2),rad,255,-1);
    int sum=0,num=0;
    for(int i=0;i<maskimg.rows;++i){
        for(int j=0;j<maskimg.cols;++j){
            if(srcimg.at<uchar>(center.y-rad-1+i,center.x-rad-1+j)==255&&maskimg.at<uchar>(i,j)==255){
                num++;
            }
            if(maskimg.at<uchar>(i,j)==255)
            {sum++;}
        }
    }
    float percent=(float)num/(float)sum;
    if(percent>=thresh){
        return 1;
    }
    else{return 0;}
}

int preprocess::LineFit(vector<cv::Point> contours,cv::Vec2f& linepara){
    /*
     a = (n*C - B*D) / (n*A - B*B)
     b = (A*D - B*C) / (n*A - B*B)
    其中：
     A = sum(Xi * Xi)
     B = sum(Xi)
     C = sum(Xi * Yi)
     D = sum(Yi)    */
    int dx,flag=0;
    float k1;
    dx=abs(contours[0].x-contours[contours.size()-1].x);
    if(dx>0){
        k1=(float)(contours[0].y-contours[contours.size()-1].y)/(contours[0].x-contours[contours.size()-1].x);
    }
    float A = 0.0;
    float B = 0.0;
    float C = 0.0;
    float D = 0.0;
    if(dx==0||fabs(k1)>100){
        for(int i=0;i<contours.size();++i){
            A+=contours[i].y*contours[i].y;
            B+=contours[i].y;
            C+=contours[i].y*contours[i].x;
            D+=contours[i].x;
        }
        flag=1;
    }
    else{
        for(int i=0;i<contours.size();++i){
            A+=contours[i].x*contours[i].x;
            B+=contours[i].x;
            C+=contours[i].x*contours[i].y;
            D+=contours[i].y;
        }
    }
    double a,b,temp=0;
    temp = (contours.size()*A - B*B);
    if(temp)// 判断分母不为0
    {
        a = (contours.size()*C - B*D) / temp;
        b = (A*D - B*C) / temp;
        if(flag==0){
            linepara[0]=a;
            linepara[1]=b;
        }
        else{
            linepara[0]=1/a;
            linepara[1]=-b/a;
        }
        return 0;//y=ax+b
    }
    else
    {
        a = 1;
        b = 0;
        return -1;//ax+by+c=0
    }
}

/*离散的二维卷积运算*/
void preprocess::conv2D(cv::InputArray _src, cv::InputArray _kernel, cv::OutputArray _dst, int ddepth, cv::Point anchor, int borderType)
{
    //卷积核顺时针旋转180
    cv::Mat kernelFlip;
    cv::flip(_kernel, kernelFlip, -1);
    //针对每一个像素,领域对应元素相乘然后相加
    cv::filter2D(_src, _dst, CV_32FC1, _kernel, anchor, 0.0, borderType);
}

/* Krisch 边缘检测算法*/
cv::Mat preprocess::krisch(cv::InputArray src,int borderType)
{
    //存储八个卷积结果
    vector<cv::Mat> eightEdge;
    eightEdge.clear();
    /*第1步：图像矩阵与8 个 卷积核卷积*/
    /*Krisch 的 8 个卷积核均不是可分离的*/
    //图像矩阵与 k1 卷积
    cv::Mat k1 = (cv::Mat_<float>(3, 3) << 5, 5, 5, -3, 0, -3, -3, -3, -3);
    cv::Mat src_k1;
    conv2D(src, k1, src_k1, CV_32FC1);
    cv::convertScaleAbs(src_k1, src_k1);
    eightEdge.push_back(src_k1);
    //图像矩阵与 k2 卷积
    cv::Mat k2 = (cv::Mat_<float>(3, 3) << -3, -3, -3, -3, 0, -3, 5, 5, 5);
    cv::Mat src_k2;
    conv2D(src, k2, src_k2, CV_32FC1);
    cv::convertScaleAbs(src_k2, src_k2);
    eightEdge.push_back(src_k2);
    //图像矩阵与 k3 卷积
    cv::Mat k3 = (cv::Mat_<float>(3, 3) << -3, 5, 5, -3, 0, 5, -3, -3, -3);
    cv::Mat src_k3;
    conv2D(src, k3, src_k3, CV_32FC1);
    cv::convertScaleAbs(src_k3, src_k3);
    eightEdge.push_back(src_k3);
    //图像矩阵与 k4 卷积
    cv::Mat k4 = (cv::Mat_<float>(3, 3) << -3, -3, -3, 5, 0, -3, 5, 5, -3);
    cv::Mat src_k4;
    conv2D(src, k4, src_k4, CV_32FC1);
    cv::convertScaleAbs(src_k4, src_k4);
    eightEdge.push_back(src_k4);
    //图像矩阵与 k5 卷积
    cv::Mat k5 = (cv::Mat_<float>(3, 3) << -3, -3, 5, -3, 0, 5, -3, -3, 5);
    cv::Mat src_k5;
    conv2D(src, k5, src_k5, CV_32FC1);
    cv::convertScaleAbs(src_k5, src_k5);
    eightEdge.push_back(src_k5);
    //图像矩阵与 k6 卷积
    cv::Mat k6 = (cv::Mat_<float>(3, 3) << 5, -3, -3, 5, 0, -3, 5, -3, -3);
    cv::Mat src_k6;
    conv2D(src, k6, src_k6, CV_32FC1);
    cv::convertScaleAbs(src_k6, src_k6);
    eightEdge.push_back(src_k6);
    //图像矩阵与 k7 卷积
    cv::Mat k7 = (cv::Mat_<float>(3, 3) << -3, -3, -3, -3, 0, 5, -3, 5, 5);
    cv::Mat src_k7;
    conv2D(src, k7, src_k7, CV_32FC1);
    cv::convertScaleAbs(src_k7, src_k7);
    eightEdge.push_back(src_k7);
    //图像矩阵与 k8 卷积
    cv::Mat k8 = (cv::Mat_<float>(3, 3) << 5, 5, -3, 5, 0, -3, -3, -3, -3);
    cv::Mat src_k8;
    conv2D(src, k8, src_k8, CV_32FC1);
    cv::convertScaleAbs(src_k8, src_k8);
    eightEdge.push_back(src_k8);
    /*第二步：将求得的八个卷积结果,取对应位置的最大值，作为最后的边缘输出*/
    cv::Mat krischEdge = eightEdge[0].clone();
    for (int i = 0; i < 8; i++)
    {
        cv::max(krischEdge, eightEdge[i], krischEdge);
    }
    return krischEdge;
}

void preprocess::CopyFile(std::string sourcefile, std::string destfile)
{
    string command = "cp ";
    command  += sourcefile;
    command  += " ";
    command  += destfile;//cp /home/file1 /root/file2
    system((char*)command.c_str());//
}

void preprocess::ExpandImage(cv::Mat srcimg, cv::Mat& dstimg){
    if(srcimg.empty())
        return;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::resize(srcimg,srcimg,cv::Size(srcimg.cols/4,srcimg.rows/4));
    cv::Mat image02,image03,image04,image05,image06;
    image02= cv::Mat::zeros(srcimg.rows,srcimg.cols,CV_8UC1);
    cv::hconcat(image02,image02,image03);cv::hconcat(image03,image02,image03);
    cv::hconcat(image02,srcimg,image04);cv::hconcat(image04,image02,image04);
    cv::hconcat(image02,image02,image05);cv::hconcat(image05,image02,image05);
    cv::vconcat(image03,image04,image06);cv::vconcat(image06,image05,image06);
    dstimg=image06.clone();
}

void preprocess::ClassifyImages2(std::string filename){
    cv::String pattern_jpg;
    pattern_jpg = filename;
    vector<cv::String> image_files;
    cv::glob(pattern_jpg, image_files);
    string folderPath = filename+"/";
    for(int i=0;i<3;++i){
        string command;
        command = "mkdir -p " + folderPath +std::to_string(i+11);
        system(command.c_str());
        //rm -rf +dir
        command="rm -rf "+ folderPath + std::to_string(i+11)+"/*";
        system(command.c_str());
    }
    for(int i=0;i<image_files.size();++i){
        if(image_files[i][image_files[i].size()-5]=='1'){
            std::string imgname;
            for(int j=image_files[i].size()-1;j>=0;--j){
                if(image_files[i][j]=='/'){
                    imgname=image_files[i].substr(j,image_files[i].size()-j);
                    //cout<<imgname<<endl;
                    break;
                }
            }
            std::string destination = folderPath + std::to_string(11) + imgname;//目标文件
            CopyFile(image_files[i], destination);
        }
        else if(image_files[i][image_files[i].size()-5]=='2'){
            std::string imgname;
            for(int j=image_files[i].size()-1;j>=0;--j){
                if(image_files[i][j]=='/'){
                    imgname=image_files[i].substr(j,image_files[i].size()-j);
                    //cout<<imgname<<endl;
                    break;
                }
            }
            std::string destination = folderPath + std::to_string(12) + imgname;//目标文件
            CopyFile(image_files[i], destination);
        }
        else if(image_files[i][image_files[i].size()-5]=='3'){
            std::string imgname;
            for(int j=image_files[i].size()-1;j>=0;--j){
                if(image_files[i][j]=='/'){
                    imgname=image_files[i].substr(j,image_files[i].size()-j);
                    //cout<<imgname<<endl;
                    break;
                }
            }
            std::string destination = folderPath + std::to_string(13) + imgname;//目标文件
            CopyFile(image_files[i], destination);
        }
    }
}

int preprocess::JPGtoPNG(std::string filename){
    std::vector<cv::String> image_files;
    std::string filepath=filename+"/*.jpg";
    ReadImagePath(filepath,image_files);
    for(int i=0;i<image_files.size();++i){
        cv::Mat srcimg=cv::imread(image_files[i],0);
        for(int j=image_files[i].size()-1;j>=0;--j){
            if(image_files[i][j]=='.'){
                std::string name=image_files[i].substr(0,j);
                cv::imwrite(name+".png",srcimg);
                continue;
            }
        }
    }
    return 0;
}

int preprocess::ClearDust(std::string filename){
    std::vector<cv::String> image_files;
    std::string filepath=filename+"/*.jpg";
    ReadImagePath(filepath,image_files);
    for(int i=0;i<image_files.size();++i){
        cv::Mat srcimg=cv::imread(image_files[i],0);
        cv::Mat dust=srcimg(cv::Rect(cv::Point(160,200),cv::Point(301,303)));
        dust=250;
        cv::imwrite(image_files[i],srcimg);
    }
    return 0;
}
