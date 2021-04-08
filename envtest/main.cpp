#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;

int CalcHistogram(cv::Mat srcimg,int thresh,int bin,cv::Mat& hist);//
int PCADecreaseDim(cv::Mat iuputMat,cv::Mat& outputMat,float percentage);
int KmeansClassify(cv::Mat inputMat,std::vector<cv::String> image_files,const int K,cv::Mat& Classes);
void CopyFile(std::string sourcefile, std::string destfile);
int OutputMat(std::string filename,cv::Mat outputMat);//write txt from Mat
int DFTFeature(cv::Mat srcimg,cv::Mat& realpart);
int ReadTxt(std::string txtname, cv::Mat& txtdata);//read txt to Mat
int ClassifyImages(std::vector<cv::String> image_files,cv::Mat labels);
int ReadXml(std::string xmlname,cv::Mat& xmldata);//read xml to Mat

int ReadImgGenerateData(){
    int rtn=0;
    cv::String pattern_jpg;
    std::vector<cv::String> image_files;
    pattern_jpg = "/mnt/hgfs/linuxsharefiles/darksurface/darksamples/*.jpg";
    cv::glob(pattern_jpg, image_files);
    cv::Mat dataMat;
    int dim=256;
    for(int num=0;num<image_files.size();++num)//read all images in the folder
    {
        cout << image_files[num] << endl;
        cv::Mat srcimg=cv::imread(image_files[num],0);
        cv::Mat hist;
        rtn=CalcHistogram(srcimg,20,dim,hist);
        //cv::Mat realpart;
        //rtn=DFTFeature(srcimg,realpart);
        if(num==0){
            dataMat=hist;
            //dataMat=realpart;
        }
        else{
            cv::vconcat(dataMat,hist,dataMat);
            //cv::vconcat(dataMat,realpart,dataMat);
        }
    }
    cv::Mat labelspy;
    ReadTxt("labels.txt",labelspy);
    ClassifyImages(image_files,labelspy);
    cout<<dataMat.rows<<"  "<<dataMat.cols<<endl;
//    cv::Mat dataMatDD;
//    float per=0.95;
//    cout<<dim<<"  "<<per<<endl;
//    rtn=PCADecreaseDim(dataMat,dataMatDD,per);
//    cout<<dataMatDD.rows<<"  "<<dataMatDD.cols<<endl;
//    //cout<<dataMatDD<<endl;
//    //cv::Mat Labels;
//    //rtn=KmeansClassify(dataMatDD,image_files,7,Labels);
//    //write xml
//    cv::FileStorage fs("feaMat-heise.xml",cv::FileStorage::WRITE);
//    fs<<"voc"<<dataMatDD;
//    fs.release();
//    //write txt
      rtn=OutputMat("feaMat256.txt",dataMat);
//    rtn=OutputMat("feaMat1.txt",dataMatDD);
//    //rtn=OutputMat("labelMat.txt",Labels);
    //cv::waitKey(0);
    return 0;
}
int OutputXMLreduceMat(){
    std::string xmlfile="/mnt/hgfs/linuxsharefiles/classify1/feaMat-HeiSe.xml";
    cv::Mat xmldata;
    ReadXml(xmlfile,xmldata);
    cv::Mat newxmldata = xmldata.rowRange(0,2670);
    cout<<newxmldata.size()<<endl;
    cv::Mat resMat;
    for(int i=0;i<newxmldata.rows;i+=30){
        cv::Mat subMat=newxmldata.rowRange(i,i+30);
        cv::Mat rowMat;
        cv::reduce(subMat, rowMat, 0, CV_REDUCE_SUM);
        if(i==0){
            resMat=rowMat;
        }
        else{
            cv::vconcat(resMat,rowMat,resMat);
        }
    }
    OutputMat("feaMat89.txt",resMat/30);
}

int main()
{
    int rtn=0;
    cv::String pattern_jpg;
    std::vector<cv::String> image_files;
    std::vector<cv::String> image_files_1;
    pattern_jpg = "/mnt/hgfs/linuxsharefiles/newimages/totalimage/*.jpg";
    cv::glob(pattern_jpg, image_files);
    cv::Mat labelspy;
    ReadTxt("labels.txt",labelspy);
    cout<<labelspy.size()<<endl;
    cv::Mat dataMat;
    int dim=256;
    for(int num=0;num<image_files.size();++num)//read all images in the folder
    {
        if(image_files[num][53]=='1'&&image_files[num][image_files[num].size()-5]=='2'){
            cout << image_files[num] << endl;
            CopyFile(image_files[num],"/mnt/hgfs/linuxsharefiles/newimages/totalimage/1-12/");
//            image_files_1.push_back(image_files[num]);
//            cv::Mat srcimg=cv::imread(image_files[num],0);
//            cv::Mat hist;
//            rtn=CalcHistogram(srcimg,20,dim,hist);
//            //cv::Mat realpart;
//            //rtn=DFTFeature(srcimg,realpart);
//            if(image_files_1.size()==1){
//                dataMat=hist;
//                //dataMat=realpart;
//            }
//            else{
//                cv::vconcat(dataMat,hist,dataMat);
//                //cv::vconcat(dataMat,realpart,dataMat);
//            }
        }
    }
    //cout<<dataMat.size()<<endl;
    //cv::Mat output;
    //PCADecreaseDim(dataMat,output, 0.9);
    //cout<<output<<endl;
    //rtn=OutputMat("feaMat256.txt",dataMat);
    //ClassifyImages(image_files_1,labelspy);
    //cout<<dataMat.rows<<"  "<<dataMat.cols<<endl;
    cv::waitKey(0);
    return 0;
}

int CalcHistogram(cv::Mat srcimg, int thresh,int bin,cv::Mat& hist){
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

int PCADecreaseDim(cv::Mat iuputMat,cv::Mat& outputMat, float percentage){
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
    return 0;
}

int KmeansClassify(cv::Mat inputMat,std::vector<cv::String> image_files,const int K,cv::Mat& Classes){
    if(inputMat.empty()){
        return -1;
    }
    const int attemps{ 100 };
    const cv::TermCriteria term_criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.01);
    cv::Mat centers_;
    double value = cv::kmeans(inputMat, K, Classes, term_criteria, attemps, cv::KMEANS_RANDOM_CENTERS, centers_);
    fprintf(stdout, "K = %d, attemps = %d, iter count = %d, compactness measure =  %f\n",K, attemps, term_criteria.maxCount, value);
    //cout<<Classes<<endl;
    cout<<Classes.size()<<endl;
    string folderPath = "/mnt/hgfs/linuxsharefiles/darksurface/kmeansClasses/";
    for(int i=0;i<K;++i){
        string command;
        command = "mkdir -p " + folderPath + std::to_string(i);
        system(command.c_str());
        //rm -rf +dir
        command="rm -rf "+ folderPath + std::to_string(i)+"/*";
        system(command.c_str());
    }
    for(int i=0;i<image_files.size();++i){
        std::string source = image_files[i];//源文件
        std::string imgname;
        for(int j=image_files[i].size()-1;j>=0;--j){
            if(image_files[i][j]=='/'){
                imgname=image_files[i].substr(j,image_files[i].size()-j);
                //cout<<imgname<<endl;
                break;
            }
        }
        std::string destination = folderPath + std::to_string(Classes.at<int>(i,0))+imgname;//目标文件
        CopyFile(source, destination);
    }
    return 0;
}

void CopyFile(std::string sourcefile, std::string destfile)
{
    string command = "cp ";
    command  += sourcefile;
    command  += " ";
    command  += destfile;//cp /home/file1 /root/file2
    system((char*)command.c_str());//
}

int OutputMat(std::string filename,cv::Mat outputMat){
    ofstream out;
    out.open(filename);
    if(!out.is_open()){
         cout << "open dst File  Error opening file" << endl;
         return -1;
     }
    if(outputMat.type()==5||outputMat.type()==6){
        for(int i=0;i<outputMat.rows;++i){
            for(int j=0;j<outputMat.cols;++j){
                out<<outputMat.at<float>(i,j)<<" ";
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

int DFTFeature(cv::Mat srcimg,cv::Mat& realpart){
    if(srcimg.empty())
    {
        return -1;
    }
    cv::resize(srcimg,srcimg,cv::Size(srcimg.cols/4,srcimg.rows/4));
    srcimg.convertTo(srcimg, CV_32F);//转换为32位浮点型
    //为了将傅里叶谱低频部分显示在频谱图的中心
    for (int x = 0; x != srcimg.rows; ++x) {
        for (int y = 0; y != srcimg.cols; ++y) {
            srcimg.ptr<float>(x)[y] *= (float)pow(-1.0, x + y);
        }
    }
    int m = cv::getOptimalDFTSize(srcimg.rows); //2,3,5的倍数有更高效率的傅里叶变换
    int n = cv::getOptimalDFTSize(srcimg.cols);
    cv::Mat padded;
    //把灰度图像放在左上角,在右边和下边扩展图像,扩展部分填充为0;
    cv::copyMakeBorder(srcimg, padded, 0, m - srcimg.rows,0, n - srcimg.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    //cout<<padded.size()<<endl;
    //这里是获取了两个Mat,一个用于存放dft变换的实部，一个用于存放虚部,初始的时候,实部就是图像本身,虚部全为零
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(),CV_32F)};
    cv::Mat complexImg;
    //将几个单通道的mat融合成一个多通道的mat,这里融合的complexImg既有实部又有虚部
    cv::merge(planes,2,complexImg);
    //对上边合成的mat进行傅里叶变换,支持原地操作,傅里叶变换结果为复数.通道1存的是实部,通道二存的是虚部
    cv::dft(complexImg,complexImg);
    //把变换后的结果分割到两个mat,一个实部,一个虚部,方便后续操作
    cv::split(complexImg,planes);
    cv::Mat real=planes[0].clone();
    cv::normalize(real, real, 1.0, 0.0, cv::NORM_INF);
    realpart=real.reshape(1,1);
    cv::magnitude(planes[0],planes[1],planes[0]);
    cv::Mat mag = planes[0];
    mag += cv::Scalar::all(1);
    cv::log(mag, mag);
    //crop the spectrum, if it has an odd number of rows or columns
    //修剪频谱,如果图像的行或者列是奇数的话,那其频谱是不对称的,因此要修剪
    mag = mag(cv::Rect(0, 0, mag.cols & -2, mag.rows & -2));
    //这一步的目的仍然是为了显示,但是幅度值仍然超过可显示范围[0,1],我们使用 normalize() 函数将幅度归一化到可显示范围。
    //cv::Mat _magI = mag.clone();
    //cv::normalize(_magI, _magI, 255.0, 0.0, cv::NORM_INF);
    //_magI.convertTo(_magI, CV_8U);
    //cv::imwrite("dft.png", _magI);
    //cout<<realpart.size()<<endl;
    return 0;
}

int ReadTxt(std::string txtname, cv::Mat& txtdata){
    fstream file1;
    int n=0;
    string tmp;
    int colnum;
    file1.open(txtname);
    if(file1.fail())//文件打开失败:返回0
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
            cout<<txtdata.at<int>(i, j)<<endl;
        }
    }
    file2.close();
    return 0;
}

int ClassifyImages(std::vector<cv::String> image_files,cv::Mat labels){
    if(labels.empty()){
        return -1;
    }
    double minv = 0.0, maxv = 0.0;
    cv::minMaxIdx(labels,&minv,&maxv);
    string folderPath = "/mnt/hgfs/linuxsharefiles/darksurface/kmeansClasses"+std::to_string((int)maxv+1)+"/";
    for(int i=0;i<=maxv;++i){
        string command;
        command = "mkdir -p " + folderPath +std::to_string(i);
        system(command.c_str());
        //rm -rf +dir
        command="rm -rf "+ folderPath + std::to_string(i)+"/*";
        system(command.c_str());
    }
    for(int i=0;i<image_files.size();++i){
        std::string source = image_files[i];//源文件
        std::string imgname;
        for(int j=image_files[i].size()-1;j>=0;--j){
            if(image_files[i][j]=='/'){
                imgname=image_files[i].substr(j,image_files[i].size()-j);
                //cout<<imgname<<endl;
                break;
            }
        }
        std::string destination = folderPath + std::to_string(labels.at<uchar>(i,0))+imgname;//目标文件
        CopyFile(source, destination);
    }
    return 0;
}

int ReadXml(std::string xmlname,cv::Mat& xmldata){
    cv::FileStorage fs;    //OpenCV 读XML文件流
    std::string filename = xmlname;//    待读取.XML文件名
    fs.open(filename,cv::FileStorage::READ);    //打开指定.xml文件
    if(!fs.isOpened()){
        std::cerr << "Error: cannot open .xml file";
        return -1;
    }
    fs["voc"] >> xmldata;    //数据从文件导入至变量
    fs.release();
    return 0;
}
