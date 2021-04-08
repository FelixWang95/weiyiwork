#include "preprocess.h"

preprocess::preprocess()
{

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
                cout<<imgname<<endl;
                break;
            }
        }
        std::string destination = folderPath + std::to_string(labels.at<int>(i,0))+imgname;//目标文件
        CopyFile(source, destination);
    }
    return 0;
}

void preprocess::CopyFile(std::string sourcefile, std::string destfile)
{
    string command = "cp ";
    command  += sourcefile;
    command  += " ";
    command  += destfile;//cp /home/file1 /root/file2
    system((char*)command.c_str());//
}

//计算数据偏差
float preprocess::getoptimaloffset(vector<float> datavec0, vector<float> datavec2, float pthreshold, float nthreshold){
    float th=nthreshold,step=0.001;
    vector<float> distvec;
    for(float k=th;k<pthreshold;k=k+step){
        vector<float> datavec1;
        for(int i=0;i<datavec0.size();++i){
            datavec1.push_back(datavec0[i]+k);
        }
        float sum=0;
        for(int i=0;i<datavec1.size();++i){
            sum+=fabs(datavec1[i]-datavec2[i]);
        }
        float b=sum/datavec1.size();
        float sumsd=0;
        for(int i=0;i<datavec1.size();++i){
            sumsd+=(fabs(datavec1[i]-datavec2[i])-b)*(fabs(datavec1[i]-datavec2[i])-b);
        }
        float c=sumsd/datavec1.size();
        distvec.push_back(sqrt(b*b+c*c));
    }
    auto smallest = std::min_element(std::begin(distvec), std::end(distvec));
    float r=th+std::distance(std::begin(distvec), smallest)*step;
    cout<<r<<endl;
    return r;
}

int preprocess::ReadCSVfile(std::string filename, vector<vector<std::string>>& data){
    // 读文件
    ifstream inFile(filename, ios::in);
    string lineStr;
    data.clear();
    while (getline(inFile, lineStr))
    {
        // 打印整行字符串
        //cout << lineStr << endl;
        // 存成二维表结构
        stringstream ss(lineStr);
        string str;
        vector<string> lineArray;
        // 按照逗号分隔
        while (getline(ss, str, ','))
            lineArray.push_back(str);
        data.push_back(lineArray);
    }
    return 0;
}

int preprocess::GetColdata(vector<vector<std::string>> data, vector<float>& coldata, int colindex,int dataindex, std::string neasureID){
    coldata.clear();
    for(int i=0;i<data.size();++i){
        if(data[i][colindex]==neasureID){
            float b = atof(data[i][dataindex].c_str());
            coldata.push_back(b);
        }
    }
    return 0;
}

int preprocess::ReadTxttoVec(std::string txtname, vector<vector<float>>& data){
    ifstream infile;
    infile.open(txtname);   //将文件流对象与文件连接起来
    assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行
    string tmp;
    data.clear();
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
    return 0;
}

int preprocess::GetColstddata(vector<vector<float>> data, vector<float>& coldata,int dataindex){
    coldata.clear();
    for(int i=0;i<data.size();++i){
        coldata.push_back(data[i][dataindex]);
    }
    return 0;
}

int preprocess::ReadTxttoString(std::string txtname, vector<string>& data){
    ifstream infile;
    infile.open(txtname);   //将文件流对象与文件连接起来
    assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行
    string tmp;
    data.clear();
    while(getline(infile,tmp,'\n'))
    {
        data.push_back(tmp);
    }
    infile.close();
    return 0;
}
