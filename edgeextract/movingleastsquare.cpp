 #include "movingleastsquare.h"

MovingLeastSquare::MovingLeastSquare()
{
    max_x=0;
    min_x=0;
}

//将m行N列数组转换为m行N列的矩阵
_Matrix Trans_Matrix(float p[][N],int m)
{
    _Matrix C(m,N);
    C.init_matrix();
    for(int i=0;i<C.m;i++)
        for(int j=0;j<C.n;j++)
        {
            C.write(i,j,p[i][j]);
        }
    return C;
}
//将1行N列数组转换为1行N列的矩阵
_Matrix Trans_Matrix_One(float p[],int n)
{
    _Matrix C(1,n);
    C.init_matrix();
    for(int i=0;i<C.m;i++)
        for(int j=0;j<C.n;j++)
        {
            C.write(i,j,p[j]);
        }
    return C;
}
//计算A矩阵个元素的函数
void MovingLeastSquare::f(float w[],float x[],float y[],float sumf[][N],float p[][N])
{
    for(int i=0;i<N;i++)
        for(int j=0;j<M;j++)
        {
            sumf[i][0] = sumf[i][0]+w[j]*1*p[j][i];
            sumf[i][1] = sumf[i][1] + w[j] * x[j] * p[j][i];
            sumf[i][2] = sumf[i][2] + w[j] * y[j]* p[j][i];
            sumf[i][3] = sumf[i][3] + w[j] * x[j] * x[j]* p[j][i];
            sumf[i][4] = sumf[i][4] + w[j] * x[j] * y[j]* p[j][i];
            sumf[i][5] = sumf[i][5] + w[j] * y[j] * y[j]* p[j][i];
        }
}

//移动最小二乘法的具体计算过程，参照论文“基于移动最小二乘法的曲线曲面拟合”，AB矩阵参照论文“移动最小二乘法的研究”
int MovingLeastSquare::MLS_Calc(int x_val,int y_val,float x[],float y[],float z[])
{
    _Matrix_Calc m_c;//定义Matrix_Calc对象
    int max_delta=max_x-min_x;//区域半径
    float p[M][N]={0};
    float sumf[N][N]={0};
    float w[M]={0};
    for(int j=0;j<M;j++)//求w
    {
        float s=fabs((x[j]-x_val))/max_delta;
        if(s<=0.5)
            w[j]=2/3.0-4*s*s+4*s*s*s;
        else
        {
            if(s<=1)
                w[j]=4/3.0-4*s+4*s*s-4*s*s*s/3.0;
            else
                w[j]=0;
        }
        p[j][0]=1;//每个采样点计算基函数
        p[j][1]=x[j];
        p[j][2]=y[j];
        p[j][3]=x[j]*x[j];
        p[j][4]=x[j]*y[j];
        p[j][5]=y[j]*y[j];
    }
    f(w,x,y,sumf,p);//计算得出A矩阵

    float p1[N];
    _Matrix A=Trans_Matrix(sumf,N);
    _Matrix A_c(A.m,A.n);
    A_c.init_matrix();
    m_c.copy(&A,&A_c);
    _Matrix A_1(A_c.m,A_c.n);
    A_1.init_matrix();
    m_c.inverse(&A_c,&A_1);//求A_c矩阵的逆A_1

    _Matrix B(N,M);//求矩阵B，N行M列
    B.init_matrix();
    for(int j=0;j<M;j++)//求得B矩阵的每列
    {
        p1[0]=1*w[j];
        p1[1]=x[j]*w[j];
        p1[2]=y[j]*w[j];
        p1[3]=x[j]*x[j]*w[j];
        p1[4]=x[j]*y[j]*w[j];
        p1[5]=y[j]*y[j]*w[j];
        _Matrix P=Trans_Matrix_One(p1,N);//数组P1转成1行N列的P矩阵
        if(j==0)//第一列直接赋值
        {
            for(int i=0;i<N;i++)
                B.write(i,0,p1[i]);
        }
        else
        {
            _Matrix P_t(P.n,P.m);
            P_t.init_matrix();
            m_c.transpos(&P,&P_t);//矩阵转置，P转为N行1列矩阵
            m_c.addcols(&B,&P_t);//矩阵B列附加，形成N行M列矩阵
            P_t.free_matrix();
        }
        P.free_matrix();
    }

    float D[N]={1,x_val,y_val,x_val*x_val,x_val*y_val,y_val*y_val};
    _Matrix D1=Trans_Matrix_One(D,N);//转成1行N列矩阵

    _Matrix D_A1_mul(1,N);//定义矩阵并初始化相乘的结果矩阵，1行N列
    D_A1_mul.init_matrix();
    if(m_c.multiply(&D1,&A_1,&D_A1_mul)==-1)
        cout<<"矩阵有误1！";//1行N列矩阵乘以N行N列矩阵得到结果为1行N列

    _Matrix D_A1_B_mul(1,M);//定义矩阵并初始化相乘的结果矩阵，1行M列
    D_A1_B_mul.init_matrix();
    if(m_c.multiply(&D_A1_mul,&B,&D_A1_B_mul)==-1)
        cout<<"矩阵有误2";//1行N列矩阵乘以N行M列矩阵得到记过矩阵为1行M列

    _Matrix z1=Trans_Matrix_One(z,M);//将数组z转换成1行M列矩阵
    _Matrix z1_t(z1.n,z1.m);
    z1_t.init_matrix();
    m_c.transpos(&z1,&z1_t);//转置得到M行1列矩阵
    _Matrix Z(1,1);//得到矩阵结果，1行1列
    Z.init_matrix();
    if(m_c.multiply(&D_A1_B_mul,&z1_t,&Z)==-1)
        cout<<"矩阵有误3！";//1行M列矩阵乘以M行1列矩阵得到1行1列矩阵，即值Z

    float z_val=Z.read(0,0);
    if(z_val>255)
        z_val=255;
    if(z_val<0)
        z_val=0;

    A.free_matrix();
    A_1.free_matrix();
    A_c.free_matrix();
    B.free_matrix();

    D1.free_matrix();
    D_A1_mul.free_matrix();
    D_A1_B_mul.free_matrix();
    z1.free_matrix();
    z1_t.free_matrix();
    Z.free_matrix();

    return (int)z_val;
}
//对图像进行取点采样，思路是：
/*对矩阵画对角线，中心点作为图像新坐标，在对角线取3/4的点，共四个，在对半线去1/2的点，共四个，如果八个点像素为0，则缩小为4/5，直到找到像素不为0的点*/
void MovingLeastSquare::x_y(float x[],float y[],float z0[],float z1[],float z2[],Mat im)
{
    float cx = im.cols/ 2;float cy = im.rows/ 2;
    float z[M]={0};
    x[0] = (0 - cx) * 0.7 + cx;y[0] = -(-0 + cy) * 0.7 + cy;
    x[1] = (cx * 2 - cx) * 0.7 + cx;y[1] = -(-0 + cy) * 0.7 + cy;
    x[2] = (0 - cx) * 0.7 + cx;y[2] = -(-cy * 2 + cy) * 0.7 + cy;
    x[3] = (cx * 2 - cx) * 0.7 + cx;y[3] = -(-cy * 2 + cy) * 0.7 + cy;
    x[4] = (cx - cx) * 0.5 + cx; y[4] = -(-0 + cy) * 0.5 + cy;
    x[5] = (0 - cx) * 0.5 + cx;y[5] = -(-cy + cy) * 0.5 + cy;
    x[6] = (cx - cx) * 0.5 + cx;y[6] = -(-cy * 2 + cy) * 0.5 + cy;
    x[7] = (cx * 2 - cx) * 0.5 + cx;y[7] = -(-cy + cy) * 0.5 + cy;
    for(int i=0;i<M;i++)
    {
        z0[i]=im.at<Vec3b>(y[i],x[i])[0];//B
        z1[i]=im.at<Vec3b>(y[i],x[i])[1];//G
        z2[i]=im.at<Vec3b>(y[i],x[i])[2];//R
        while(z0[i] == 255 && z1[i] == 255 && z2[i] == 255)
        {
            x[i] = (x[i] - cx) * 0.8 + cx;
            y[i] = -(-y[i] + cy) * 0.8 + cy;
            z[i] = im.at<uchar>((int)y[i],(int)x[i]);
            if (x[i] > cx-1)
                break;
        }
        z0[i]=im.at<Vec3b>(y[i],x[i])[0];//B
        z1[i]=im.at<Vec3b>(y[i],x[i])[1];//G
        z2[i]=im.at<Vec3b>(y[i],x[i])[2];//R
        float t = x[i];//xy交换
        x[i] = y[i];
        y[i] = t;

    }
}

void MovingLeastSquare::MLS(){
    string path="/home/adt/edgeextract/aaa.jpg";
    cv::Mat image=imread(path);
    imshow("image",image);
    cv::GaussianBlur(image,image,cv::Size(3,3),0,0);//高斯滤波去噪
    cv::Mat image_new=Mat::zeros(image.size(),image.type());
    float x0[M]={0,0,0,0,0,0,0,0};
    float y0[M]={0,0,0,0,0,0,0,0};
    float z0[M]={0,0,0,0,0,0,0,0};
    float z1[M]={0,0,0,0,0,0,0,0};
    float z2[M]={0,0,0,0,0,0,0,0};
    x_y(x0,y0,z0,z1,z2,image);//图像采样取点
    max_x=image_new.rows;
    min_x=0;
    int max_y=image_new.cols;
    int min_y=0;
    //计算移动最小二乘法并绘制成图像
    int count=0;
    int j_min=0,j_max;
    for(int Y=min_y;Y<max_y;Y++)
        for(int X=min_x;X<max_x;X++)
        {
            int Z0 = MLS_Calc(X, Y, x0, y0, z0);
            int Z1 = MLS_Calc(X, Y, x0, y0, z1);
            int Z2 = MLS_Calc(X, Y, x0, y0, z2);
            image_new.at<Vec3b>(X,Y)[0]=Z0;
            image_new.at<Vec3b>(X,Y)[1]=Z1;
            image_new.at<Vec3b>(X,Y)[2]=Z2;
        }
    path="/home/adt/edgeextract/mlsresult.png";
    cv::imwrite(path.c_str(),image_new);//保存拟合的图像
    imshow("image_new",image_new);
    waitKey(0);
}
