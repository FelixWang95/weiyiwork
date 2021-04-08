#include "findoutercontour.h"

FindOuterContour::FindOuterContour()
{
    th1 = CV_PI/180*5, th2=CV_PI/180*85,th3=CV_PI/180*90*(-1);
    NN=2;
    Point2f tm;
    tm.x=0,tm.y=0;
    srcTri[0]=tm,srcTri[1]=tm,srcTri[2]=tm,srcTri[3]=tm;
    dstTri[0]=tm,dstTri[1]=tm,dstTri[2]=tm,dstTri[3]=tm;
}
void FindOuterContour::ScanPoints(cv::Mat imgGray,std::vector<float> &source_x, std::vector<float> &source_y)
{



    for(float th=th1;th<=th2;th+=(CV_PI/180))
        {
            for(int i=0;i<imgGray.cols;i++)
          {
            int j=0;
            j=int(tan(th)*float(i));
            if(j>=(imgGray.rows-1))break;
            if(imgGray.at<uchar>(j,i)>200&&imgGray.at<uchar>(j,i)!=100)
            {
                source_x.push_back(float(i));
                source_y.push_back(float(j));
                imgGray.at<uchar>(j,i)=100;
                break;
            }
          }
        }


    for(float th=th1;th<=th2;th+=(CV_PI/180))
    {
        for(int i=imgGray.cols-1;i>=0;i--)
        {
            int j=0;
            j=int(tan(th)*abs(float(i-(imgGray.cols-1))));
            if(j>=(imgGray.rows-1))break;
            if(imgGray.at<uchar>(j,i)>200&&imgGray.at<uchar>(j,i)!=100)
            {
                source_x.push_back(float(i));
                source_y.push_back(float(j));
                imgGray.at<uchar>(j,i)=100;
                break;
            }
        }
    }

    for(float th=th1;th<=th2;th+=(CV_PI/180))
    {
        for(int i=imgGray.cols-1;i>=0;i--)
        {
            int j=0;
            j=imgGray.rows-int(tan(th)*abs(float(i-(imgGray.cols-1))));
            if(j>=(imgGray.rows-1))break;
            if(imgGray.at<uchar>(j,i)>200&&imgGray.at<uchar>(j,i)!=100)
            {
                source_x.push_back(float(i));
                source_y.push_back(float(j));
                imgGray.at<uchar>(j,i)=100;
                break;
            }
        }
    }


    for(float th=th1;th<=th2;th+=(CV_PI/180))
    {
        for(int i=0;i<imgGray.cols;i++)
        {
            int j=0;
            j=imgGray.rows-1-int(tan(th)*float(i));
            if(j>=(imgGray.rows-1))break;
            if(imgGray.at<uchar>(j,i)>200&&imgGray.at<uchar>(j,i)!=100)
            {
                source_x.push_back(float(i));
                source_y.push_back(float(j));
                imgGray.at<uchar>(j,i)=100;
                break;
            }
        }
    }
}

void FindOuterContour::setNN(int nn)
{
    NN=nn;

}
void FindOuterContour::PeripheralTraversa(cv::Mat srcImg,std::vector<float> &source_x, std::vector<float> &source_y)
{

        cv::Mat imgGray,imgGray01,imgGray02;
        cv::Mat image01,image02,image03,image04,image05,image06;
        medianBlur(srcImg,srcImg,5);
        resize(srcImg, image01, Size(srcImg.cols / NN, srcImg.rows / NN), (0, 0), (0, 0), INTER_LINEAR);
        if(image01.channels()>1)
        cvtColor(image01, imgGray, CV_RGB2GRAY);
        else
        imgGray=image01.clone();
        threshold(imgGray, imgGray, 50, 255, CV_THRESH_BINARY);
        //threshold(imgGray, imgGray, 0, 255, CV_THRESH_OTSU);
        //ScanPoints(imgGray,source_x, source_y);

        image02= cv::Mat::zeros(imgGray.rows,imgGray.cols,CV_8UC1);
        hconcat(image02,image02,image03),hconcat(image03,image02,image03);
        hconcat(image02,imgGray,image04),hconcat(image04,image02,image04);
        hconcat(image02,image02,image05),hconcat(image05,image02,image05);
        vconcat(image03,image04,image06),vconcat(image06,image05,image06);
        imgGray01=image06.clone();

        float th[6]={0};
        th[0]=0,th[0]=CV_PI/180*30,th[1]=CV_PI/180*60,th[2]=CV_PI/180*90,th[3]=CV_PI/180*120,th[4]=CV_PI/180*150,th[5]= CV_PI/180;
        //#pragma omp parallel for
        //for(float th=0;th<CV_PI;th+=(CV_PI/180*30))
        for(int ii=0;ii<6;ii++)
        {
            th3=th[ii];
            imgGray02=cv::Mat::zeros(imgGray01.rows,imgGray01.cols,CV_8UC1);

            int xc=imgGray01.cols/2, yc=imgGray01.rows/2;
            //double Time0 = (double)cvGetTickCount();
//            #pragma omp parallel for
            for(int i=imgGray.cols;i<imgGray.cols*2;i++)
            {
                for(int j=imgGray.rows;j<imgGray.rows*2;j++)
                {
                    if(imgGray01.at<uchar>(j,i)<80)continue;
                    else
                    {
                        int ix=0,jy=0;
                        ix=xc+(i-xc)*cos(th3)-(j-yc)*sin(th3);
                        jy=yc+(j-yc)*cos(th3)+(i-xc)*sin(th3);
                        imgGray02.at<uchar>(jy,ix)=imgGray01.at<uchar>(j,i);
                    }

                }
            }
            //Time0 = (double)cvGetTickCount() - Time0;
            //printf("run time0 = %gms\n", Time0 / (cvGetTickFrequency() * 1000));//ms
            vector<float> source_xt,source_yt;
            //imwrite("tt.png",imgGray02);
            //double Time0 = (double)cvGetTickCount();
            ScanPoints(imgGray02,source_xt, source_yt);
//            Time0 = (double)cvGetTickCount() - Time0;
//            printf("run time0 = %gms\n", Time0 / (cvGetTickFrequency() * 1000));//ms
            for(int i=0;i<int(source_xt.size());i++)
            {
                float tx=0,ty=0;
                tx=xc + source_xt[i]*cos(th3) - xc*cos(th3) + source_yt[i]*sin(th3) - yc*sin(th3);
                ty=yc + source_yt[i]*cos(th3) - yc*cos(th3) - source_xt[i]*sin(th3) + xc*sin(th3);
                source_x.push_back(tx-imgGray.cols),source_y.push_back(ty-imgGray.rows);
            }
        }
}

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
double FindOuterContour::angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

void FindOuterContour::PerspectiveTransformation(const cv::Mat &image,cv::Mat &dstImage)
{
    vector<vector<Point>> squares;
    //vector<Point> dstTri;
//    Point2f srcTri[4];
//    Point2f dstTri[4];

    findSquares(image, squares);

    cv::Mat warp_mat( 3, 3, CV_32FC1 );
    if(squares.size()>0)
    {
        vector<float> dis0,dis1,dis2,dis3;
        for(int i=0;i<4;i++)
        {
            dis0.push_back(sqrt(gsl_pow_2(squares[0][i].x-0)+gsl_pow_2(squares[0][i].y-0)));
            dis1.push_back(sqrt(gsl_pow_2(squares[0][i].x-image.cols)+gsl_pow_2(squares[0][i].y-0)));
            dis2.push_back(sqrt(gsl_pow_2(squares[0][i].x-image.cols)+gsl_pow_2(squares[0][i].y-image.rows)));
            dis3.push_back(sqrt(gsl_pow_2(squares[0][i].x-0)+gsl_pow_2(squares[0][i].y-image.rows)));
        }
        vector<float>::iterator minnest0 = min_element(dis0.begin(), dis0.end());
        int minId0 = distance(dis0.begin(), minnest0);
        vector<float>::iterator minnest1 = min_element(dis1.begin(), dis1.end());
        int minId1 = distance(dis1.begin(), minnest1);
        vector<float>::iterator minnest2 = min_element(dis2.begin(), dis2.end());
        int minId2 = distance(dis2.begin(), minnest2);
        vector<float>::iterator minnest3 = min_element(dis3.begin(), dis3.end());
        int minId3 = distance(dis3.begin(), minnest3);
        srcTri[0]=squares[0][minId0],srcTri[1]=squares[0][minId1],srcTri[2]=squares[0][minId2],srcTri[3]=squares[0][minId3];


        dstTri[0].x =0;
        dstTri[0].y =0;
        dstTri[1].x =(int)(sqrt(gsl_pow_2((srcTri[1].x-srcTri[0].x))+gsl_pow_2((srcTri[1].y-srcTri[0].y))));
        dstTri[1].y =0;
        dstTri[2].x =(int)(sqrt(gsl_pow_2((srcTri[1].x-srcTri[0].x))+gsl_pow_2((srcTri[1].y-srcTri[0].y))));
        dstTri[2].y =(int)(sqrt(gsl_pow_2((srcTri[2].x-srcTri[1].x))+gsl_pow_2((srcTri[2].y-srcTri[1].y))));
        dstTri[3].x =0;
        dstTri[3].y =(int)(sqrt(gsl_pow_2((srcTri[2].x-srcTri[1].x))+gsl_pow_2((srcTri[2].y-srcTri[1].y))));

        dstImage=cv::Mat::zeros(int(dstTri[2].y), int(dstTri[2].x),CV_8UC1);
        warp_mat=getPerspectiveTransform(srcTri,dstTri);
        warpPerspective(image, dstImage, warp_mat, dstImage.size() );
    }
    //imwrite("/home/qzs/DataFile/WyData/0/imag.jpg",dstImage);

}

void FindOuterContour::getsrcTri(vector<Point2f> &srcAns)
{
    for(int i=0;i<4;i++)
    {
        srcAns.push_back(srcTri[i]);
    }
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
void FindOuterContour::findSquares( const cv::Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();

    cv::Mat timg(image);
    medianBlur(image, timg, 9);
    cv::Mat gray0(timg.size(), CV_8U), gray;

    vector<vector<Point> > contours;

    threshold(timg, gray, 0, 255, CV_THRESH_OTSU);
    //Canny(gray0, gray, 5, thresh, 5);
    // find contours and store them all as a list
    findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    vector<Point> approx;

    // test each contour
    for( size_t i = 0; i < contours.size(); i++ )
    {
        // approximate contour with accuracy proportional
        // to the contour perimeter
        approxPolyDP(cv::Mat(contours[i]), approx, arcLength(cv::Mat(contours[i]), true)*0.02, true);

        // square contours should have 4 vertices after approximation
        // relatively large area (to filter out noisy contours)
        // and be convex.
        // Note: absolute value of an area is used because
        // area may be positive or negative - in accordance with the
        // contour orientation
        if( approx.size() == 4 &&
            fabs(contourArea(cv::Mat(approx))) > 13000000 &&
            isContourConvex(cv::Mat(approx)) )
        {
            double maxCosine = 0;

            for( int j = 2; j < 5; j++ )
            {
                // find the maximum cosine of the angle between joint edges
                double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                maxCosine = MAX(maxCosine, cosine);
            }

            // if cosines of all angles are small
            // (all angles are ~90 degree) then write quandrange
            // vertices to resultant sequence
            if( maxCosine < 0.3 )
                squares.push_back(approx);
        }
    }
    if(squares.size()>1)
    {
        vector<Point> approx2=squares[0];
        for(int i=0;i<int(squares.size()-1);i++)
        {
            if(fabs(contourArea(cv::Mat(approx2)))<fabs(contourArea(cv::Mat(squares[i+1]))))
            {
                approx2=squares[i+1];
            }

        }
        squares[0]=approx2;
    }


}

// the function draws all the squares in the image
void FindOuterContour::drawSquares( cv::Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];

        int n = (int)squares[i].size();
        //dont detect the border
        if (p-> x > 3 && p->y > 3)
          polylines(image, &p, &n, 1, true, cv::Scalar(0,255,255), 3);
        //imwrite("/home/qzs/DataFile/WyData/0/imag.jpg",image);
    }
//    imwrite("/home/qzs/DataFile/WyData/0/imag.jpg",image);
//    imshow("imgt", image);
//    int c = waitKey();
//    if( (char)c == 27 )
//        return ;
}

vector<cv::Point> FindOuterContour::SortContour(vector<cv::Point> contour){
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

vector<cv::Point> FindOuterContour::SortContourCenter(vector<cv::Point> contour, cv::Point center){
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

float FindOuterContour::GetAngle(cv::Point A, cv::Point B, cv::Point C, cv::Point D){
    float ABx=B.x-A.x;
    float ABy=B.y-A.y;
    float CDx=D.x-C.x;
    float CDy=D.y-C.y;
    if((ABx==0&&ABy==0)||(CDx==0&&CDy==0))
        return 0;
    float cosa=(ABx*CDx+ABy*CDy)/(sqrt(ABx*ABx+ABy*ABy)*sqrt(CDx*CDx+CDy*CDy));
    if(cosa>=1)
        cosa=0.999999;
    float angle=acos(cosa)*180/PI;
    if(isnan(angle))
        cout<<cosa<<endl;
    return angle;
}
