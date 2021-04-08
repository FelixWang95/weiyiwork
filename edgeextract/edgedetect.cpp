#include "edgedetect.h"
#include "rectextract.h"

edgedetect::edgedetect()
{

}
int edgedetect::FindDatumPoint(vector<cv::Point2f> xpts, vector<cv::Point2f> ypts, vector<cv::Point2f>& datumpts){
    datumpts.resize(3);
    vector<float> toppara;
    LineFitLeastSquaresf(xpts,toppara);
    if(toppara[0]<-0.000001||toppara[0]>0.000001){
        float yres=-10000;
        for(int i=0;i<xpts.size();++i){
            float y=xpts[i].y-(xpts[i].x*toppara[0]+toppara[1]);
            if(yres<y){
                yres=y;
                datumpts[1]=xpts[i];
            }
        }
    }
    else{
        float yres=-10000;
        for(int i=0;i<xpts.size();++i){
            if(yres<xpts[i].y){
                yres=xpts[i].y;
                datumpts[1]=xpts[i];
            }
        }
    }
    for(int i=0;i<ypts.size();++i){
        cv::Point2f temp=ypts[i];
        ypts[i].x=temp.y;
        ypts[i].y=temp.x;
    }
    vector<float> leftpara;
    LineFitLeastSquaresf(ypts,leftpara);
    if(leftpara[0]<-0.000001||leftpara[0]>0.000001){
        float yres=-10000;
        for(int i=0;i<ypts.size();++i){
            float y=ypts[i].y-(ypts[i].x*leftpara[0]+leftpara[1]);
            if(yres<y){
                yres=y;
                datumpts[2]=cv::Point2f(ypts[i].y,ypts[i].x);
            }
        }
    }
    else{
        float yres=-10000;
        for(int i=0;i<ypts.size();++i){
            if(yres<ypts[i].y){
                yres=ypts[i].y;
                datumpts[2]=cv::Point2f(ypts[i].y,ypts[i].x);
            }
        }
    }
    cv::Point2f topleft(0,0);
    if(toppara[0]-1/leftpara[0]<-0.000001||toppara[0]-1/leftpara[0]>0.000001){
        topleft.x=(-leftpara[1]/leftpara[0]-toppara[1])/(toppara[0]-1/leftpara[0]);
    }
    else{
        topleft.x=datumpts[2].x;
    }
    topleft.y=topleft.x*toppara[0]+toppara[1];
    datumpts[0]=topleft;
    return 0;
}

int edgedetect::NewKirschEdgeOuter(cv::Mat srcimg, const vector<int> paras, vector<cv::Point2f>& dist){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    vector<cv::Point> seedpoints;
    vector<cv::Point> contours;
    dist.clear();
    if(paras[2]==0){
        for(int i=0;i<paras[9];++i){
            seedpoints.clear();
            contours.clear();
            GetSeedPoints(srcimg,paras[8],paras[4],paras[9+2*i+1],paras[9+2*i+2],paras[3],seedpoints);
            GetEdgePoint(srcimg, seedpoints,paras[5],paras[1],paras[6],(float)paras[7]/100,contours);
            cv::Point2f selectpoint(0,0);
            for(int j=0;j<contours.size();++j){
                selectpoint.x+=contours[j].x;
                selectpoint.y+=contours[j].y;
            }
            selectpoint.x=selectpoint.x/contours.size();
            selectpoint.y=selectpoint.y/contours.size();
            dist.push_back(selectpoint);
            for(int i=0;i<seedpoints.size();++i){
                cv::circle(srcimg,seedpoints[i],1,255,-1);
            }
            for(int i=0;i<contours.size();++i){
                cv::circle(srcimg,contours[i],1,255,-1);
            }
        }
    }
    else if(paras[2]==1){
        for(int i=0;i<paras[9];++i){
            seedpoints.clear();
            contours.clear();
            GetSeedPointsSec(srcimg,paras[8],paras[4],paras[9+2*i+1],paras[9+2*i+2],paras[3],seedpoints);
            GetEdgePoint(srcimg, seedpoints,paras[5],paras[1],paras[6],(float)paras[7]/100,contours);
            cv::Point2f selectpoint(0,0);
            for(int i=0;i<paras[9];++i){
                for(int j=0;j<contours.size();++j){
                    selectpoint.x+=contours[j].x;
                    selectpoint.y+=contours[j].y;
                }
            }
            selectpoint.x=selectpoint.x/contours.size();
            selectpoint.y=selectpoint.y/contours.size();
            dist.push_back(selectpoint);
            for(int i=0;i<seedpoints.size();++i){
                cv::circle(srcimg,seedpoints[i],1,255,-1);
            }
            for(int i=0;i<contours.size();++i){
                cv::circle(srcimg,contours[i],1,255,-1);
            }
        }
    }
    else if(paras[2]==2){
        for(int i=0;i<paras[7];++i){
            contours.clear();
            GetSrcEdgePoints(srcimg,paras[6],paras[4],paras[5],paras[7+2*i+1],paras[7+2*i+2]+1,paras[3],paras[1],contours);
            cv::Point2f selectpoint(0,0);
            for(int j=0;j<contours.size();++j){
                selectpoint.x+=contours[j].x;
                selectpoint.y+=contours[j].y;
            }
            selectpoint.x=selectpoint.x/contours.size();
            selectpoint.y=selectpoint.y/contours.size();
            dist.push_back(selectpoint);
            for(int i=0;i<contours.size();++i){
                cv::circle(srcimg,contours[i],1,160,-1);
            }
        }
        if(paras[0]==8){
            contours.clear();
            for(int imgr=paras[7+2*1+1];imgr<paras[7+2*1+2];++imgr){
                for(int imgc=82;imgc<100;++imgc){
                    if(srcimg.at<uchar>(imgr,imgc)>=paras[3]){
                        contours.push_back(cv::Point(imgc,imgr));
                    }
                }
            }
            cv::Point2f selectpoint(0,0);
            for(int j=0;j<contours.size();++j){
                selectpoint.x+=contours[j].x;
                selectpoint.y+=contours[j].y;
            }
            selectpoint.x=selectpoint.x/contours.size();
            selectpoint.y=selectpoint.y/contours.size();
            dist[1]=selectpoint;
            for(int i=0;i<contours.size();++i){
                cv::circle(srcimg,contours[i],1,255,-1);
            }
        }
    }
//    cv::imshow("dst",srcimg);
    cv::imwrite("se"+std::to_string(paras[0])+".jpg",srcimg);
    return 0;
}

int edgedetect::NewCircleEdge(cv::Mat srcimg, const vector<int> paras, vector<cv::Point>& contours){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat edge;
    if(paras[2]==0){
        edge = krisch(srcimg);
        cv::threshold(edge, edge, paras[1], 255, cv::THRESH_BINARY);
    }
    else if(paras[2]==1){
        cv::threshold(srcimg, edge, paras[1], 255, cv::THRESH_BINARY_INV);
    }
    else if(paras[2]==2){
        cv::threshold(srcimg, edge, paras[1], 255, cv::THRESH_BINARY);
    }
//    cv::imshow("edge",edge);
    edge.at<uchar>(paras[4],paras[3])=0;
    FindEdge(edge, cv::Point(paras[3],paras[4]), contours, paras[5], (float)paras[6]/100, paras[7]);
    contours=ContoursCut(contours, paras[8], paras[9]);
    for(int i=0;i<contours.size();++i){
        cv::circle(srcimg,contours[i],1,160,-1);
    }
    //cout<<contours.size()<<endl;
    //cv::imshow("edgepoint",srcimg);
    cv::imwrite("circle"+std::to_string(paras[0])+".jpg",srcimg);
    return 0;
}

int edgedetect::NewSmallCircle(cv::Mat srcimg, const vector<int> paras,  vector<float>& result){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    vector<cv::Point> contours;
    NewCircleEdge(srcimg, paras,contours);
    result=FitCircle(contours);
    return 0;
}

int edgedetect::ScanLineRange(cv::Mat srcimg,int lineori, vector<cv::Point>& contours,int startline,int startedge, int endedge, int radius,float percent){
    if(srcimg.empty()||srcimg.cols<radius*2+2||srcimg.rows<radius*2+2||startline<0)
        return -1;
    if(startedge<radius+1){
        startedge=radius+1;
    }
    contours.clear();
    if(lineori==SCAN_DIRECTION_UP){
        if(endedge>srcimg.cols-radius-1){
            endedge=srcimg.cols-radius-1;
        }
        for(int i=startedge;i<endedge;++i){
            cv::Point seed(i,startline);
            srcimg.at<uchar>(seed.y,seed.x)=0;
            cv::Point contourpoint(i,seed.y-1);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y-=1;
                if(contourpoint.y==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(lineori==SCAN_DIRECTION_RIGHT){
        if(endedge>srcimg.rows-radius-1){
            endedge=srcimg.rows-radius-1;
        }
        for(int i=startedge;i<endedge;++i){
            cv::Point seed(startline,i);
            srcimg.at<uchar>(seed.y,seed.x)=0;
            cv::Point contourpoint(seed.x+1,i);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x+=1;
                if(contourpoint.x>=srcimg.cols)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(lineori==SCAN_DIRECTION_DOWN){
        if(endedge>srcimg.cols-radius-1){
            endedge=srcimg.cols-radius-1;
        }
        for(int i=startedge;i<endedge;++i){
            cv::Point seed(i,startline);
            srcimg.at<uchar>(seed.y,seed.x)=0;
            cv::Point contourpoint(i,seed.y+1);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y+=1;
                if(contourpoint.y>=srcimg.rows)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(lineori==SCAN_DIRECTION_LEFT){
        if(endedge>srcimg.rows-radius-1){
            endedge=srcimg.rows-radius-1;
        }
        for(int i=startedge;i<endedge;++i){
            cv::Point seed(startline,i);
            srcimg.at<uchar>(seed.y,seed.x)=0;
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

float edgedetect::PointLineDist(cv::Point2f A,vector<float> Lineparas){
    if(Lineparas.size()<2){
        return -1;
    }
    float T=fabs(Lineparas[0]*A.x-A.y+Lineparas[1]);
    float B=sqrt(Lineparas[0]*Lineparas[0]+1);
    return T/B;
}

int edgedetect::NewGetRectEdge(cv::Mat srcimg, const vector<int> paras, vector<float>& dists, vector<vector<cv::Point2f>>& contourres){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat edge;
    if(paras[2]==0){
        edge = krisch(srcimg);
        cv::threshold(edge, edge, paras[1], 255, cv::THRESH_BINARY);
    }
    else if(paras[2]==1){
        cv::threshold(srcimg, edge, paras[1], 255, cv::THRESH_BINARY_INV);
    }
//    cv::imshow("edge",edge);
    dists.clear();
    contourres.clear();
    if(paras[0]==1){
        vector<cv::Point> contours;
        //top line
        ScanLineRange(edge,SCAN_DIRECTION_UP,contours,100,paras[5],paras[6],paras[3],(float)paras[4]/100);
        vector<cv::Point> fitpoints(10);
        for(int i=0;i<fitpoints.size();++i){
            fitpoints[i]=contours[i];
            if(i>4){
                fitpoints[i]=contours[contours.size()-10+i];
            }
        }
        vector<float> topres;
        LineFitLeastSquares(fitpoints, topres);
        float maxdist1=0;
        for(int i=5;i<contours.size()-5;++i){
            float dist=PointLineDist(contours[i],topres);
            if(maxdist1<dist){
                maxdist1=dist;
            }
        }
        dists.push_back(maxdist1);
        //bottom line
        ScanLineRange(edge,SCAN_DIRECTION_DOWN,contours,edge.rows-100,paras[7],paras[8],paras[3],(float)paras[4]/100);
        for(int i=0;i<fitpoints.size();++i){
            fitpoints[i]=contours[i];
            if(i>4){
                fitpoints[i]=contours[contours.size()-10+i];
            }
        }
        vector<float> botres;
        LineFitLeastSquares(fitpoints, botres);
        float maxdist2=0;
        for(int i=5;i<contours.size()-5;++i){
            float dist=PointLineDist(contours[i],botres);
            if(maxdist2<dist){
                maxdist2=dist;
            }
        }
        dists.push_back(maxdist2);
        //left line & right line and parallel
        ScanLineRange(edge,SCAN_DIRECTION_LEFT,contours,100,paras[9],paras[10],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf[i]=temp;
        }
        contourres.push_back(contoursf);
        for(int i=0;i<contours.size();++i){
            cv::Point temp=contours[i];
            contours[i].x=temp.y;
            contours[i].y=temp.x;
        }
        vector<float> leftres;
        LineFitLeastSquares(contours, leftres);
        ScanLineRange(edge,SCAN_DIRECTION_RIGHT,contours,edge.cols-100,paras[9],paras[10],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf1(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf1[i]=temp;
        }
        contourres.push_back(contoursf1);
        for(int i=0;i<contours.size();++i){
            cv::Point temp=contours[i];
            contours[i].x=temp.y;
            contours[i].y=temp.x;
        }
        vector<float> rightres;
        LineFitLeastSquares(contours, rightres);
        vector<cv::Point2f> centerline(paras[10]-paras[9]);
        for(int i=paras[9];i<paras[10];++i){
            centerline[i-paras[9]].y=i;
            centerline[i-paras[9]].x=fabs(i*leftres[0]+leftres[1]+i*rightres[0]+rightres[1])/2.0;
        }
        contourres.push_back(centerline);
        //top lines parallel and top bottom lines
        vector<cv::Point> contours2;
        ScanLineRange(edge,SCAN_DIRECTION_UP,contours,100,paras[11],paras[12],paras[3],(float)paras[4]/100);
        ScanLineRange(edge,SCAN_DIRECTION_UP,contours2,100,paras[13],paras[14],paras[3],(float)paras[4]/100);
        contours.insert(contours.end(),contours2.begin(),contours2.end());
        vector<float> topres1;
        LineFitLeastSquares(contours, topres1);
        vector<cv::Point2f> contoursf2(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf2[i]=temp;
        }
        contourres.push_back(contoursf2);
        ScanLineRange(edge,SCAN_DIRECTION_DOWN,contours,edge.rows-100,paras[15],paras[16],paras[3],(float)paras[4]/100);
        vector<float> botres1;
        LineFitLeastSquares(contours, botres1);
        vector<cv::Point2f> contoursf3(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf3[i]=temp;
        }
        contourres.push_back(contoursf3);
        vector<cv::Point2f> centerline2(paras[13]-paras[12]);
        for(int i=paras[12];i<paras[13];++i){
            centerline2[i-paras[12]].x=i;
            centerline2[i-paras[12]].y=fabs(i*botres1[0]+botres1[1]+i*topres1[0]+topres1[1])/2.0;
        }
        contourres.push_back(centerline2);

    }
    else if(paras[0]==2){
        vector<cv::Point> contours;
        //top line
        ScanLineRange(edge,SCAN_DIRECTION_UP,contours,100,paras[5],paras[6],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf[i]=temp;
        }
        contourres.push_back(contoursf);
        //bottom line
        ScanLineRange(edge,SCAN_DIRECTION_DOWN,contours,edge.rows-200,paras[5],paras[6],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf1(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf1[i]=temp;
        }
        contourres.push_back(contoursf1);
        //left line
        ScanLineRange(edge,SCAN_DIRECTION_LEFT,contours,100,paras[7],paras[8],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf2(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf2[i]=temp;
        }
        contourres.push_back(contoursf2);
        //right line
        ScanLineRange(edge,SCAN_DIRECTION_RIGHT,contours,edge.cols-200,paras[7],paras[8],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf3(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf3[i]=temp;
        }
        contourres.push_back(contoursf3);
    }
    else if(paras[0]==3){
        vector<cv::Point> contours;
        //top line
        ScanLineRange(edge,SCAN_DIRECTION_UP,contours,70,paras[5],paras[6],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf[i]=temp;
        }
        contourres.push_back(contoursf);
        //bottom line
        ScanLineRange(edge,SCAN_DIRECTION_DOWN,contours,695,paras[7],paras[8],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf1(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf1[i]=temp;
        }
        contourres.push_back(contoursf1);
        //left line
        ScanLineRange(edge,SCAN_DIRECTION_LEFT,contours,90,paras[9],paras[10],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf2(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf2[i]=temp;
        }
        contourres.push_back(contoursf2);
        ScanLineRange(edge,SCAN_DIRECTION_LEFT,contours,90,paras[11],paras[12],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf21(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf21[i]=temp;
        }
        //contoursf2.insert(contoursf2.end(),contoursf21.begin(),contoursf21.end());
        contourres.push_back(contoursf21);
        //right line
        ScanLineRange(edge,SCAN_DIRECTION_RIGHT,contours,680,paras[13],paras[14],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf3(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf3[i]=temp;
        }
        contourres.push_back(contoursf3);
        ScanLineRange(edge,SCAN_DIRECTION_RIGHT,contours,680,paras[15],paras[16],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf31(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf31[i]=temp;
        }
        //contoursf3.insert(contoursf3.end(),contoursf31.begin(),contoursf31.end());
        contourres.push_back(contoursf31);
    }
    else if(paras[0]==4){
        vector<cv::Point> contours;
        //top line
        ScanLineRange(edge,SCAN_DIRECTION_UP,contours,260,paras[5],paras[6],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf[i]=temp;
        }
        contourres.push_back(contoursf);
    }
    else if(paras[0]==5){
        vector<cv::Point> contours;
        //left line
        ScanLineRange(edge,SCAN_DIRECTION_LEFT,contours,100,paras[5],paras[6],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf[i]=temp;
        }
        ScanLineRange(edge,SCAN_DIRECTION_LEFT,contours,100,paras[7],paras[8],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf1(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf1[i]=temp;
        }
        edge=255-edge;
        ScanLineRange(edge,SCAN_DIRECTION_LEFT,contours,92,paras[9],paras[10],paras[3],(float)paras[4]/100);
        vector<cv::Point2f> contoursf2(contours.size());
        for(int i=0;i<contours.size();++i){
            cv::Point2f temp;
            temp.x=contours[i].x;
            temp.y=contours[i].y;
            contoursf2[i]=temp;
        }
        contoursf.insert(contoursf.end(),contoursf1.begin(),contoursf1.end());
        contoursf.insert(contoursf.end(),contoursf2.begin(),contoursf2.end());
        contourres.push_back(contoursf);
    }
    for(int i=0;i<contourres.size();++i){
        for(int j=0;j<contourres[i].size();++j){
            cv::circle(edge,contourres[i][j],1,160,-1);
        }
    }
    cv::imwrite("rect"+std::to_string(paras[0])+".jpg",edge);
    return 0;
}

int edgedetect::GetLineContours(cv::Mat srcimg, const vector<int> paras, vector<cv::Point>& contours){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat edge;
    if(paras[2]==0){
        edge = krisch(srcimg);
        cv::threshold(edge, edge, paras[1], 255, cv::THRESH_BINARY);
    }
    else if(paras[2]==1){
        cv::threshold(srcimg, edge, paras[1], 255, cv::THRESH_BINARY_INV);
    }
    else if(paras[2]==2){
        cv::threshold(srcimg, edge, paras[1], 255, cv::THRESH_BINARY);
    }
    cv::imshow("edge",edge);
    contours.clear();
    ScanLineRange(edge,paras[5],contours,paras[6],paras[7],paras[8],paras[3],(float)paras[4]/100);
    for(int i=0;i<contours.size();++i){
        cv::circle(srcimg,contours[i],2,160,-1);
    }
    cv::imwrite("linecontour"+std::to_string(paras[0])+".jpg",srcimg);
    return 0;
}

int edgedetect::LDmeasure(cv::Mat srcimg, const vector<int> paras, vector<cv::Point2f>& respts){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat edge;
    if(paras[1]==1){
        cv::threshold(srcimg, edge, paras[2], 255, cv::THRESH_BINARY_INV);
    }
    else if(paras[1]==2){
        cv::threshold(srcimg, edge, paras[2], 255, cv::THRESH_BINARY);
    }
    //cv::imshow("edge11",edge);
    respts.clear();
    for(int i=0;i<paras[3];++i){
        vector<cv::Point> contours;
        ScanLineRange(edge,paras[9+i*4],contours,paras[6+i*4],paras[7+i*4],paras[8+i*4],paras[4],(float)paras[5]/100);
        cv::Point2f point1(0,0);
        for(int j=0;j<contours.size();++j){
            point1.x+=contours[j].x;
            point1.y+=contours[j].y;
            cv::circle(edge,contours[j],2,160,-1);
        }
        point1.x=point1.x/contours.size();
        point1.y=point1.y/contours.size();
        respts.push_back(point1);
    }
    cv::imshow("edge",edge);
    cv::imwrite("ld"+std::to_string(paras[0])+".jpg",edge);
    return 0;
}

int edgedetect::SmallCircle(cv::Mat srcimg, const vector<int> paras, vector<float>& res, int flag){
    cv::Rect2f centerarea(paras[0],paras[1],paras[2],paras[3]);
    float r=paras[4];
    float rd=paras[5];
    float d=paras[6];
    int randnum=paras[7];
    vector<vector<float>> circleparas(randnum);
    vector<double> sumres(randnum,0);
    cv::RNG rng((unsigned)time(NULL));
    for(int i=0;i<randnum;++i){
        vector<float> para(3);
        para[0]=rng.uniform(centerarea.x, centerarea.x+centerarea.width);
        para[1]=rng.uniform(centerarea.y, centerarea.y+centerarea.height);
        para[2]=rng.uniform(r, r+rd);
        circleparas[i]=para;
        float theta;
        cv::Point2f center(para[0],para[1]);
        for(int j=0;j<180;++j){
            theta=j;
            float sum=0;
            if(theta>=0&&theta<=90){
                cv::Point contourPoint11(center.x+(para[2]+d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint12(center.x+(para[2]+2*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint13(center.x+(para[2]+3*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+3*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint14(center.x+(para[2]-d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint15(center.x+(para[2]-2*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint16(center.x+(para[2]-3*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-3*d)*fabs(sin(theta/180.0*PI)));
                sum=srcimg.at<uchar>(contourPoint11.y,contourPoint11.x)+srcimg.at<uchar>(contourPoint12.y,contourPoint12.x)+srcimg.at<uchar>(contourPoint13.y,contourPoint13.x)-srcimg.at<uchar>(contourPoint14.y,contourPoint14.x)-srcimg.at<uchar>(contourPoint15.y,contourPoint15.x)-srcimg.at<uchar>(contourPoint16.y,contourPoint16.x);
                sum=sum/6;
                cv::Point contourPoint21(center.x-(para[2]+d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint22(center.x-(para[2]+2*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint23(center.x-(para[2]+3*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint24(center.x-(para[2]-d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint25(center.x-(para[2]-2*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint26(center.x-(para[2]-3*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-3*d)*fabs(sin(theta/180.0*PI)));
                float sum2=srcimg.at<uchar>(contourPoint21.y,contourPoint21.x)+srcimg.at<uchar>(contourPoint22.y,contourPoint22.x)+srcimg.at<uchar>(contourPoint23.y,contourPoint23.x)-srcimg.at<uchar>(contourPoint24.y,contourPoint24.x)-srcimg.at<uchar>(contourPoint25.y,contourPoint25.x)-srcimg.at<uchar>(contourPoint26.y,contourPoint26.x);
                sum2=sum2/6;
                sum=sum+sum2;
            }
            else if(theta>90&&theta<180){
                cv::Point contourPoint11(center.x-(para[2]+d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint12(center.x-(para[2]+2*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint13(center.x-(para[2]+3*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+3*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint14(center.x-(para[2]-d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint15(center.x-(para[2]-2*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint16(center.x-(para[2]-3*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-3*d)*fabs(sin(theta/180.0*PI)));
                sum=srcimg.at<uchar>(contourPoint11.y,contourPoint11.x)+srcimg.at<uchar>(contourPoint12.y,contourPoint12.x)+srcimg.at<uchar>(contourPoint13.y,contourPoint13.x)-srcimg.at<uchar>(contourPoint14.y,contourPoint14.x)-srcimg.at<uchar>(contourPoint15.y,contourPoint15.x)-srcimg.at<uchar>(contourPoint16.y,contourPoint16.x);
                sum=sum/6;
                cv::Point contourPoint21(center.x+(para[2]+d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint22(center.x+(para[2]+2*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint23(center.x+(para[2]+3*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint24(center.x+(para[2]-d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint25(center.x+(para[2]-2*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint26(center.x+(para[2]-3*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-3*d)*fabs(sin(theta/180.0*PI)));
                float sum2=srcimg.at<uchar>(contourPoint21.y,contourPoint21.x)+srcimg.at<uchar>(contourPoint22.y,contourPoint22.x)+srcimg.at<uchar>(contourPoint23.y,contourPoint23.x)-srcimg.at<uchar>(contourPoint24.y,contourPoint24.x)-srcimg.at<uchar>(contourPoint25.y,contourPoint25.x)-srcimg.at<uchar>(contourPoint26.y,contourPoint26.x);
                sum2=sum2/6;
                sum=sum+sum2;
            }
            sumres[i]+=sum;
        }
    }
    int pos=0;
    if(flag==1){
        auto maxPosition = max_element(sumres.begin(), sumres.end());
        pos=maxPosition-sumres.begin();
    }
    else if(flag==2){
        auto minPosition = min_element(sumres.begin(), sumres.end());
        pos=minPosition-sumres.begin();
    }
    res=circleparas[pos];
    float theta;
    cv::Point2f center(res[0],res[1]);
    vector<double> sumres2(60,0);
    for(int i=1;i<61;++i){
        vector<float> para={res[0],res[1],res[2]-(float)i*0.1-paras[8]};
        for(int j=0;j<180;++j){
            theta=j;
            float sum=0;
            if(theta>=0&&theta<=90){
                cv::Point contourPoint11(center.x+(para[2]+d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint12(center.x+(para[2]+2*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint13(center.x+(para[2]+3*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+3*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint14(center.x+(para[2]-d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint15(center.x+(para[2]-2*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint16(center.x+(para[2]-3*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-3*d)*fabs(sin(theta/180.0*PI)));
                sum=srcimg.at<uchar>(contourPoint11.y,contourPoint11.x)+srcimg.at<uchar>(contourPoint12.y,contourPoint12.x)+srcimg.at<uchar>(contourPoint13.y,contourPoint13.x)-srcimg.at<uchar>(contourPoint14.y,contourPoint14.x)-srcimg.at<uchar>(contourPoint15.y,contourPoint15.x)-srcimg.at<uchar>(contourPoint16.y,contourPoint16.x);
                sum=sum/6;
                cv::Point contourPoint21(center.x-(para[2]+d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint22(center.x-(para[2]+2*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint23(center.x-(para[2]+3*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint24(center.x-(para[2]-d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint25(center.x-(para[2]-2*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint26(center.x-(para[2]-3*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-3*d)*fabs(sin(theta/180.0*PI)));
                float sum2=srcimg.at<uchar>(contourPoint21.y,contourPoint21.x)+srcimg.at<uchar>(contourPoint22.y,contourPoint22.x)+srcimg.at<uchar>(contourPoint23.y,contourPoint23.x)-srcimg.at<uchar>(contourPoint24.y,contourPoint24.x)-srcimg.at<uchar>(contourPoint25.y,contourPoint25.x)-srcimg.at<uchar>(contourPoint26.y,contourPoint26.x);
                sum2=sum2/6;
                sum=sum+sum2;
            }
            else if(theta>90&&theta<180){
                cv::Point contourPoint11(center.x-(para[2]+d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint12(center.x-(para[2]+2*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint13(center.x-(para[2]+3*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]+3*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint14(center.x-(para[2]-d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint15(center.x-(para[2]-2*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint16(center.x-(para[2]-3*d)*fabs(cos(theta/180.0*PI)),center.y-(para[2]-3*d)*fabs(sin(theta/180.0*PI)));
                sum=srcimg.at<uchar>(contourPoint11.y,contourPoint11.x)+srcimg.at<uchar>(contourPoint12.y,contourPoint12.x)+srcimg.at<uchar>(contourPoint13.y,contourPoint13.x)-srcimg.at<uchar>(contourPoint14.y,contourPoint14.x)-srcimg.at<uchar>(contourPoint15.y,contourPoint15.x)-srcimg.at<uchar>(contourPoint16.y,contourPoint16.x);
                sum=sum/6;
                cv::Point contourPoint21(center.x+(para[2]+d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint22(center.x+(para[2]+2*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint23(center.x+(para[2]+3*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]+2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint24(center.x+(para[2]-d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint25(center.x+(para[2]-2*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-2*d)*fabs(sin(theta/180.0*PI)));
                cv::Point contourPoint26(center.x+(para[2]-3*d)*fabs(cos(theta/180.0*PI)),center.y+(para[2]-3*d)*fabs(sin(theta/180.0*PI)));
                float sum2=srcimg.at<uchar>(contourPoint21.y,contourPoint21.x)+srcimg.at<uchar>(contourPoint22.y,contourPoint22.x)+srcimg.at<uchar>(contourPoint23.y,contourPoint23.x)-srcimg.at<uchar>(contourPoint24.y,contourPoint24.x)-srcimg.at<uchar>(contourPoint25.y,contourPoint25.x)-srcimg.at<uchar>(contourPoint26.y,contourPoint26.x);
                sum2=sum2/6;
                sum=sum+sum2;
            }
            sumres2[i-1]+=sum;
        }
    }
    auto minPosition = min_element(sumres2.begin(), sumres2.end());
    pos=minPosition-sumres2.begin();
    if(sumres2[pos]<2000){
        res[2]=res[2]-30*0.1-0.1-paras[8];
    }
    else{
        res[2]=res[2]-pos*0.1-0.1-paras[8];
    }
    cv::imwrite("small"+std::to_string(paras[0])+".jpg",srcimg);
    return 0;
}

int edgedetect::GetNewDatum575(cv::Mat srcimg, vector<cv::Point2f>& datum){
    cv::Mat top=srcimg(cv::Rect(cv::Point(40,1010),cv::Point(130,1080)));
    cv::Mat top2=srcimg(cv::Rect(cv::Point(4480,980),cv::Point(4900,1080)));
    cv::Mat left=srcimg(cv::Rect(cv::Point(390,370),cv::Point(550,700)));
    cv::threshold(top,top,60,255,CV_THRESH_BINARY);
    cv::threshold(top2,top2,60,255,CV_THRESH_BINARY);
    cv::threshold(left,left,60,255,CV_THRESH_BINARY);
    vector<cv::Point> contourstr;
    ScanLine(top,1,contourstr,top.rows-1,5,0.2);
    for(int i=0;i<contourstr.size();++i){
        contourstr[i].x+=40;
        contourstr[i].y+=1010;
    }
    vector<cv::Point> contourstr1;
    ScanLine(top2,3,contourstr1,1,5,0.2);
    for(int i=0;i<contourstr1.size();++i){
        contourstr1[i].x+=4480;
        contourstr1[i].y+=980;
    }
    vector<float> toppara;
    contourstr.insert(contourstr.end(),contourstr1.begin(),contourstr1.end());
    LineFitLeastSquares(contourstr,toppara);
    vector<cv::Point> contoursl;
    ScanLine(left,4,contoursl,left.cols-1,5,0.2);
    for(int i=0;i<contoursl.size();++i){
        cv::Point temp=contoursl[i];
        contoursl[i].x=temp.y+370;
        contoursl[i].y=temp.x+390;
    }
    vector<float> leftpara;
    LineFitLeastSquares(contoursl,leftpara);
    cv::Point2f topleft(0,0);
    if(toppara[0]-1/leftpara[0]<-0.000001||toppara[0]-1/leftpara[0]>0.000001){
        topleft.x=(-leftpara[1]/leftpara[0]-toppara[1])/(toppara[0]-1/leftpara[0]);
    }
    topleft.y=topleft.x*toppara[0]+toppara[1];
    cv::Point2f toppoint(0,0),leftpoint(0,0);
    toppoint.x=3800;
    toppoint.y=3800*toppara[0]+toppara[1];
    leftpoint.y=2200;
    leftpoint.x=leftpoint.y*leftpara[0]+leftpara[1];
    datum.clear();
    datum.push_back(topleft);
    datum.push_back(toppoint);
    datum.push_back(leftpoint);
    return 0;
}

int edgedetect::GetDatum616(cv::Mat srcimg, const vector<int> paras, vector<cv::Point2f> &datum)
{
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    cv::Mat edge;
    if(paras[1]==1){
        cv::threshold(srcimg, edge, paras[2], 255, cv::THRESH_BINARY);
    }
    else if(paras[1]==2){
        cv::threshold(srcimg, edge, paras[2], 255, cv::THRESH_BINARY_INV);
    }
    vector<cv::Point> vercontours;
    vector<cv::Point> contours;
    cv::imshow("edge",edge);
    ScanLineRange(edge,paras[5],contours,paras[6],paras[7],paras[8],paras[3],(float)paras[4]/100);
    vercontours=contours;
    ScanLineRange(edge,paras[9],contours,paras[10],paras[11],paras[12],paras[3],(float)paras[4]/100);
    vercontours.insert(vercontours.end(),contours.begin(),contours.end());
    vector<float> verpara;
    for(int i=0;i<vercontours.size();++i){
        cv::Point temp=vercontours[i];
        vercontours[i].x=temp.y;
        vercontours[i].y=temp.x;
    }
    LineFitLeastSquares(vercontours,verpara);
    ScanLineRange(edge,paras[13],contours,paras[14],paras[15],paras[16],paras[3],(float)paras[4]/100);
    vector<float> horpara;
    LineFitLeastSquares(contours,horpara);
    cv::Point2f topleft(0,0);
    if(horpara[0]-1/verpara[0]<-0.000001||horpara[0]-1/verpara[0]>0.000001){
        topleft.x=(-verpara[1]/verpara[0]-horpara[1])/(horpara[0]-1/verpara[0]);
    }
    topleft.y=topleft.x*horpara[0]+horpara[1];
    cv::Point2f toppoint(0,0),leftpoint(0,0);
    toppoint.x=srcimg.cols/2;
    toppoint.y=toppoint.x*horpara[0]+horpara[1];
    leftpoint.y=srcimg.rows/2;
    leftpoint.x=leftpoint.y*verpara[0]+verpara[1];
    datum.clear();
    datum.push_back(topleft);
    datum.push_back(toppoint);
    datum.push_back(leftpoint);
    return 0;
}

int edgedetect::GetSideDatum616(vector<cv::Point> contour1,vector<cv::Point> contour2,vector<cv::Point2f>& datum){
    vector<float> verpara;
    for(int i=0;i<contour1.size();++i){
        cv::Point temp=contour1[i];
        contour1[i].x=temp.y;
        contour1[i].y=temp.x;
    }
    LineFitLeastSquares(contour1,verpara);
    vector<float> horpara;
    LineFitLeastSquares(contour2,horpara);
    cv::Point2f topleft(0,0);
    if(fabs(horpara[0])<=0.00001){
        topleft.y=horpara[1];
        topleft.x=topleft.y*verpara[0]+verpara[1];
    }
    else if(fabs(verpara[0])<=0.00001){
        topleft.x=verpara[1];
        topleft.y=topleft.x*horpara[0]+horpara[1];
    }
    else if(horpara[0]-1/verpara[0]<-0.000001||horpara[0]-1/verpara[0]>0.000001){
        topleft.x=(-verpara[1]/verpara[0]-horpara[1])/(horpara[0]-1/verpara[0]);
        topleft.y=topleft.x*horpara[0]+horpara[1];
    }
    cv::Point2f toppoint(0,0),leftpoint(0,0);
    toppoint.x=3000;
    toppoint.y=toppoint.x*horpara[0]+horpara[1];
    leftpoint.y=3000;
    leftpoint.x=leftpoint.y*verpara[0]+verpara[1];
    datum.clear();
    datum.push_back(topleft);
    datum.push_back(toppoint);
    datum.push_back(leftpoint);
    return 0;
}

int edgedetect::GetDatum452(vector<cv::Point> contour1,vector<cv::Point> contour2,vector<cv::Point2f>& datum){
    vector<float> verpara;
    for(int i=0;i<contour1.size();++i){
        cv::Point temp=contour1[i];
        contour1[i].x=temp.y;
        contour1[i].y=temp.x;
    }
    LineFitLeastSquares(contour1,verpara);
    vector<float> horpara;
    LineFitLeastSquares(contour2,horpara);
    cv::Point2f topleft(0,0);
    if(fabs(horpara[0])<=0.00001){
        topleft.y=horpara[1];
        topleft.x=topleft.y*verpara[0]+verpara[1];
    }
    else if(fabs(verpara[0])<=0.00001){
        topleft.x=verpara[1];
        topleft.y=topleft.x*horpara[0]+horpara[1];
    }
    else if(horpara[0]-1/verpara[0]<-0.000001||horpara[0]-1/verpara[0]>0.000001){
        topleft.x=(-verpara[1]/verpara[0]-horpara[1])/(horpara[0]-1/verpara[0]);
        topleft.y=topleft.x*horpara[0]+horpara[1];
    }
    cv::Point2f toppoint(0,0),leftpoint(0,0);
    toppoint.x=3000;
    toppoint.y=toppoint.x*horpara[0]+horpara[1];
    leftpoint.y=3000;
    leftpoint.x=leftpoint.y*verpara[0]+verpara[1];
    datum.clear();
    datum.push_back(topleft);
    datum.push_back(toppoint);
    datum.push_back(leftpoint);
    return 0;
}


void edgedetect::LineFitLeastSquares(vector<cv::Point> contours, vector<float> &vResult)
{
    float A = 0.0;
    float B = 0.0;
    float C = 0.0;
    float D = 0.0;
    float E = 0.0;
    float F = 0.0;

    for (int i = 0; i < contours.size(); i++)
    {
        A += contours[i].x * contours[i].x;
        B += contours[i].x;
        C += contours[i].x * contours[i].y;
        D += contours[i].y;
    }
    float a, b, temp = 0;
    if( temp = (contours.size()*A - B*B) )
    {
        a = (contours.size()*C - B*D) / temp;
        b = (A*D - B*C) / temp;
    }
    else
    {
        a = 1;
        b = 0;
    }
    float Xmean, Ymean;
    Xmean = B / contours.size();
    Ymean = D / contours.size();

    float tempSumXX = 0.0, tempSumYY = 0.0;
    for (int i=0; i<contours.size(); i++)
    {
        tempSumXX += (contours[i].x - Xmean) * (contours[i].x - Xmean);
        tempSumYY += (contours[i].y - Ymean) * (contours[i].y - Ymean);
        E += (contours[i].x - Xmean) * (contours[i].y - Ymean);
    }
    F = sqrt(tempSumXX) * sqrt(tempSumYY);

    float r;
    r = E / F;

    vResult.push_back(a);
    vResult.push_back(b);
    vResult.push_back(r*r);
}

int edgedetect::FindEdge(cv::Mat srcimg, cv::Point seed, vector<cv::Point>& contours, int radius, float percent, int radthresh){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    contours.clear();
    cv::Point center=seed;
    float theta;
    for(int i=0;i<180;++i){
        theta=i;
        float k=tan(theta/180.0*PI);
        float b=center.y-k*center.x;
        if(theta>=0&&theta<=45){
            cv::Point contourPoint1(center.x+1+radthresh*fabs(cos(theta/180.0*PI)),(center.x+1+radthresh*fabs(cos(theta/180.0*PI)))*k+b);
            cv::Point contourPoint2(center.x-1-radthresh*fabs(cos(theta/180.0*PI)),(center.x-1-radthresh*fabs(cos(theta/180.0*PI)))*k+b);
            while(!PointDense(srcimg,contourPoint1,radius,percent)||abs(srcimg.at<uchar>(contourPoint1.y,contourPoint1.x)-srcimg.at<uchar>(center.y,center.x))<15){
                contourPoint1.x=contourPoint1.x+1;
                contourPoint1.y=contourPoint1.x*k+b;
                if(contourPoint1.y>=srcimg.rows||contourPoint1.x>=srcimg.cols||contourPoint1.y<0||contourPoint1.x<0)
                    break;
            }
            contours.push_back(cv::Point(contourPoint1.x,(contourPoint1.x)*k+b));
            while(!PointDense(srcimg,contourPoint2,radius,percent)||abs(srcimg.at<uchar>(contourPoint2.y,contourPoint2.x)-srcimg.at<uchar>(center.y,center.x))<15){
                contourPoint2.x=contourPoint2.x-1;
                contourPoint2.y=contourPoint2.x*k+b;
                if(contourPoint2.y>=srcimg.rows||contourPoint2.x>=srcimg.cols||contourPoint2.y<0||contourPoint2.x<0)
                    break;
            }
            contours.push_back(cv::Point(contourPoint2.x,(contourPoint2.x)*k+b));
        }
        else if(theta>=135&&theta<=180){
            cv::Point contourPoint1(center.x-1-radthresh*fabs(cos(theta/180.0*PI)),(center.x-1-radthresh*fabs(cos(theta/180.0*PI)))*k+b);
            cv::Point contourPoint2(center.x+1+radthresh*fabs(cos(theta/180.0*PI)),(center.x+1+radthresh*fabs(cos(theta/180.0*PI)))*k+b);
            while(!PointDense(srcimg,contourPoint1,radius,percent)||abs(srcimg.at<uchar>(contourPoint1.y,contourPoint1.x)-srcimg.at<uchar>(center.y,center.x))<15){
                contourPoint1.x=contourPoint1.x-1;
                contourPoint1.y=contourPoint1.x*k+b;
                if(contourPoint1.y>=srcimg.rows||contourPoint1.x>=srcimg.cols||contourPoint1.y<0||contourPoint1.x<0)
                    break;
            }
            contours.push_back(cv::Point(contourPoint1.x,(contourPoint1.x)*k+b));
            while(!PointDense(srcimg,contourPoint2,radius,percent)||abs(srcimg.at<uchar>(contourPoint2.y,contourPoint2.x)-srcimg.at<uchar>(center.y,center.x))<15){
                contourPoint2.x=contourPoint2.x+1;
                contourPoint2.y=contourPoint2.x*k+b;
                if(contourPoint2.y>=srcimg.rows||contourPoint2.x>=srcimg.cols||contourPoint2.y<0||contourPoint2.x<0)
                    break;
            }
            contours.push_back(cv::Point(contourPoint2.x,(contourPoint2.x)*k+b));
        }
        else{
            cv::Point contourPoint1((center.y+1+radthresh*fabs(sin(theta/180.0*PI))-b)/k,center.y+1+radthresh*fabs(sin(theta/180.0*PI)));
            cv::Point contourPoint2((center.y-1-radthresh*fabs(sin(theta/180.0*PI))-b)/k,center.y-1-radthresh*fabs(sin(theta/180.0*PI)));
            while(!PointDense(srcimg,contourPoint1,radius,percent)||abs(srcimg.at<uchar>(contourPoint1.y,contourPoint1.x)-srcimg.at<uchar>(center.y,center.x))<15){
                contourPoint1.y=contourPoint1.y+1;
                contourPoint1.x=(contourPoint1.y-b)/k;
                if(contourPoint1.y>=srcimg.rows||contourPoint1.x>=srcimg.cols||contourPoint1.y<0||contourPoint1.x<0)
                    break;
            }
            contours.push_back(cv::Point((contourPoint1.y-b)/k,contourPoint1.y));
            while(!PointDense(srcimg,contourPoint2,radius,percent)||abs(srcimg.at<uchar>(contourPoint2.y,contourPoint2.x)-srcimg.at<uchar>(center.y,center.x))<15){
                contourPoint2.y=contourPoint2.y-1;
                contourPoint2.x=(contourPoint2.y-b)/k;
                if(contourPoint2.y>=srcimg.rows||contourPoint2.x>=srcimg.cols||contourPoint2.y<0||contourPoint2.x<0)
                    break;
            }
            contours.push_back(cv::Point((contourPoint2.y-b)/k,contourPoint2.y));
        }
    }
    return 0;
}

int edgedetect::ScanLine(cv::Mat srcimg, int lineori, vector<cv::Point>& contours,int startline, int radius, float percent){
    if(srcimg.empty()||srcimg.cols<radius*2+2||srcimg.rows<radius*2+2||startline<0)
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    if(lineori==SCAN_DIRECTION_UP){
        for(int i=radius+1;i<srcimg.cols-radius-1;++i){
            cv::Point seed(i,startline);
            srcimg.at<uchar>(seed.y,seed.x)=0;
            cv::Point contourpoint(i,seed.y-1);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y-=1;
                if(contourpoint.y==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(lineori==SCAN_DIRECTION_RIGHT){
        for(int i=radius+1;i<srcimg.rows-radius-1;++i){
            cv::Point seed(startline,i);
            srcimg.at<uchar>(seed.y,seed.x)=0;
            cv::Point contourpoint(seed.x+1,i);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x+=1;
                if(contourpoint.x>=srcimg.cols)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(lineori==SCAN_DIRECTION_DOWN){
        for(int i=radius+1;i<srcimg.cols-radius-1;++i){
            cv::Point seed(i,startline);
            srcimg.at<uchar>(seed.y,seed.x)=0;
            cv::Point contourpoint(i,seed.y+1);
            while(!PointDense(srcimg,contourpoint,radius,percent)||abs(srcimg.at<uchar>(contourpoint.y,contourpoint.x)-srcimg.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y+=1;
                if(contourpoint.y>=srcimg.rows)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(lineori==SCAN_DIRECTION_LEFT){
        for(int i=radius+1;i<srcimg.rows-radius-1;++i){
            cv::Point seed(startline,i);
            srcimg.at<uchar>(seed.y,seed.x)=0;
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

int edgedetect::PointDense(cv::Mat srcimg, cv::Point center, int rad, float thresh){
    if(srcimg.empty())
        return -1;
    if(srcimg.channels()==3)
        cv::cvtColor(srcimg,srcimg,CV_BGR2GRAY);
    if(center.x-rad-1<0||center.y-rad-1<0||center.x+rad+1>=srcimg.cols||center.y+rad+1>=srcimg.rows)
    {return 0;}
    cv::Mat maskimg=cv::Mat::zeros(rad*2+2,rad*2+2,CV_8UC1);
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

/* Krisch 边缘检测算法*/
cv::Mat edgedetect::krisch(cv::InputArray src,int borderType)
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

/*离散的二维卷积运算*/
void edgedetect::conv2D(cv::InputArray _src, cv::InputArray _kernel, cv::OutputArray _dst, int ddepth, cv::Point anchor, int borderType)
{
    //卷积核顺时针旋转180
    cv::Mat kernelFlip;
    cv::flip(_kernel, kernelFlip, -1);
    //针对每一个像素,领域对应元素相乘然后相加
    cv::filter2D(_src, _dst, CV_32FC1, _kernel, anchor, 0.0, borderType);
}

void edgedetect::GetSeedPoints(cv::Mat srcimg,int offset,int orientation,int startline,int endline,int threshold, vector<cv::Point>& seedpoints){
    vector<vector<float>> pixelcurves;
    Pixel2Curve(srcimg, startline, endline, orientation, pixelcurves);
    vector<int> seedpointspos;
    GetSeedPos(pixelcurves,offset,threshold,seedpointspos);
    for(int i=0;i<seedpointspos.size();++i){
        if(orientation==SCAN_DIRECTION_RIGHT){
            seedpoints.push_back(cv::Point(seedpointspos[i]+5,startline+i));
        }
        else if(orientation==SCAN_DIRECTION_DOWN){
            seedpoints.push_back(cv::Point(startline+i,seedpointspos[i]+5));
        }
        else if(orientation==SCAN_DIRECTION_LEFT){
            seedpoints.push_back(cv::Point(srcimg.cols-1-5-seedpointspos[i],startline+i));
        }
        else if(orientation==SCAN_DIRECTION_UP){
            seedpoints.push_back(cv::Point(startline+i,srcimg.rows-1-5-seedpointspos[i]));
        }
    }
}

void edgedetect::GetSrcEdgePoints(cv::Mat srcimg,int offset0,int orientation,int offset,int startline,int endline,int threshold, int threshold1, vector<cv::Point>& edgepoints){
    vector<vector<float>> pixelcurves;
    Pixel2Curve(srcimg, startline, endline, orientation, pixelcurves);
    vector<int> seedpointspos;
    vector<int> edgepointspos;
    GetSeedPos(pixelcurves, offset0,threshold1,seedpointspos);
    for(int i=0;i<seedpointspos.size();++i){
        seedpointspos[i]+=offset;
    }
    if(seedpointspos.size()==0){
        return;
    }
    for(int i=0;i<pixelcurves.size();++i){
        for(int j=seedpointspos[i];j<pixelcurves[i].size();++j){
            if(pixelcurves[i][j]>threshold){
                edgepointspos.push_back(j);
                break;
            }
        }
    }
    for(int i=0;i<edgepointspos.size();++i){
        if(orientation==SCAN_DIRECTION_RIGHT){
            edgepoints.push_back(cv::Point(edgepointspos[i]+5,startline+i));
        }
        else if(orientation==SCAN_DIRECTION_DOWN){
            edgepoints.push_back(cv::Point(startline+i,edgepointspos[i]+5));
        }
        else if(orientation==SCAN_DIRECTION_LEFT){
            edgepoints.push_back(cv::Point(srcimg.cols-1-5-edgepointspos[i],startline+i));
        }
        else if(orientation==SCAN_DIRECTION_UP){
            edgepoints.push_back(cv::Point(startline+i,srcimg.rows-1-5-edgepointspos[i]));
        }
    }
}

void edgedetect::GetEdgePoint(cv::Mat srcimg, vector<cv::Point> seedpoints, int orientation, int thresh,int radius,float percent,  vector<cv::Point>& contours){
    cv::Mat edge = krisch(srcimg);
    cv::threshold(edge, edge, thresh, 255, cv::THRESH_BINARY);
    //cv::imshow("edge",edge);
    if(orientation==SCAN_DIRECTION_UP){
        for(int i=0;i<seedpoints.size();++i){
            cv::Point seed=seedpoints[i];
            edge.at<uchar>(seed.y,seed.x)=0;
            cv::Point contourpoint(seed.x,seed.y-1);
            while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y-=1;
                if(contourpoint.y==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(orientation==SCAN_DIRECTION_RIGHT){
        for(int i=0;i<seedpoints.size();++i){
            cv::Point seed=seedpoints[i];
            edge.at<uchar>(seed.y,seed.x)=0;
            cv::Point contourpoint(seed.x+1,seed.y);
            while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x+=1;
                if(contourpoint.x>=edge.cols)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(orientation==SCAN_DIRECTION_DOWN){
        for(int i=0;i<seedpoints.size();++i){
            cv::Point seed=seedpoints[i];
            edge.at<uchar>(seed.y,seed.x)=0;
            cv::Point contourpoint(seed.x,seed.y+1);
            while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.y+=1;
                if(contourpoint.y>=edge.rows)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
    else if(orientation==SCAN_DIRECTION_LEFT){
        for(int i=0;i<seedpoints.size();++i){
            cv::Point seed=seedpoints[i];
            edge.at<uchar>(seed.y,seed.x)=0;
            cv::Point contourpoint(seed.x-1,seed.y);
            while(!PointDense(edge,contourpoint,radius,percent)||abs(edge.at<uchar>(contourpoint.y,contourpoint.x)-edge.at<uchar>(seed.y,seed.x))<15){
                contourpoint.x-=1;
                if(contourpoint.x==0)
                    break;
            }
            contours.push_back(contourpoint);
        }
    }
}

void edgedetect::GetSeedPointsSec(cv::Mat srcimg,int offset,int orientation,int startline,int endline,int threshold, vector<cv::Point>& seedpoints){
    vector<vector<float>> pixelcurves;
    Pixel2Curve(srcimg, startline, endline, orientation, pixelcurves);
    vector<int> seedpointspos;
    GetSeedPosSecond(pixelcurves,offset,threshold,seedpointspos);
    for(int i=0;i<seedpointspos.size();++i){
        if(orientation==SCAN_DIRECTION_RIGHT){
            seedpoints.push_back(cv::Point(seedpointspos[i]+5,startline+i));
        }
        else if(orientation==SCAN_DIRECTION_DOWN){
            seedpoints.push_back(cv::Point(startline+i,seedpointspos[i]+5));
        }
        else if(orientation==SCAN_DIRECTION_LEFT){
            seedpoints.push_back(cv::Point(srcimg.cols-1-5-seedpointspos[i],startline+i));
        }
        else if(orientation==SCAN_DIRECTION_UP){
            seedpoints.push_back(cv::Point(startline+i,srcimg.rows-1-5-seedpointspos[i]));
        }
    }
}

int edgedetect::Pixel2Curve(cv::Mat srcimg,int startpos, int endpos, int orientation, vector<vector<float>>& curves){
    curves.clear();
    if(orientation==SCAN_DIRECTION_RIGHT){//从左往右
        for(int i=startpos;i<endpos-1;i=i+1){
            vector<float> curve;
            for(int j=5;j<srcimg.cols-1;++j){
                float kerneldata=(float)(srcimg.at<uchar>(i,j)+srcimg.at<uchar>(i,j+1)+srcimg.at<uchar>(i+1,j)+srcimg.at<uchar>(i+1,j+1))/4;
                curve.push_back(kerneldata);
            }
            curves.push_back(curve);
        }
    }
    else if(orientation==SCAN_DIRECTION_DOWN){//从上往下
        for(int i=startpos;i<endpos-1;i=i+1){
            vector<float> curve;
            for(int j=5;j<srcimg.rows-1;++j){
                float kerneldata=(float)(srcimg.at<uchar>(j,i)+srcimg.at<uchar>(j,i+1)+srcimg.at<uchar>(j+1,i)+srcimg.at<uchar>(j+1,i+1))/4;
                curve.push_back(kerneldata);
            }
            curves.push_back(curve);
        }
    }
    else if(orientation==SCAN_DIRECTION_LEFT){//从右往左
        for(int i=startpos;i<endpos-1;i=i+1){
            vector<float> curve;
            for(int j=srcimg.cols-1-5;j>0;--j){
                float kerneldata=(float)(srcimg.at<uchar>(i,j)+srcimg.at<uchar>(i,j-1)+srcimg.at<uchar>(i+1,j)+srcimg.at<uchar>(i+1,j-1))/4;
                curve.push_back(kerneldata);
            }
            curves.push_back(curve);
        }
    }
    else if(orientation==SCAN_DIRECTION_UP){//从下往上
        for(int i=startpos;i<endpos-1;i=i+1){
            vector<float> curve;
            for(int j=srcimg.rows-1-5;j>0;--j){
                float kerneldata=(float)(srcimg.at<uchar>(j,i)+srcimg.at<uchar>(j,i+1)+srcimg.at<uchar>(j-1,i)+srcimg.at<uchar>(j-1,i+1))/4;
                curve.push_back(kerneldata);
            }
            curves.push_back(curve);
        }
    }
    return 0;
}

void edgedetect::GetSeedPos(vector<vector<float>> curves, int offset,int thresh, vector<int>& seedpoints){
    seedpoints.clear();
    for(int i=0;i<curves.size();++i){
        int startpos=0,endpos=curves[i].size()-2;
        int flag=0;
        if(offset<2){
            offset=2;
        }
        for(int j=offset;j<curves[i].size()-2;++j){
            if(curves[i][j-2]+curves[i][j-1]+curves[i][j]+curves[i][j+1]+curves[i][j+2]<thresh*5){
                if(!flag){
                    flag=1;
                    startpos=j;
                }
            }
            else{
                if(flag){
                    endpos=j;
                    break;
                }
            }
        }
        if((startpos+endpos)/2>=2&&(startpos+endpos)/2<curves[i].size()-2){
            seedpoints.push_back((startpos+endpos)/2);
        }
    }
}

void edgedetect::GetSeedPosSecond(vector<vector<float>> curves,int offset,int thresh, vector<int>& seedpoints){
    seedpoints.clear();
    for(int i=0;i<curves.size();++i){
        int startpos=0,endpos=curves[i].size()-2;
        int flag=0,num=0;
        for(int j=offset;j<curves[i].size()-2;++j){
            if(curves[i][j-2]+curves[i][j-1]+curves[i][j]+curves[i][j+1]+curves[i][j+2]<thresh*5){
                flag=1;
                if(flag&&num==1){
                    startpos=j;
                    num++;
                }
            }
            else{
                if(flag){
                    num++;
                }
                if(num==3){
                    endpos=j;
                    break;
                }
                flag=0;
            }
        }
        if((startpos+endpos)/2>=2&&(startpos+endpos)/2<curves[i].size()-2){
            seedpoints.push_back((startpos+endpos)/2);
        }
    }
}

vector<cv::Point> edgedetect::ContoursCut(vector<cv::Point> contours,int startangle,int endangle){
    vector<cv::Point> segcontours;
    if(startangle>endangle){
        for(int i=startangle;i<360;++i){
            if(i<180&&i>=1){segcontours.push_back(contours[359-(i-1)*2]);}
            else if(i>180&&i<=360){segcontours.push_back(contours[359-2*(i-180)+1]);}
            else if(i==180){segcontours.push_back(contours[1]);}
            else{segcontours.push_back(contours[0]);}
        }
        for(int i=0;i<endangle;++i){
            if(i<180&&i>=1){segcontours.push_back(contours[359-(i-1)*2]);}
            else if(i>180&&i<=360){segcontours.push_back(contours[359-2*(i-180)+1]);}
            else if(i==180){segcontours.push_back(contours[1]);}
            else{segcontours.push_back(contours[0]);}
        }
    }
    else{
        for(int i=startangle;i<endangle;++i){
            if(i<180&&i>=1){segcontours.push_back(contours[359-(i-1)*2]);}
            else if(i>180&&i<=360){segcontours.push_back(contours[359-2*(i-180)+1]);}
            else if(i==180){segcontours.push_back(contours[1]);}
            else{segcontours.push_back(contours[0]);}
        }
    }
    return segcontours;
}

vector<float> edgedetect::FitCircle(vector<cv::Point>& contours){
    float a=0,b=0,r=0;
    double sumX,sumY,sumR;
    for(int i=0;i<contours.size();++i){
        sumX+=contours[i].x;
        sumY+=contours[i].y;
    }
    a=sumX/contours.size();
    b=sumY/contours.size();
    for(int i=0;i<contours.size();++i){
        sumR+=sqrt((contours[i].x-a)*(contours[i].x-a)+(contours[i].y-b)*(contours[i].y-b));
    }
    r=sumR/contours.size();
    return vector<float>{a,b,r};
}
void edgedetect::LineFitLeastSquaresf(vector<cv::Point2f> contours, vector<float> &vResult)
{
    float A = 0.0;
    float B = 0.0;
    float C = 0.0;
    float D = 0.0;
    float E = 0.0;
    float F = 0.0;

    for (int i = 0; i < contours.size(); i++)
    {
        A += contours[i].x * contours[i].x;
        B += contours[i].x;
        C += contours[i].x * contours[i].y;
        D += contours[i].y;
    }
    float a, b, temp = 0;
    if( temp = (contours.size()*A - B*B) )
    {
        a = (contours.size()*C - B*D) / temp;
        b = (A*D - B*C) / temp;
    }
    else
    {
        a = 1;
        b = 0;
    }
    float Xmean, Ymean;
    Xmean = B / contours.size();
    Ymean = D / contours.size();

    float tempSumXX = 0.0, tempSumYY = 0.0;
    for (int i=0; i<contours.size(); i++)
    {
        tempSumXX += (contours[i].x - Xmean) * (contours[i].x - Xmean);
        tempSumYY += (contours[i].y - Ymean) * (contours[i].y - Ymean);
        E += (contours[i].x - Xmean) * (contours[i].y - Ymean);
    }
    F = sqrt(tempSumXX) * sqrt(tempSumYY);

    float r;
    r = E / F;

    vResult.push_back(a);
    vResult.push_back(b);
    vResult.push_back(r*r);
}

float edgedetect::GetParallelism(vector<cv::Point2f> contour,int flag){
    if(flag==1){//X
        float max=0;
        float min=10000;
        for(int i=0;i<contour.size();++i){
            if(max<contour[i].x){
                max=contour[i].x;
            }
            if(min>contour[i].x){
                min=contour[i].x;
            }
        }
        return max-min;
    }
    else if(flag==2){//Y
        float max=0;
        float min=10000;
        for(int i=0;i<contour.size();++i){
            if(max<contour[i].y){
                max=contour[i].y;
            }
            if(min>contour[i].y){
                min=contour[i].y;
            }
        }
        return max-min;
    }
    return -1;
}

float edgedetect::GetLineDist(vector<cv::Point2f> contour1,vector<cv::Point2f> contour2,int flag){
    if(flag==1){//X
        float sum1=0;
        float sum2=0;
        for(int i=0;i<contour1.size();++i){
            sum1+=contour1[i].x;
        }
        for(int i=0;i<contour2.size();++i){
            sum2+=contour2[i].x;
        }
        return fabs(sum1/contour1.size()-sum2/contour2.size());
    }
    else if(flag==2){//Y
        float sum1=0;
        float sum2=0;
        for(int i=0;i<contour1.size();++i){
            sum1+=contour1[i].y;
        }
        for(int i=0;i<contour2.size();++i){
            sum2+=contour2[i].y;
        }
        return fabs(sum1/contour1.size()-sum2/contour2.size());
    }
    return -1;
}

float edgedetect::GetLine2LineMid(vector<cv::Point2f> contour1,vector<cv::Point2f> contour2,int flag){
    if(flag==1){//X
        float ypos=(contour1[0].y+contour1[contour1.size()-1].y)/2;
        for(int i=0;i<contour1.size();++i){
            cv::Point2f temp=contour1[i];
            contour1[i].x=temp.y;
            contour1[i].y=temp.x;
        }
        for(int i=0;i<contour2.size();++i){
            cv::Point2f temp=contour2[i];
            contour2[i].x=temp.y;
            contour2[i].y=temp.x;
        }
        vector<float> xparas1;
        LineFitLeastSquaresf(contour1,xparas1);
        vector<float> xparas2;
        LineFitLeastSquaresf(contour2,xparas2);
        float res=0;
        res=fabs((xparas1[0]*ypos+xparas1[1])+(xparas2[0]*ypos+xparas2[1]))/2;
        return res;
    }
    else if(flag==2){//Y
        float xpos=(contour1[0].x+contour1[contour1.size()-1].x)/2;
        vector<float> yparas1;
        LineFitLeastSquaresf(contour1,yparas1);
        vector<float> yparas2;
        LineFitLeastSquaresf(contour2,yparas2);
        float res=0;
        res=fabs((yparas1[0]*xpos+yparas1[1])+(yparas2[0]*xpos+yparas2[1]))/2;
        return res;
    }
    return -1;
}

float edgedetect::GetRoundness(vector<cv::Point> contour, cv::Point2f center){
    float maxrad=0;
    float minrad=10000;
    for(int i=0;i<contour.size();++i){
        float rad=Point2Point<cv::Point,cv::Point2f>(contour[i],center);
        if(maxrad<rad){
            maxrad=rad;
        }
        if(minrad>rad){
            minrad=rad;
        }
    }
    return fabs(minrad-maxrad);
}

template<typename T1,typename T2>
float edgedetect::Point2Point(T1 A,T2 B){
    return sqrt((A.x-B.x)*(A.x-B.x)+(A.y-B.y)*(A.y-B.y));
}

float edgedetect::GetPosition(cv::Point2f stdposition,cv::Point2f measurepos){
    return 2*Point2Point<cv::Point2f,cv::Point2f>(stdposition,measurepos);
}
