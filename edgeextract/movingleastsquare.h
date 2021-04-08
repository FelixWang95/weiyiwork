#ifndef MOVINGLEASTSQUARE_H
#define MOVINGLEASTSQUARE_H

#include<opencv2/opencv.hpp>
#include "_matrix.h"

using namespace std;
using namespace cv;

#define M 8//采样点的个数
#define N 6//基数个数，线性基三个，二次基六个，三次基十个

typedef struct point
{
    double x;
    double y;
}point;

typedef struct Lnode
{
    int data;
    struct Lnode *next;
}Lnode,*Linklist;

extern void Beizer(vector<Linklist>&p_M);

class MovingLeastSquare
{
public:
    MovingLeastSquare();
    int max_x;
    int min_x;

    void f(float w[],float x[],float y[],float sumf[][N],float p[][N]);
    int MLS_Calc(int x_val,int y_val,float x[],float y[],float z[]);
    void x_y(float x[],float y[],float z0[],float z1[],float z2[],Mat im);
    void MLS();
};

#endif // MOVINGLEASTSQUARE_H
