#include "edgedetecction.h"

void scharr(InputArray src, OutputArray dst, int ddepth, int x, int y, int borderType)
{
    CV_Assert(!(x == 0 && y == 0));
    Mat scharr_x = (Mat_<float>(3, 3) << 3, 0, -3, 10, 0, -10, 3, 0, -3);
    Mat scharr_y = (Mat_<float>(3, 3) << 3, 10, 3, 0, 0, 0, -3, -10, -3);
    //当 x 不等于零时，src 和 scharr_x 卷积
    if (x != 0)
    {
        conv2D(src, scharr_x, dst, ddepth, Point(-1, -1),borderType);
    }
    //当 y 不等于零时，src 和 scharr_y 卷积
    if (y != 0)
    {
        conv2D(src, scharr_y, dst, ddepth, Point(-1, -1), borderType);
    }
}
