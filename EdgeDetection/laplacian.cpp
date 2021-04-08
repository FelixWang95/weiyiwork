#include "edgedetecction.h"

/*laplacian 卷积*/
void laplacian(InputArray src, OutputArray dst, int ddepth,int borderType)
{
    Mat lapKernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
    conv2D(src, lapKernel, dst, ddepth, Point(-1, -1), borderType);
}
