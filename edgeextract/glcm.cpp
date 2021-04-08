#include "glcm.h"

GLCM::GLCM() : m_grayLevel(16)
{

}

GLCM::~GLCM()
{
	vecGLCMVec.clear();
}

//==============================================================================
// 函数名称: initGLCM
// 参数说明: vecGLCM,要进行初始化的共生矩阵,为二维方阵
//          size, 二维矩阵的大小,必须与图像划分的灰度等级相等
// 函数功能: 初始化二维矩阵
//==============================================================================

void GLCM::initGLCM(cv::Mat& vecGLCM, int size)
{
    assert(size == m_grayLevel);
	for (int i = 0; i < 4; i++)
	{
		cv::Mat tp = cv::Mat::zeros(m_grayLevel, m_grayLevel, CV_32SC1);
		vecGLCMVec.push_back(tp.clone());
	}
	vecGLCM = cv::Mat::zeros(m_grayLevel, m_grayLevel, CV_32SC1);
}

//==============================================================================
// 函数名称: getHorisonGLCM
// 参数说明: src,要进行处理的矩阵,源数据
//          dst,输出矩阵,计算后的矩阵，即要求的灰度共生矩阵
//          imgWidth, 图像宽度
//          imgHeight, 图像高度
// 函数功能: 计算水平方向的灰度共生矩阵
//==============================================================================

void GLCM::getHorisonGLCM(cv::Mat &src, cv::Mat &dst, int imgWidth, int imgHeight)
{
    int height = imgHeight;
    int width = imgWidth;

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width - 1; ++j)
        {
            int rows = src.at<int>(i,j);
			int cols = src.at<int>(i,j + 1);
			dst.at<int>(rows, cols)=dst.at<int>(rows, cols) + 1;
        }
    }


}

//==============================================================================
// 函数名称: getVertialGLCM
// 参数说明: src,要进行处理的矩阵,源数据
//          dst,输出矩阵,计算后的矩阵，即要求的灰度共生矩阵
//          imgWidth, 图像宽度
//          imgHeight, 图像高度
// 函数功能: 计算垂直方向的灰度共生矩阵
//==============================================================================

void GLCM::getVertialGLCM(cv::Mat &src, cv::Mat &dst, int imgWidth, int imgHeight)
{
    int height = imgHeight;
    int width = imgWidth;
    for (int i = 0; i < height - 1; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
			int rows = src.at<int>(i, j);
			int cols = src.at<int>(i+1, j);
			dst.at<int>(rows, cols) = dst.at<int>(rows, cols) + 1;
        }
    }
}

//==============================================================================
// 函数名称: getGLCM45
// 参数说明: src,要进行处理的矩阵,源数据
//          dst,输出矩阵,计算后的矩阵，即要求的灰度共生矩阵
//          imgWidth, 图像宽度
//          imgHeight, 图像高度
// 函数功能: 计算45度的灰度共生矩阵
//==============================================================================

void GLCM::getGLCM45(cv::Mat &src, cv::Mat &dst, int imgWidth, int imgHeight)
{
    int height = imgHeight;
    int width = imgWidth;
    for (int i = 0; i < height - 1; ++i)
    {
        for (int j = 0; j < width - 1; ++j)
        {
            int rows = src.at<int>(i,j);
            int cols = src.at<int>(i + 1,j + 1);
			dst.at<int>(rows, cols) = dst.at<int>(rows, cols) + 1;
        }
    }
}


//==============================================================================
// 函数名称: getGLCM135
// 参数说明: src,要进行处理的矩阵,源数据
//          dst,输出矩阵,计算后的矩阵，即要求的灰度共生矩阵
//          imgWidth, 图像宽度
//          imgHeight, 图像高度
// 函数功能: 计算 135 度的灰度共生矩阵
//==============================================================================

void GLCM::getGLCM135(cv::Mat &src, cv::Mat &dst, int imgWidth, int imgHeight)
{
    int height = imgHeight;
    int width = imgWidth;
    for (int i = 0; i < height - 1; ++i)
    {
        for (int j = 1; j < width; ++j)
        {
            int rows = src.at<int>(i,j);
            int cols = src.at<int>(i + 1,j - 1);
			dst.at<int>(rows, cols) = dst.at<int>(rows, cols) + 1;
        }
    }
}
void GLCM::getGLCMALL(cv::Mat &src, vector<cv::Mat> &dst, int imgWidth, int imgHeight)
{
	int height = imgHeight;
	int width = imgWidth;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width ; ++j)
		{
			if (i>=0 && j>=0 && i< height && j < width - 1)
			{
                int rows = src.at<int>(i, j);
                int cols = src.at<int>(i, j + 1);
                dst[0].at<int>(rows, cols) = dst[0].at<int>(rows, cols) + 1;

			}
			if (i >= 0 && j >= 0 && i < height - 1 && j<width)
			{
                int rows = src.at<int>(i, j);
                int cols = src.at<int>(i + 1, j);
                dst[1].at<int>(rows, cols) = dst[1].at<int>(rows, cols) + 1;

			}
			if (i >= 0 && j >= 0 && i< height - 1 && j< width - 1)
			{
                int rows = src.at<int>(i, j);
                int cols = src.at<int>(i + 1, j + 1);
                dst[2].at<int>(rows, cols) = dst[2].at<int>(rows, cols) + 1;

			}
			if (i >= 0 && j >= 1 && i<height - 1 && j<width)
			{
                int rows = src.at<int>(i, j);
                int cols = src.at<int>(i + 1, j - 1);
                dst[3].at<int>(rows, cols) = dst[3].at<int>(rows, cols) + 1;

			}
			
		}
	}
}
//==============================================================================
// 函数名称: calGLCM
// 参数说明: inputImg,要进行纹理特征计算的图像,为灰度图像
//          vecGLCM, 输出矩阵,根据灰度图像计算出的灰度共生阵
//          angle,灰度共生矩阵的方向,有水平、垂直、45度、135度四个方向
// 函数功能: 计算灰度共生矩阵
//==============================================================================

void GLCM::calGLCM(cv::Mat& inputImg, cv::Mat& vecGLCM, int angle)
{
    assert(inputImg.channels() == 1);
    cv::Mat src;
    inputImg.convertTo(src, CV_32SC1);
    int height = src.rows;
    int width = src.cols;
    int maxGrayLevel = 0;
    // 寻找最大像素灰度最大值
	double minv = 0.0, maxv = 0.0;
	double* minp = &minv;
	double* maxp = &maxv;

	cv::minMaxIdx(src, minp, maxp);
	maxGrayLevel = int(maxv);

    ++maxGrayLevel;
    cv::Mat tempMat(src.rows,src.cols,CV_32SC1);

    if (maxGrayLevel > 16)//若灰度级数大于16，则将图像的灰度级缩小至16级，减小灰度共生矩阵的大小。
    {
        for (int k1 = 0; k1 < tempMat.rows; k1++)
        {
            for (int k2 = 0; k2 < tempMat.cols; k2++)
            {
                tempMat.at<int>(k1, k2)=src.at<int>(k1, k2) / m_grayLevel;
            }
        }
        if (angle == GLCM_HORIZATION)  // 水平方向
			getHorisonGLCM(tempMat, vecGLCM, width, height);
        if (angle == GLCM_VERTICAL)    // 垂直方向
			getVertialGLCM(tempMat, vecGLCM, width, height);
        if (angle == GLCM_ANGLE45)     // 45 度灰度共生阵
			getGLCM45(tempMat, vecGLCM, width, height);
        if (angle == GLCM_ANGLE135)    // 135 度灰度共生阵
			getGLCM135(tempMat, vecGLCM, width, height);
		if (angle == GLCM_ANGLEALL)    // 所有灰度共生阵
			getGLCMALL(tempMat, vecGLCMVec, width, height);
    }
    else//若灰度级数小于16，则生成相应的灰度共生矩阵
    {
        for (int k1 = 0; k1 < tempMat.rows; k1++)
        {
            for (int k2 = 0; k2 < tempMat.cols; k2++)
            {
                tempMat.at<int>(k1, k2)=src.at<int>(k1, k2) ;
            }
        }
        if (angle == GLCM_HORIZATION)  // 水平方向
            getHorisonGLCM(tempMat, vecGLCM, width, height);
        if (angle == GLCM_VERTICAL)    // 垂直方向
			getVertialGLCM(tempMat, vecGLCM, width, height);
        if (angle == GLCM_ANGLE45)     // 45 度灰度共生阵
			getGLCM45(tempMat, vecGLCM, width, height);
        if (angle == GLCM_ANGLE135)    // 135 度灰度共生阵
			getGLCM135(tempMat, vecGLCM, width, height);
		if (angle == GLCM_ANGLEALL)    // 所有灰度共生阵
			getGLCMALL(tempMat, vecGLCMVec, width, height);
    }

}


//==============================================================================
// 函数名称: getGLCMFeatures
// 参数说明: vecGLCM, 输入矩阵,灰度共生阵
//          features,灰度共生矩阵计算的特征值,主要包含了能量、熵、对比度、逆差分矩
// 函数功能: 根据灰度共生矩阵计算的特征值
//==============================================================================

void GLCM::getGLCMFeatures(cv::Mat& vecGLCM0, GLCMFeatures& features)
{
	cv::Mat vecGLCM;
	vecGLCM0.convertTo(vecGLCM, CV_32FC1);
	cv::Mat temp;
    //cout << vecGLCM << endl;
	Scalar st = cv::sum(vecGLCM);
	temp = vecGLCM.mul(float(1.0 / st[0]));
	//normalize(vecGLCM, temp, cv::NORM_L1);
	//cout << temp << endl;
    for (int i = 0; i < m_grayLevel; ++i)
    {
        for (int j = 0; j < m_grayLevel; ++j)
        {
			features.energy += temp.at<float>(i, j) * temp.at<float>(i, j);

			if (temp.at<float>(i, j)>0)
				features.entropy -= temp.at<float>(i, j) * log(temp.at<float>(i, j));               //熵     
			features.contrast += (double)(i - j)*(double)(i - j)*temp.at<float>(i, j);        //对比度
			features.idMoment += temp.at<float>(i, j) / (1 + (double)(i - j)*(double)(i - j));//逆差矩
        }
    }
}

void GLCM::getavgfeatures(cv::Mat &img, vector<float> &feaVec)
{
	feaVec.clear();
	GLCM glcm;
	cv::Mat vec;
	glcm.initGLCM(vec);
	glcm.calGLCM(img, vec, GLCM::GLCM_ANGLEALL);
    float energy_average = 0.0, entropy_average = 0.0, contrast_average = 0.0, idMoment_average = 0.0;
    for (int i = 0; i < 4; i++)
	{

        GLCMFeatures features;
        glcm.getGLCMFeatures(glcm.vecGLCMVec[i], features);
        energy_average += features.energy;
        entropy_average += features.entropy;
        contrast_average += features.contrast;
        idMoment_average += features.idMoment;
//        feaVec.push_back(features.energy),feaVec.push_back(features.entropy);
//        feaVec.push_back(features.contrast),feaVec.push_back(features.idMoment);
	}
	feaVec.push_back(energy_average / 4), feaVec.push_back(entropy_average / 4);
	feaVec.push_back(contrast_average / 4), feaVec.push_back(idMoment_average / 4);
}
