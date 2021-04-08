#ifndef GLCM_H
#define GLCM_H
#include"headerall.h"

typedef struct _GLCMFeatures
{
    _GLCMFeatures()
        : energy(0.0)
        , entropy(0.0)
        , contrast(0.0)
        , idMoment(0.0)
    {

    }

    double energy;      // 能量 
    double entropy;     // 熵
    double contrast;    // 对比度
    double idMoment;    // 逆差分矩， inverse difference moment

} GLCMFeatures;

class GLCM
{
public:
    GLCM();
    ~GLCM();

public:
    // 枚举灰度共生矩阵的方向
    enum 
    {
        GLCM_HORIZATION = 0,        // 水平
        GLCM_VERTICAL = 1,          // 垂直
        GLCM_ANGLE45 = 2,           // 45度角
        GLCM_ANGLE135 = 3,           // 135度角
		GLCM_ANGLEALL = 4           // 所有度角
    };

public:
    // 计算灰度共生矩阵
	void calGLCM(cv::Mat &inputImg, cv::Mat& vecGLCM, int angle);
	//void calGLCM(cv::Mat &inputImg, cv::Mat& vecGLCM, int angle);
    // 计算特征值
    void getGLCMFeatures(cv::Mat& vecGLCM0, GLCMFeatures& features);
	// 计算均值特征值
	void getavgfeatures(cv::Mat &img, vector<float> &feaVec);
    // 初始化灰度共生矩阵
    void initGLCM(cv::Mat& vecGLCM, int size = 16);
    // 设置灰度划分等级，默认值为 16
    void setGrayLevel(int grayLevel) { m_grayLevel = grayLevel; }
    // 获取灰度等级
    int getGrayLevel() const { return m_grayLevel; }
private:
    // 计算水平灰度共生矩阵
	void getHorisonGLCM(cv::Mat &src, cv::Mat &dst, int imgWidth, int imgHeight);
	//void getHorisonGLCM(cv::Mat &src, cv::Mat &dst, int imgWidth, int imgHeight);
	// 计算垂直灰度共生矩阵
	void getVertialGLCM(cv::Mat &src, cv::Mat &dst, int imgWidth, int imgHeight);
	//void getVertialGLCM(cv::Mat &src, cv::Mat &dst, int imgWidth, int imgHeight);
    // 计算 45 度灰度共生矩阵
	void getGLCM45(cv::Mat &src, cv::Mat &dst, int imgWidth, int imgHeight);
	//void getGLCM45(cv::Mat &src, cv::Mat &dst, int imgWidth, int imgHeight);
    // 计算 135 度灰度共生矩阵
	void getGLCM135(cv::Mat &src, cv::Mat &dst, int imgWidth, int imgHeight);
	//void getGLCM135(cv::Mat &src, cv::Mat &dst, int imgWidth, int imgHeight);

	//void getGLCMALL(VecGLCM &src, vector<VecGLCM> &dst, int imgWidth, int imgHeight);
	void getGLCMALL(cv::Mat &src, vector<cv::Mat> &dst, int imgWidth, int imgHeight);
	vector<cv::Mat> vecGLCMVec;
private:
    int m_grayLevel;        // 将灰度共生矩阵划分为 grayLevel 个等级



};
#endif // BFILESYSTEMTEST_H
