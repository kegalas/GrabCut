#ifndef __grabcut__H__
#define __grabcut__H__

#include <opencv2/opencv.hpp>
#include <cmath>

class GMM{
public:
    static int const K = 5; // 论文中指出，高斯混合模型的K值取5为佳
private:
    double const PI = acos(-1);

    double coefs[K];    // 高斯模型的混合参数
    cv::Mat mean[K];  // 均值，RGB分别的
    cv::Mat cov[K];   // 协方差

    double covDet[K];   // cov的行列式
    cv::Mat covInv[K];   // 协方差的逆矩阵

    size_t sampleCnt[K];
    size_t totalSampleCnt;

    cv::Mat colorSum[K];

public:
    GMM(){
        // 混合参数1位，RGB三通道的均值3位，其协方差矩阵3x3即9位
        std::fill(coefs, coefs+K, 0.0);
        std::fill(mean, mean+K, cv::Mat::zeros(3, 1, CV_64FC1));
        std::fill(cov, cov+K, cv::Mat::zeros(3, 3, CV_64FC1));

        std::fill(covDet, covDet+K, 0.0);
        std::fill(covInv, covInv+K, cv::Mat::zeros(3, 3, CV_64FC1));

        std::fill(sampleCnt, sampleCnt+K, 0);
        totalSampleCnt = 0;

        std::fill(colorSum, colorSum+K, cv::Mat::zeros(3, 1, CV_64FC1));

    }

    double possibility(int ki, cv::Vec3d const & color){
        // 某个color属于第ki个分量的概率
        double ret=0.0;
        if(coefs[ki]<=0) return ret;

        cv::Mat delta = cv::Mat(color)-mean[ki];
        cv::Mat m = delta.t() * covInv[ki] * delta;
        double mult = -0.5 * m.at<double>(0,0);

        ret = 1.0/pow((2.0*PI), 3.0/2.0);
        ret /= sqrt(covDet[ki]);
        ret *= cv::exp(mult);

        return ret;
    }

    double possibility(cv::Vec3d const & color){
        // 所有分量概率的和
        double ret = 0.0;

        for(int ki=0;ki<K;ki++){
            ret += possibility(ki, color);
        }

        return ret;
    }

    int whichComponent(cv::Vec3d const & color){
        // 输出某个颜色应该属于哪个分量，即概率最大的那个分量
        int k=0;
        double maxv=0.0;

        for(int ki=0;ki<K;ki++){
            double p = possibility(ki, color);
            if(p>maxv){
                maxv = p;
                k = ki;
            }
        }

        return k;
    }

    void calcCovInvAndDet(int ki){
        if(coefs[ki]<=0) return;
        covDet[ki] = cv::determinant(cov[ki]);
        cv::invert(cov[ki], covInv[ki]);
    }

    void addSample(int ki, cv::Vec3d const & color){
        // 向第ki个分量添加一个样本

    }
};

#endif


