#ifndef __grabcut__H__
#define __grabcut__H__

#include <opencv2/opencv.hpp>
#include <cmath>
#include <graph.h>
#include <random>
#include <ctime>

namespace GC{

double const PI = acos(-1);
double constexpr EPS = 1e-9;

enum
{
    BGD    = 0,  // 肯定是背景
    FGD    = 1,  // 肯定是前景
    MAY_BGD = 2,  // 更可能是背景
    MAY_FGD = 3   // 更可能是前景
};

class GMM{
public:
    static int constexpr K = 5; // 论文中指出，高斯混合模型的K值取5为佳
private:

    double coefs[K];    // 高斯模型的混合参数
    cv::Mat mean[K];  // 均值，RGB分别的
    cv::Mat cov[K];   // 协方差

    double covDet[K];   // cov的行列式
    cv::Mat covInv[K];   // 协方差的逆矩阵

    size_t sampleCnt[K]; // 该分量的样本数
    size_t totalSampleCnt; // 样本总数

    cv::Mat colorSum[K]; // 用于计算均值
    cv::Mat colorProdSum[K]; // 用于计算协方差

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
        std::fill(colorProdSum, colorProdSum+K, cv::Mat::zeros(3, 3, CV_64FC1));
    }

    double possibility(int ki, cv::Vec3d const & color) const {
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

    double possibility(cv::Vec3d const & color) const {
        // 所有分量概率的和
        double ret = 0.0;

        for(int ki=0;ki<K;ki++){
            ret += possibility(ki, color);
        }

        return ret;
    }

    int whichComponent(cv::Vec3d const & color) const {
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
        colorSum[ki] += cv::Mat(color);
        colorProdSum[ki] += cv::Mat(color)*cv::Mat(color).t();
        sampleCnt[ki]++;
        totalSampleCnt++;
    }

    void deleteAllSamples(){
        // 删除所有分量中的样本信息
        for(int ki=0;ki<K;ki++){
            colorSum[ki].setTo(cv::Scalar(0));
            colorProdSum[ki].setTo(cv::Scalar(0));
            sampleCnt[ki] = 0;
        }
        totalSampleCnt = 0;
    }

    void calcMeanAndCovWithSamples(){
        // 根据样本来计算每个分量各自的均值和协方差
        for(int ki=0;ki<K;ki++){
            int n = sampleCnt[ki];
            if(n==0){
                coefs[ki] = 0;
                continue;
            }

            coefs[ki] = static_cast<double>(n)/totalSampleCnt;
            mean[ki] = colorSum[ki] / n;
            cov[ki] = colorProdSum[ki]/n - mean[ki]*mean[ki].t();
            double dt = cv::determinant(cov[ki]);
            if(dt<EPS){
                cov[ki].at<double>(0,0) += 0.01;
                cov[ki].at<double>(1,1) += 0.01;
                cov[ki].at<double>(2,2) += 0.01;
            }

            calcCovInvAndDet(ki);
        }
    }
};

std::vector<uint8_t> kMeans(std::vector<cv::Vec3d> const & colors, int classNum=GMM::K, int iterCnt = 20){
    std::vector<uint8_t> labels;
    std::vector<cv::Vec3d> classCenter(classNum);
    std::vector<int> idx(colors.size());

    std::random_device rd;
    std::mt19937 rng(rd());

    int sz=colors.size();

    assert(sz>=classNum);

    for(int i=0;i<sz;i++){
        idx[i] = i;
    }
    std::shuffle(idx.begin(), idx.end(), rng);

    for(int i=0;i<classNum;i++){
        classCenter[i] = colors[idx[i]];
    }

    std::vector<int> classCnt(classNum);
    std::vector<cv::Vec3d> classColorSum(classNum);
    auto init = [&classCnt, &classColorSum, &classNum](){
        for(int i=0;i<classNum;i++){
            classCnt[i] = 0;
            classColorSum[i] = cv::Vec3d(0.0,0.0,0.0);
        }
    };

    while(iterCnt--){
        init();
        for(int i=0;i<sz;i++){
            double mindis = 1e15;
            int minc = 0;
            for(int j=0;j<classNum;j++){
                cv::Vec3d delta = colors[i] - classCenter[j];
                double dis = delta.dot(delta);
                if(mindis>dis){
                    mindis = dis;
                    minc = j;
                }
            }
            ++classCnt[minc];
            classColorSum[minc] += colors[i];
            labels[i] = minc;
        }
        for(int j=0;j<classNum;j++){
            classCenter[j] = classColorSum[j] / classCnt[j];
        }
    }

    return labels;
}

double calcBeta(cv::Mat const & img){
    // 计算beta，见原论文公式(5)
    double beta = 0.0;
    size_t cnt = 0;

    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            cv::Vec3d color = img.at<cv::Vec3b>(i, j);
            // 原论文这个公式因为只有黑白，所以是平方。我们这里有三个通道，计算内积
            // 原论文这个相邻的像素是8个，我想试一下4个的情况
            // 按照我的理解，beta是在全图意义上的期望，而不是每个像素都有一个beta
            for(int di=-1;di<=1;di++){
                for(int dj=-1;dj<=1;dj++){
                    if(di==0&&dj==0) continue;
                    int ii = i+di, jj = j+dj;
                    if(ii>=0&&jj>=0&&ii<img.rows&&jj<img.cols){
                        cnt++;
                        cv::Vec3d delta = color - static_cast<cv::Vec3d>(img.at<cv::Vec3b>(ii, jj));
                        beta += delta.dot(delta);
                    }
                }
            }
        }
    }

    if(beta<EPS) beta = 0.0;
    else beta = 1.0/(2.0*beta/cnt);
    // 总共选中像素点的次数

    return beta;
}

class GCGraph{

};

using GraphType = Graph<double, double, double>;

void buildSmoothnessTermGraph(GraphType* g, cv::Mat const & img, cv::Mat const & mask, double beta, double gamma = 50.0){
    // 传入一个图、一张图片、一个mask，beta的计算见前，gamma在论文中默认给50
    // 计算平滑项，即图片上的节点的边权，似乎在graph cut的论文里叫n weights
    g->reset();
    g->add_node(img.rows*img.cols);

    auto maskEqu = [&mask](int i1, int j1, int i2, int j2){
        return (mask.at<int>(i1,j1)&1)!=(mask.at<int>(i2,j2)&1);
    };

    // 原文在grabcut部分不再有dis^-1这一项，但是opencv的实现有，怀疑是论文弄错了
    auto calcWeight = [&img, &beta, &gamma](int i1, int j1, int i2, int j2){
        cv::Vec3d color = img.at<cv::Vec3b>(i1, j1);
        cv::Vec3d delta = color - static_cast<cv::Vec3d>(img.at<cv::Vec3b>(i2, j2));
        return gamma*cv::exp(-beta*delta.dot(delta));
    };

    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int idx1 = i*img.cols+j;

            // 只需要对每个节点的左、左上、上、右上建边，就可以防止重复建边
            if(i>0 && maskEqu(i, j, i-1, j)){
                int idx2 = (i-1)*img.cols+j;
                double weight = calcWeight(i, j, i-1, j);
                g->add_edge(idx1, idx2, weight, weight);
            }
            if(j>0 && maskEqu(i, j, i, j-1)){
                int idx2 = i*img.cols+j-1;
                double weight = calcWeight(i, j, i, j-1);
                g->add_edge(idx1, idx2, weight, weight);
            }
            if(i>0 && j>0 && maskEqu(i, j, i-1, j-1)){
                int idx2 = (i-1)*img.cols+j-1;
                double weight = calcWeight(i, j, i-1, j-1);
                g->add_edge(idx1, idx2, weight, weight);
            }
            if(i>0 && j<img.cols-1 && maskEqu(i, j, i-1, j+1)){
                int idx2 = (i-1)*img.cols+j+1;
                double weight = calcWeight(i, j, i-1, j+1);
                g->add_edge(idx1, idx2, weight, weight);
            }
        }
    }
}

void initMask(cv::Mat& mask, cv::Size imgSz, cv::Rect rect){
    // 通过一个矩形来初始化Mask，和论文中描述的一致，框内可能是前景，框外一定是背景
    mask.create(imgSz, CV_8UC1);
    mask.setTo(cv::Scalar(BGD));

    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, imgSz.width-rect.x);
    rect.height = std::min(rect.height, imgSz.height-rect.y);

    mask(rect).setTo(cv::Scalar(MAY_FGD));
}

void initGMMs(cv::Mat const & img, cv::Mat const & mask, GMM& fgdGMM, GMM& bgdGMM){
    // 使用K-means算法把BGD和FGB各自的初始节点分为K类，然后放到对应的GMM分量中
    std::vector<cv::Vec3d> bgdSamples, fgdSamples;
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            if(mask.at<uint8_t>(i,j)&1){
                fgdSamples.push_back(static_cast<cv::Vec3d>(img.at<cv::Vec3b>(i,j)));
            }
            else{
                bgdSamples.push_back(static_cast<cv::Vec3d>(img.at<cv::Vec3b>(i,j)));
            }
        }
    }
    std::vector<uint8_t> fgdLabel = kMeans(fgdSamples);
    std::vector<uint8_t> bgdLabel = kMeans(bgdSamples);

    fgdGMM.deleteAllSamples();
    for(int i=0, sz=fgdLabel.size();i<sz;i++){
        fgdGMM.addSample(fgdLabel[i], fgdSamples[i]);
    }
    fgdGMM.calcMeanAndCovWithSamples();

    bgdGMM.deleteAllSamples();
    for(int i=0, sz=bgdLabel.size();i<sz;i++){
        bgdGMM.addSample(bgdLabel[i], bgdSamples[i]);
    }
    bgdGMM.calcMeanAndCovWithSamples();
}

void assignGMMsToPixels(cv::Mat const & img, cv::Mat const & mask, GMM const & fgdGMM, GMM const & bgdGMM, cv::Mat& PixToComp){
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            if(mask.at<uint8_t>(i,j)&1){
                PixToComp.at<uint8_t>(i,j) = fgdGMM.whichComponent(static_cast<cv::Vec3d>(img.at<cv::Vec3b>(i,j)));
            }
            else{
                PixToComp.at<uint8_t>(i,j) = bgdGMM.whichComponent(static_cast<cv::Vec3d>(img.at<cv::Vec3b>(i,j)));
            }
        }
    }
}

void learnGMMs(cv::Mat const & img, cv::Mat const & mask, GMM & fgdGMM, GMM & bgdGMM, cv::Mat const & PixToComp){
    fgdGMM.deleteAllSamples();
    bgdGMM.deleteAllSamples();

    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int ki = PixToComp.at<uint8_t>(i, j);
            if(mask.at<uint8_t>(i,j)&1){
                fgdGMM.addSample(ki, static_cast<cv::Vec3d>(img.at<cv::Vec3b>(i,j)));
            }
            else{
                bgdGMM.addSample(ki, static_cast<cv::Vec3d>(img.at<cv::Vec3b>(i,j)));
            }
        }
    }

    fgdGMM.calcMeanAndCovWithSamples();
    bgdGMM.calcMeanAndCovWithSamples();
}

} // namespace GC

#endif


