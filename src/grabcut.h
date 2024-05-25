#ifndef __grabcut__H__
#define __grabcut__H__

#include <opencv2/opencv.hpp>
#include <cmath>
#include <graph.h>
#include <random>
#include <ctime>
#include <memory>

namespace GC{

double const PI = std::acos(-1);
double constexpr EPS = 1e-9;
double const term1 = 1.0/std::pow((2.0*PI), 3.0/2.0); // 多元正态分布的一个因子

enum
{
    BGD    = 0,  // 肯定是背景
    FGD    = 1,  // 肯定是前景
    MAY_BGD = 2,  // 更可能是背景
    MAY_FGD = 3   // 更可能是前景
};


inline double vTMv33(double const * vt, double const * m, double const * v){
    return vt[0]*(v[0]*m[0]+v[1]*m[1]+v[2]*m[2])
           + vt[1]*(v[0]*m[3]+v[1]*m[4]+v[2]*m[5])
           + vt[2]*(v[0]*m[6]+v[1]*m[7]+v[2]*m[8]);
}

inline void vvT3AddTo(double * m_out, double const * v, double const * vT){
    m_out[0] += v[0]*vT[0]; m_out[1] += v[0] * vT[1]; m_out[2] += v[0] * vT[2];
    m_out[3] += v[1]*vT[0]; m_out[4] += v[1] * vT[1]; m_out[5] += v[1] * vT[2];
    m_out[6] += v[2]*vT[0]; m_out[7] += v[2] * vT[1]; m_out[8] += v[2] * vT[2];
}

inline void vvT3SubTo(double * m_out, double const * v, double const * vT){
    m_out[0] -= v[0]*vT[0]; m_out[1] -= v[0] * vT[1]; m_out[2] -= v[0] * vT[2];
    m_out[3] -= v[1]*vT[0]; m_out[4] -= v[1] * vT[1]; m_out[5] -= v[1] * vT[2];
    m_out[6] -= v[2]*vT[0]; m_out[7] -= v[2] * vT[1]; m_out[8] -= v[2] * vT[2];
}

inline double vTv3(double const * vt, double const * v){
    return vt[0]*v[0] + vt[1]*v[1] + vt[2]*v[2];
}

class GMM{
public:
    static int constexpr K = 5; // 论文中指出，高斯混合模型的K值取5为佳
private:

    double coefs[K];    // 高斯模型的混合参数
    cv::Vec3d mean[K];  // 均值，RGB分别的
    cv::Matx33d cov[K];   // 协方差

    double covDet[K];   // cov的行列式
    cv::Matx33d covInv[K];   // 协方差的逆矩阵
    double covDetSqrtInv[K]; // cov的行列式的根号的倒数

    size_t sampleCnt[K]; // 该分量的样本数
    size_t totalSampleCnt; // 样本总数

    cv::Vec3d colorSum[K]; // 用于计算均值
    cv::Matx33d colorProdSum[K]; // 用于计算协方差

public:
    GMM(){
        // 混合参数1位，RGB三通道的均值3位，其协方差矩阵3x3即9位
        // TODO: 初始化疑似可以优化
        cv::Vec3d zv(0.0, 0.0, 0.0);
        std::fill(coefs, coefs+K, 0.0);
        std::fill(mean, mean+K, zv);
        std::fill(cov, cov+K, cv::Matx33d::zeros());

        std::fill(covDet, covDet+K, 0.0);
        std::fill(covInv, covInv+K, cv::Matx33d::zeros());
        std::fill(covDetSqrtInv, covDetSqrtInv+K, 0.0);

        std::fill(sampleCnt, sampleCnt+K, 0);
        totalSampleCnt = 0;

        std::fill(colorSum, colorSum+K, zv);
        std::fill(colorProdSum, colorProdSum+K, cv::Matx33d::zeros());
    }

    double possibility(int ki, cv::Vec3d const & color) const {
        // 某个color属于第ki个分量的概率
        double ret=0.0;
        if(coefs[ki]<=0) return ret;

        cv::Vec3d delta = color-mean[ki];
        //double m = (delta.t() * covInv[ki] * delta)[0]; // 暴力用下标去算更优，见下一行
        double m = vTMv33(delta.val, covInv[ki].val, delta.val);
        double mult = -0.5 * m;

        ret = term1;
        ret *= covDetSqrtInv[ki];
        ret *= cv::exp(mult);

        return ret;
    }

    double possibility(cv::Vec3d const & color) const {
        // 所有分量概率的和
        double ret = 0.0;

        for(int ki=0;ki<K;ki++){
            ret += coefs[ki] * possibility(ki, color);
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
        covDetSqrtInv[ki] = 1.0/std::sqrt(covDet[ki]);
        cv::invert(cov[ki], covInv[ki]);
    }

    void addSample(int ki, cv::Vec3d const & color){
        // 向第ki个分量添加一个样本
        colorSum[ki] += color;
        //colorProdSum[ki] += color*color.t(); // // 暴力用下标去算更优，见下一行
        vvT3AddTo(colorProdSum[ki].val, color.val, color.val);
        sampleCnt[ki]++;
        totalSampleCnt++;
    }

    void deleteAllSamples(){
        // 删除所有分量中的样本信息
        for(int ki=0;ki<K;ki++){
            std::fill(colorSum[ki].val, colorSum[ki].val+3, 0.0);
            std::fill(colorProdSum[ki].val, colorProdSum[ki].val+9, 0.0);
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
            // cov[ki] = colorProdSum[ki]*(1.0/n) - mean[ki]*mean[ki].t(); // TODO: 可优化，暴力用val下标去算，可能可以减小构造开销
            cov[ki] = colorProdSum[ki]*(1.0/n);
            vvT3SubTo(cov[ki].val, mean[ki].val, mean[ki].val);
            double dt = cv::determinant(cov[ki]);
            if(dt<EPS){
                cov[ki].val[0] += 0.01;
                cov[ki].val[4] += 0.01;
                cov[ki].val[8] += 0.01;
            }

            calcCovInvAndDet(ki);
        }
    }
};

std::vector<uint8_t> kMeans(std::vector<cv::Vec3d> const & colors, int classNum=GMM::K, int iterCnt = 10){
    std::vector<uint8_t> labels(colors.size());
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
            std::fill(classColorSum[i].val, classColorSum[i].val+3, 0.0);
        }
    };

    while(iterCnt--){
        init();
        for(int i=0;i<sz;i++){
            double mindis = 1e15;
            int minc = 0;
            for(int j=0;j<classNum;j++){
                cv::Vec3d delta = colors[i] - classCenter[j];
                double dis = vTv3(delta.val, delta.val);
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
            // 原论文这个相邻的像素是8个
            // 按照我的理解，beta是在全图意义上的期望，而不是每个像素都有一个beta

            if(i>0){
                cnt++;
                cv::Vec3d delta = color - static_cast<cv::Vec3d>(img.at<cv::Vec3b>(i-1, j));
                beta += vTv3(delta.val, delta.val);
            }
            if(j>0){
                cnt++;
                cv::Vec3d delta = color - static_cast<cv::Vec3d>(img.at<cv::Vec3b>(i, j-1));
                beta += vTv3(delta.val, delta.val);
            }
            if(i>0 && j>0){
                cnt++;
                cv::Vec3d delta = color - static_cast<cv::Vec3d>(img.at<cv::Vec3b>(i-1, j-1));
                beta += vTv3(delta.val, delta.val);
            }
            if(i>0 && j<img.cols-1){
                cnt++;
                cv::Vec3d delta = color - static_cast<cv::Vec3d>(img.at<cv::Vec3b>(i-1, j+1));
                beta += vTv3(delta.val, delta.val);
            }
        }
    }

    if(beta<EPS) beta = 0.0;
    else beta = 1.0/(2.0*beta/cnt);
    // 总共选中像素点的次数

    return beta;
}

class GCGraph{
public:
    using GraphType = Graph<double, double, double>;

private:
    std::unique_ptr<GraphType> g;
    double kTerm; // graph cut的论文中，t weights的一种取值

    void addNLinks(cv::Mat const & img, cv::Mat const & mask, double beta, double gamma = 50.0){
        // 传入一个图、一张图片、一个mask，beta的计算见前，gamma在论文中默认给50
        // 计算平滑项，即图片上的节点的边权，似乎在graph cut的论文里叫n weights、n links
        g->add_node(img.rows*img.cols);

        auto maskEqu = [&mask](int i1, int j1, int i2, int j2){
            return true;
//            int mk1 = mask.at<int>(i1,j1), mk2 = mask.at<int>(i2,j2);
//            if(mk1>1||mk2>1) return true;
//            return mk1 != mk2;
            // 经过测试，原文中说的艾弗森括号[alpha_n != alpha_m]，无论以什么角度去理解，都不如直接全部返回true
            // 比较怀疑是因为这里是软segmentation，大家的alpha都不一样，所以干脆全部返回true
        };

        // 原文在grabcut部分不再有dis^-1这一项，但是opencv的实现有
        // 自己测试发现有没有不太影响
        auto calcWeight = [&img, &beta, &gamma](int i1, int j1, int i2, int j2){
            cv::Vec3d color = img.at<cv::Vec3b>(i1, j1);
            cv::Vec3d delta = color - static_cast<cv::Vec3d>(img.at<cv::Vec3b>(i2, j2));
//            if(std::abs(i1-i2)+std::abs(j1-j2)==2)
//                return gamma/std::sqrt(2)*cv::exp(-beta*delta.dot(delta)); // 即dis^-1
            return gamma*cv::exp(-beta*vTv3(delta.val, delta.val));
        };

        auto coordToIdx = [&img](int i, int j){
            return i*img.cols+j;
        };

        for(int i=0;i<img.rows;i++){
            for(int j=0;j<img.cols;j++){
                int idx1 = coordToIdx(i, j);

                double sum_weight=0.0;
                // 只需要对每个节点的左、左上、上、右上建边，就可以防止重复建边
                if(i>0 && maskEqu(i, j, i-1, j)){
                    int idx2 = coordToIdx(i-1, j);
                    double weight = calcWeight(i, j, i-1, j);
                    sum_weight+=weight;
                    g->add_edge(idx1, idx2, weight, weight);
                }
                if(j>0 && maskEqu(i, j, i, j-1)){
                    int idx2 = coordToIdx(i, j-1);
                    double weight = calcWeight(i, j, i, j-1);
                    sum_weight+=weight;
                    g->add_edge(idx1, idx2, weight, weight);
                }
                if(i>0 && j>0 && maskEqu(i, j, i-1, j-1)){
                    int idx2 = coordToIdx(i-1, j-1);
                    double weight = calcWeight(i, j, i-1, j-1);
                    sum_weight+=weight;
                    g->add_edge(idx1, idx2, weight, weight);
                }
                if(i>0 && j<img.cols-1 && maskEqu(i, j, i-1, j+1)){
                    int idx2 = coordToIdx(i-1, j+1);
                    double weight = calcWeight(i, j, i-1, j+1);
                    sum_weight+=weight;
                    g->add_edge(idx1, idx2, weight, weight);
                }
                kTerm = std::max(kTerm, sum_weight);
            }
        }
        kTerm += 1.0;
    }

    void addTLinks(cv::Mat const & img, cv::Mat const & mask, GMM const & fgdGMM, GMM const & bgdGMM){
        // 传入图像和mask
        // 添加数据项，即图上的像素到源点汇点的边权，似乎在graph cut的论文里叫t weights、t links

        auto coordToIdx = [&img](int i, int j){
            return i*img.cols+j;
        };

        for(int i=0;i<img.rows;i++){
            for(int j=0;j<img.cols;j++){
                double sw, tw; // 到源点的权值，到汇点的权值

                cv::Vec3d color = img.at<cv::Vec3b>(i, j);
                uint8_t mk = mask.at<uint8_t>(i, j);
                if(mk==MAY_FGD || mk==MAY_BGD){
                    sw = -std::log(bgdGMM.possibility(color)); // graph cut的论文指出，源点那边是前景
                    tw = -std::log(fgdGMM.possibility(color)); // 汇点那边是背景。
                    // 这里是惩罚项（正数），即分类错误的惩罚。我们需要使惩罚最小，能量最小
                    // grab cut的论文虽然说的是和k_n有关，但经过测试发现，没有k_n，而是所有概率加和的惩罚项明显更好
                    // 即，不使用-cv::log(bgdGMM.possibility(ki, color))
                }
//                else continue;
                else if(mk==FGD){
                    sw = kTerm; // 见graph cut论文，这一部分没有在grab cut中提到
                    tw = 0;     // 但是我测试过后，神奇地发现直接去掉也可以
                }
                else{
                    sw = 0;
                    tw = kTerm;
                }

                g->add_tweights(coordToIdx(i, j), sw, tw);
            }
        }
    }

public:
    GCGraph(int nodeNumMax_, int edgeNumMax_):  g(std::make_unique<GraphType>(nodeNumMax_, edgeNumMax_)),
                                                kTerm(0.0){}
    ~GCGraph(){
        g->reset();
    }

    void buildGraph(cv::Mat const & img, cv::Mat const & mask, GMM const & fgdGMM, GMM const & bgdGMM, double beta, double gamma = 50.0){
        g->reset();
        kTerm = 0.0;

        addNLinks(img, mask, beta, gamma);
        addTLinks(img, mask, fgdGMM, bgdGMM);
    }

    void estimateSegmentation(cv::Mat & mask){
        g->maxflow();
        auto coordToIdx = [&mask](int i, int j){
            return i*mask.cols+j;
        };

        for(int i=0;i<mask.rows;i++){
            for(int j=0;j<mask.cols;j++){
                uint8_t mk = mask.at<uint8_t>(i, j);
                if(mk==MAY_FGD || mk==MAY_BGD){
                    if(g->what_segment(coordToIdx(i,j))==GraphType::SOURCE){
                        mask.at<uint8_t>(i, j) = MAY_FGD;
                    }
                    else{
                        mask.at<uint8_t>(i, j) = MAY_BGD;
                    }
                }
            }
        }
    }

};

void initMask(cv::Mat& mask, cv::Size imgSz, cv::Rect rect){
    // 通过一个矩形来初始化Mask，和论文中描述的一致，框内可能是前景，框外一定是背景
    mask.create(imgSz, CV_8UC1);
    mask.setTo(cv::Scalar(BGD));

    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, imgSz.width-rect.x);
    rect.height = std::min(rect.height, imgSz.height-rect.y);

    (mask(rect)).setTo(cv::Scalar(MAY_FGD));
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

void grabCut(cv::Mat const & img, cv::Mat & mask, int iterCnt = 20){
    GMM fgdGMM, bgdGMM;
    GCGraph gcg(img.rows*img.cols, img.rows*img.cols*8);
    cv::Mat PixToComp;
    PixToComp.create(img.size(), CV_8UC1);
    assert(img.type()==CV_8UC3);
    assert(mask.type()==CV_8UC1);

    double const beta = calcBeta(img);

    initGMMs(img, mask, fgdGMM, bgdGMM);

    while(iterCnt--){
        std::cout<<"left iter cnt: "<<iterCnt<<std::endl;
        assignGMMsToPixels(img, mask, fgdGMM, bgdGMM, PixToComp);
        learnGMMs(img, mask, fgdGMM, bgdGMM, PixToComp);
        gcg.buildGraph(img, mask, fgdGMM, bgdGMM, beta);
        gcg.estimateSegmentation(mask);
    }
}

} // namespace GC

#endif


