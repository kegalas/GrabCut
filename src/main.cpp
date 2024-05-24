#include <opencv2/opencv.hpp>
#include "grabcut.h"
#include <string>

int main() {
    //cv::Rect rect(251,141,276,316);
    cv::Rect rect(80, 9, 222, 220);
    std::string filename = "lena.jpg";
    cv::Mat img = cv::imread(filename, 1);
    cv::Size sz;
    sz.height = img.rows/2;
    sz.width = img.cols/2;
    cv::resize(img, img, sz);

    cv::Mat mask;

    GC::initMask(mask, img.size(), rect);
    GC::grabCut(img, mask, 10);

    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            uint8_t mk = mask.at<uint8_t>(i, j);
            if(mk&1){

            }
            else{
                img.at<cv::Vec3b>(i, j) = cv::Vec3b(0,0,0);
            }
        }
    }

    cv::imwrite("result.jpg", img);

    return 0;
}
