#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "grabcut.h"
#include <string>
#include <chrono>

int main() {
//    cv::Rect rect(80, 9, 222, 220);
//    cv::Rect rect(251,141,276,316);
//    cv::Rect rect(69, 1, 682, 462);
    std::string filename = "lena3.jpg";
    cv::Mat img = cv::imread(filename, 1);

    cv::Size sz;
    sz.height = img.rows/2;
    sz.width = img.cols/2;
    cv::resize(img, img, sz);

    GCGUI::gui_main(img);


//    cv::Mat mask;

//    GC::initMask(mask, img.size(), rect);

//    auto start = std::chrono::high_resolution_clock::now();
//    GC::grabCut(img, mask, 100);
//    auto end = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//    std::cout << duration << "ms" << std::endl;

//    for(int i=0;i<img.rows;i++){
//        for(int j=0;j<img.cols;j++){
//            uint8_t mk = mask.at<uint8_t>(i, j);
//            if(mk&1){

//            }
//            else{
//                img.at<cv::Vec3b>(i, j) = cv::Vec3b(0,0,0);
//            }
//        }
//    }

//    cv::imwrite("result.jpg", img);

//    cv::Mat fgd, bgd;
//    start = std::chrono::high_resolution_clock::now();
//    cv::grabCut(img, mask, rect, bgd, fgd, 10, cv::GC_INIT_WITH_RECT);
//    end = std::chrono::high_resolution_clock::now();
//    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//    std::cout << duration << "ms" << std::endl;

//    for(int i=0;i<img.rows;i++){
//        for(int j=0;j<img.cols;j++){
//            uint8_t mk = mask.at<uint8_t>(i, j);
//            if(mk&1){

//            }
//            else{
//                img.at<cv::Vec3b>(i, j) = cv::Vec3b(0,0,0);
//            }
//        }
//    }

//    cv::imwrite("result2.jpg", img);

    return 0;
}
