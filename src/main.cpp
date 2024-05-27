#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "grabcut.h"
#include <string>
#include <chrono>

int main(int argc, char* argv[]) {
    if(argc!=2){
        std::cout<<"usage: GrabCut.exe image_filename";
        return 0;
    }

    std::string filename = argv[1];
    cv::Mat img = cv::imread(filename, 1);
    if(img.empty()){
        std::cout<<"img invalid";
        return 0;
    }

    cv::Size sz;
    sz.height = img.rows/2;
    sz.width = img.cols/2;
    cv::resize(img, img, sz);

    GCGUI::gui_main(img);

    return 0;
}
