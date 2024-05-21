#include <opencv2/opencv.hpp>

int main() {
    cv::Mat m = cv::Mat::ones(3, 3, CV_64FC1);
    cv::Mat v = cv::Mat::ones(3, 1, CV_64FC1);
    std::cout<<m<<"\n";
    std::cout<<v<<"\n";
    std::cout<<v.t()*m*v<<"\n";
    return 0;
}
