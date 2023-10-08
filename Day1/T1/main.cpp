#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

class TrackerData {
public:
    int thres, ratio_green, size_open, size_close, size_blur, ratio_blue;
    TrackerData(int thres_, int ratio1, int ratio2, int size1, int size2, int size3) {
        thres = thres_; ratio_green = ratio_green;  ratio_blue = ratio2;
        size_open = size1; size_close = size2; size_blur = size3;
    }
};

void Onchange_thres(int value, void* data) {
    TrackerData &user_data = *(TrackerData*)data;
    user_data.thres = value;
}

void Onchange_ratio_green(int value, void* data) {
    TrackerData &user_data = *(TrackerData*)data;
    user_data.ratio_green = value;
}

void Onchange_ratio_blue(int value, void* data) {
    TrackerData &user_data = *(TrackerData*)data;
    user_data.ratio_blue = value;
}

void Onchange_size_open(int value, void* data) {
    TrackerData &user_data = *(TrackerData*)data;
    user_data.size_open = value;
}

void Onchange_size_close(int value, void* data) {
    TrackerData &user_data = *(TrackerData*)data;
    user_data.size_close = value;
}

void Onchange_size_blur(int value, void* data) {
    TrackerData &user_data = *(TrackerData*)data;
    if (value % 2 == 0) value += 1;
    user_data.size_blur = value;
}

int main() {
    cv::Mat apple = cv::imread("../apple.png");
    assert(apple.channels() == 3);
    cv::Mat result;
    int thres = 110, ratio_green = 958, ratio_blue = 565, size_open = 17, size_close = 3, size_blur = 7;
    TrackerData data = TrackerData(thres, ratio_green, ratio_blue, size_open, size_close, size_blur);
    cv::namedWindow("result");
    cv::createTrackbar("thres", "result", &thres, 200, Onchange_thres, &data);
    cv::createTrackbar("ratio_green", "result", &ratio_green, 2000, Onchange_ratio_green, &data);
    cv::createTrackbar("ratio_blue", "result", &ratio_blue, 1000, Onchange_ratio_blue, &data);
    cv::createTrackbar("size_open", "result", &size_open, 50, Onchange_size_open, &data);
    cv::createTrackbar("size_close", "result", &size_close, 50, Onchange_size_close, &data);
    cv::createTrackbar("size_blur", "result", &size_blur, 50, Onchange_size_blur, &data);

    while(true) {
        cv::Mat channels[3];
        cv::split(apple, channels);
        channels[2] -= channels[1] * 1.0 * data.ratio_green / 1000 + 1.0 * channels[0] * data.ratio_blue / 1000;
        cv::threshold(channels[2],  result, 1.0 * data.thres / 10, 255, cv::THRESH_BINARY);
        cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, {data.size_open, data.size_open});
        cv::morphologyEx(result, result, cv::MORPH_OPEN, kernel_open);
        cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, {data.size_close, data.size_close});
        cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel_close);
        cv::medianBlur(result, result, data.size_blur);
//        cv::imshow("result", result);
//        cv::waitKey(0);

        std::vector< std::vector<cv::Point> > contours;
        std::vector< cv::Vec4i > hierachy;
        cv::Mat contour_result = apple.clone();
        cv::findContours(result, contours, hierachy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        int max_contour_id = std::max_element(contours.begin(), contours.end(),
                                              [](const std::vector<cv::Point> &c1, const std::vector<cv::Point> &c2) {
            return cv::contourArea(c1) < cv::contourArea(c2);
        }) - contours.begin();
        cv::drawContours(contour_result, contours, max_contour_id, {220, 220, 220}, 1);
        cv::rectangle(contour_result, cv::boundingRect(contours[max_contour_id]),{220, 220, 220}, 2);
        cv::imshow("contour", contour_result);
        cv::waitKey(0);
    }
    return 0;
}
