#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>

class TrackerData {
public:
    int hmin, hmax, smin, smax, vmin, vmax;
    TrackerData(int v1, int v2, int v3, int v4, int v5, int v6) {
        hmin = v1; hmax = v2;
        smin = v3; smax = v4;
        vmin = v5; vmax = v6;
    }
};

void Onchange_hmin(int value, void* data) {
    TrackerData &user_data = *(TrackerData*)data;
    user_data.hmin = value;
}

void Onchange_hmax(int value, void* data) {
    TrackerData &user_data = *(TrackerData*)data;
    user_data.hmax = value;
}

void Onchange_smin(int value, void* data) {
    TrackerData &user_data = *(TrackerData*)data;
    user_data.smin = value;
}

void Onchange_smax(int value, void* data) {
    TrackerData &user_data = *(TrackerData*)data;
    user_data.smax = value;
}

void Onchange_vmin(int value, void* data) {
    TrackerData &user_data = *(TrackerData*)data;
    user_data.vmin = value;
}

void Onchange_vmax(int value, void* data) {
    TrackerData &user_data = *(TrackerData*)data;
    user_data.vmax = value;
}

void Solve(int id) {
    std::string path = "../plates/00" + std::to_string(id) + ".jpg";
    cv::Mat img = cv::imread(path);
    cv::Mat hsv, result;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    int v1 = 90, v2 = 124, v3 = 140, v4 = 255, v5 = 100, v6 = 255;
    TrackerData data = TrackerData(v1, v2, v3, v4, v5, v6);
//    cv::namedWindow("trackbar");
//    cv::createTrackbar("hmin", "trackbar", &v1, 255, Onchange_hmin, &data);
//    cv::createTrackbar("hmax", "trackbar", &v2, 255, Onchange_hmax, &data);
//    cv::createTrackbar("smin", "trackbar", &v3, 255, Onchange_smin, &data);
//    cv::createTrackbar("smax", "trackbar", &v4, 255, Onchange_smax, &data);
//    cv::createTrackbar("vmin", "trackbar", &v5, 255, Onchange_vmin, &data);
//    cv::createTrackbar("vmax", "trackbar", &v6, 255, Onchange_vmax, &data);
//    while (cv::waitKey(1) != 13) {
//        cv::inRange(hsv, cv::Scalar(data.hmin, data.smin, data.vmin),
//                    cv::Scalar(data.hmax, data.smax, data.vmax), result);
//        cv::imshow("result", result);
//    }
    cv::inRange(hsv, cv::Scalar(data.hmin, data.smin, data.vmin),
                cv::Scalar(data.hmax, data.smax, data.vmax), result);
    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_RECT, {5, 5});
    cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel_open);
    std::vector< std::vector<cv::Point> > contours;
    std::vector< cv::Vec4i > hierachy;
    cv::findContours(result, contours, hierachy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::Mat contour_result = img.clone();
    for (int i = 0; i + 1; i = hierachy[i][0]) {
        cv::Rect rect = cv::boundingRect(contours[i]);
        if (rect.area() >= 2000 && rect.width > rect.height) {
            std::vector<cv::Point> poly_contour;
            double limit = (id == 5 ? 0.076 : 0.073) * cv::arcLength(contours[i], true);
            cv::approxPolyDP(contours[i], poly_contour,limit, true);
            std::cout << poly_contour << std::endl;
            cv::polylines(contour_result, poly_contour,
                          true, {20, 220, 220}, 2);
//            cv::rectangle(contour_result, cv::boundingRect(contours[i]), {20, 220, 220}, 2);
        }
//        cv.approxPolyDP(contours[0], 0.01 * cv.arcLength(contours[0], True), True)
    }
    cv::imshow("contours", contour_result);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main() {
    for (int i = 1; i <= 5; i++) {
        Solve(i);
    }
    return 0;
}
