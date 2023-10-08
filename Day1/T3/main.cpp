#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat template_three = cv::imread("../Template/3.jpg");
const int template_width = template_three.size[0];
const int template_height = template_three.size[1];

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

int get_area(const cv::RotatedRect &rect) {
    return rect.size.height * rect.size.width;
}

void draw_rect(cv::Mat &img, const cv::RotatedRect &rect) {
    cv::Point2f p[4];
    rect.points(p);
    for (int i = 0; i < 4; i++) {
        cv::line(img, p[i], p[(i + 1) % 4], {220, 20, 20}, 2);
    }
}

bool check_rect(const cv::RotatedRect &rect) {
    cv::Rect bound_rect = rect.boundingRect();
    return bound_rect.height > 1.4 * bound_rect.width;
}

void open_binary(cv::Mat &img, int x, int y) {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {x, y});
    cv::morphologyEx(img, img, cv::MORPH_OPEN, kernel);
}

void close_binary(cv::Mat &img, int x, int y) {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {x, y});
    cv::morphologyEx(img, img, cv::MORPH_CLOSE, kernel);
}

double calc_match(const cv::Mat &temp, const cv::Mat &img) {
    assert(temp.size == img.size);
    assert(temp.channels() == img.channels());
    assert(temp.type() == img.type());
    int r = temp.size[0], c = temp.size[1];
    double res = 0;
    for(int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            if (temp.at<int>(i, j) == img.at<int>(i, j))
                res += 1;
        }
    return res / (r * c);
}

void Pre_detect_armor_id(cv::Mat &img, std::vector<cv::Rect> &id_rects) {
    cv::Mat hsv, result;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    int v1 = 0, v2 = 75, v3 = 0, v4 = 88, v5 = 71, v6 = 110;
    TrackerData data = TrackerData(v1, v2, v3, v4, v5, v6);
    cv::inRange(hsv, cv::Scalar(data.hmin, data.smin, data.vmin),
                cv::Scalar(data.hmax, data.smax, data.vmax), result);
    close_binary(result, 5, 5);
    std::vector< std::vector<cv::Point> > contours;
    std::vector< cv::Vec4i > hierachy;
    cv::findContours(result, contours, hierachy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::Mat contour_result = img.clone();
    for (int i = 0; i + 1; i = hierachy[i][0]) {
        if (cv::contourArea(contours[i]) < 200) continue;
        cv::Mat contour_img = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8UC3);
        cv::drawContours(contour_img, contours, i, {255, 255, 255}, cv::FILLED);
        cv::Rect rect = cv::boundingRect(contours[i]);
        int width = rect.size().width, height = rect.size().height;
        cv::Mat roi(contour_img, cv::Rect(cv::Point(rect.x, rect.y), cv::Point(rect.x + width, rect.y + height)));
        cv::Mat temp = template_three.clone();
        cv::resize(temp, temp, {width, height});
        cv::threshold(temp, temp, 100, 255, cv::INTER_LINEAR);
        double matching_degree = calc_match(temp, roi);
        if (matching_degree > 0.5) {
            id_rects.push_back(rect);
        }
    }
}

cv::Point2f get_center(const cv::Rect &rect) {
    return cv::Point2f(rect.x + 1.0 * rect.size().width / 2, rect.y + 1.0 * rect.size().height / 2);
}

void Process_img(cv::Mat &img) {
    cv::Mat hsv, result;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    int v1 = 7, v2 = 35, v3 = 65, v4 = 255, v5 = 109, v6 = 255;
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
//        close_binary(result, 5, 5);
//        cv::imshow("img", img);
//        cv::imshow("result", result);
//    }
//    cv::destroyAllWindows();
//    return;
    cv::inRange(hsv, cv::Scalar(data.hmin, data.smin, data.vmin),
                cv::Scalar(data.hmax, data.smax, data.vmax), result);
    close_binary(result, 5, 5);
    std::vector< std::vector<cv::Point> > contours;
    std::vector< cv::Vec4i > hierachy;
    cv::findContours(result, contours, hierachy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::Mat contour_result = img.clone();
    std::vector<cv::RotatedRect> rects;
    for (int i = 0; i + 1; i = hierachy[i][0]) {
        cv::RotatedRect rect = cv::minAreaRect(contours[i]);
        if (check_rect(rect) && get_area(rect) > 200) {
            rects.push_back(rect);
        }
    }
    std::vector<cv::Rect> id_rects;
    Pre_detect_armor_id(img, id_rects);
    std::sort(rects.begin(), rects.end(), [](const cv::RotatedRect &r1, const cv::RotatedRect &r2) {
        return r1.center.x < r2.center.x;
    });
    for (int i = 0; i < (int)rects.size() - 1; i++) {
        if (cv::norm((rects[i].center - rects[i + 1].center)) < 90) {
            int id = -1;
            for (int j = 0; j < (int)id_rects.size(); j++) {
                if (id_rects[j].contains((rects[i].center + rects[i + 1].center) / 2.0)) {
                    id = j;
                    break;
                }
            }
            if (id != -1) {
                draw_rect(img, rects[i]);
                draw_rect(img, rects[i + 1]);
                cv::rectangle(img, id_rects[id], {20, 220, 220}, 2);
                i += 1;
            }
        }
    }
}

void print_img(const cv::Mat &img) {
    int r = img.size[0], c = img.size[1];
    for(int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            std::cout << img.at<int>(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    cv::threshold(template_three, template_three, 100, 255, cv::INTER_LINEAR);
    cv::VideoCapture cap("../armor.mp4");
    cv::Mat img;
    cv::VideoWriter writer("../armor_detect.avi",
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           30, {1920, 1080}, true);
    while (cap.read(img)) {
        assert(img.channels() == 3);
        cv::Mat result = img.clone();
        Process_img(result);
//        cv::imshow("result", result);
//        cv::waitKey(0);
        writer << result;
    }
    writer.release();
    return 0;
}
