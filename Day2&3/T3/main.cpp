#include <iostream>
#include <ctime>
#include <random>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>
using namespace Eigen;

const auto F((Matrix<double, 3, 4>() <<
        1., 0., 1., 0.,
        0., 1., 1., 0.,
        0., 0., 1., 0.
             ).finished());

const int img_w = 1300;
const int img_h = 800;
const Vector3d cam_pos(0, 0, 1);
const auto Base((Matrix<double, 3, 2>() <<
        1, 0,
        0, 1,
        0, 0
            ).finished());

Matrix3Xd convert(const Matrix3Xd &coords, const AngleAxisd &r_vec) {
    auto R = r_vec.toRotationMatrix();
    auto A = R * Base;
    Matrix3d P = A * (A.transpose() * A).inverse() * A.transpose();
    return R.inverse() * P * coords;
}

cv::Mat draw_img(const Matrix3Xd &proj_coords, const AngleAxisd &r_vec) {
    double min_x = 0x3f3f3f3f, max_x = 0, min_y = 0x3f3f3f3f, max_y = 0;
    cv::Mat img = cv::Mat::zeros(img_h, img_w, CV_8UC3);
    std::vector< cv::Point2f > points;
    for (int i = 0; i < proj_coords.cols(); i++) {
        Vector3d p3 = proj_coords.col(i);
        cv::Point2f p(p3[0] * img_w, p3[1] * img_h);
        points.push_back(p);
    }
    Matrix3Xd center_coord(3, 1);
    center_coord << img_w / 2, img_h / 2, 0;
    Matrix3Xd proj_center = convert(center_coord, r_vec);
    for (auto p : points) {
//        p.x += img_w / 2 - center_coord(0, 0);
//        p.y += img_h / 2 - center_coord(1, 0);
        p.x += img_w / 2;
        p.y += img_h / 2;
        if (p.x >= 0 && p.x <= img_w && p.y >= 0 && p.y <= img_h) {
            cv::circle(img, p, 1, {255, 255, 255}, 1);
        }
    }
    return img;
}

int main() {
    cv::VideoWriter writer("../output.avi",cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           60, cv::Size(img_w, img_h), true);
    freopen("../points_output.txt", "r", stdin);
    int n; std::cin >> n;
    Matrix3Xd coords(3, n);
    std::uniform_real_distribution< double > randu(-1, 1);
    std::default_random_engine e(114514);
    for (int i = 0; i < n; i++) {
        double x, y, z;
        std::cin >> x >> y;
        z = randu(e);
        coords.col(i) << x / img_w, y / img_h, z;
    }

    for (int i = 0; i <= 360; i++) {
        double theta = 150 * std::exp(-0.02 * i);
//        if (theta < 0.1) theta = std::exp(-0.08 * i);
        AngleAxisd r_vec(M_PI * theta / 180, Vector3d(1, 1, 0.5));
        Matrix3Xd proj_coords = convert(coords, r_vec);
        cv::Mat img = draw_img(proj_coords, r_vec);

//        cv::imshow("img", img);
//        cv::waitKey(0);
        writer << img;
    }
    writer.release();
    return 0;
}
