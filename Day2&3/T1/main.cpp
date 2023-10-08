#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <Eigen/Dense>
using namespace Eigen;

const auto F((Matrix<double, 3, 4>() <<
        400., 0., 190., 0.,
        0., 400., 160., 0.,
        0., 0., 1., 0.
             ).finished());

const Quaternion Q(-0.5, 0.5, 0.5, -0.5);
const Vector3d T(2, 2, 2);

cv::Mat img(1000, 1500, CV_8UC3);

int main() {
    freopen("../points.txt", "r", stdin);
    int n; std::cin >> n;
    Matrix4Xd coord(4, n);
    for (int i = 0; i < n; i++) {
        std::cin >> coord(0, i) >> coord(1, i) >> coord(2, i);
        coord(3, i) = 1;
    }
    Matrix4d QT = Matrix4d::Zero();
    QT.block(0, 0, 3, 3) = Q.toRotationMatrix().transpose();
    QT.block(0, 3, 3, 1) = -Q.toRotationMatrix().transpose() * T;
    QT(3, 3) = 1;
    Matrix3Xd proj_coord = F * QT * coord;

    for (int i = 0;i < n; i++) {
        Vector3d p = proj_coord.col(i);
        cv::circle(img, cv::Point(p[0] / p[2], p[1] / p[2]), 1, {255, 255, 255}, 1);
    }
    cv::imshow("img", img);
    cv::imwrite("../res.jpg", img);
    cv::waitKey(0);
    return 0;
}
