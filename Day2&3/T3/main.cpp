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

//const Quaternion Q(-0.5, 0.5, 0.5, -0.5);
//const Vector3d T(2, 2, 2);

const int img_w = 1300;
const int img_h = 800;


Quaternion<double> euler_to_quater(const Vector3d euler) {
    return Eigen::AngleAxisd(euler[0] * M_PI / 180, Eigen::Vector3d::UnitZ()) *
           Eigen::AngleAxisd(euler[1] * M_PI / 180, Eigen::Vector3d::UnitY()) *
           Eigen::AngleAxisd(euler[2] * M_PI / 180, Eigen::Vector3d::UnitX());
}

int main() {
    freopen("../points.txt", "r", stdin);
    int n; std::cin >> n;
    Matrix4Xd coord(4, n);
    for (int i = 0; i < n; i++) {
        std::cin >> coord(0, i) >> coord(1, i) >> coord(2, i);
        coord(3, i) = 1;
    }
    Vector3d start_pos(5, -4, 6);
    Vector3d end_pos(2, 2, 2);
    Vector3d start_euler(45, -30, -60); // 90 0 -90
    Vector3d end_euler(0, 0, 0);
    Quaternion origin_posture(-0.5, 0.5, 0.5, -0.5);
    int steps = 300;

    cv::VideoWriter writer("../output.mp4",cv::VideoWriter::fourcc('H','2','6','4'),
                           60, cv::Size(img_w, img_h));
    for (int i = 0; i <= steps; i++) {
        double ratio = 1 - std::exp(-0.0001 * i * i); // i / steps
        cv::Mat img = cv::Mat::zeros(img_h, img_w, CV_8UC3);
        auto Q = euler_to_quater(start_euler + (end_euler - start_euler) * ratio) * origin_posture;
        std::cout << Q << std::endl;
        auto T = start_pos + (end_pos - start_pos) * ratio;
        Matrix4d QT = Matrix4d::Zero();
        QT.block(0, 0, 3, 3) = Q.toRotationMatrix().transpose();
        QT.block(0, 3, 3, 1) = -Q.toRotationMatrix().transpose() * T;
        QT(3, 3) = 1;
        Matrix3Xd proj_coord = F * QT * coord;

        for (int i = 0; i < n; i++) {
            Vector3d p3 = proj_coord.col(i);
            cv::Point2f p(p3[0] / p3[2], p3[1] / p3[2]);
            if (p.x >= 0 && p.x <= img_w && p.y >= 0 && p.y <= img_h) {
                cv::circle(img, p, 1, {255, 255, 255}, 1);
            }
        }
//        cv::imshow("img", img);
//        cv::waitKey(20);
//        cv::destroyAllWindows();
        writer << img;
    }
    writer.release();
    return 0;
}
