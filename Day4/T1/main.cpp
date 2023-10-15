#include <iostream>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "hw4_t1/big_armor_scale.hpp"

using namespace Eigen;

cv::Mat cameraMatrix, distCoeffs;
const std::vector<cv::Point2f> points = {{575.508, 282.175},
                                         {573.93, 331.819},
                                         {764.518, 337.652},
                                         {765.729, 286.741}};

int main() {
    cv::FileStorage reader("../hw4_t1/f_mat_and_c_mat.yml",cv::FileStorage::READ);
    reader["F"] >> cameraMatrix;
    reader["C"] >> distCoeffs;

    cv::Mat rvec, tvec, rotation_matrix;
    cv::solvePnP(PW_BIG,points,cameraMatrix,distCoeffs, rvec, tvec);
    cv::Rodrigues(rvec, rotation_matrix);
    MatrixXd T;
    cv::cv2eigen(tvec, T);

    Quaternion q(-0.0816168, 0.994363, -0.0676645, -0.00122528);
    std::cout << q.toRotationMatrix() * T << std::endl;
    return 0;
}
