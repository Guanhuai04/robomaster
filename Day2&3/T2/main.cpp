#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>
using namespace cv;

const int board_r = 6, board_c = 9;
const int board_n = board_r * board_c;
const Size board_size = {board_c, board_r};
const Size square_size = {10, 10};
const std::string path = "../chess/";
const int img_n = 41;

int main() {
    std::vector< std::vector< Point2f > > point_pix_pos;
    Size img_size;
    int successes = 0;

    for (int i = 0; i < img_n; i++) {
        Mat img = imread(path + std::to_string(i) + ".jpg");
        Mat res = img.clone();
        if (!i) {
            img_size.width = img.cols;
            img_size.height = img.rows;
        }
        std::vector< Point2f > buf;
        int found = findChessboardCorners(img, board_size, buf);
        if (found && buf.size() == board_n) {
            successes += 1;
            cvtColor(img, img, COLOR_BGR2GRAY);
            find4QuadCornerSubpix(img, buf, {5, 5});
            point_pix_pos.push_back(buf);
            drawChessboardCorners(res, board_size, buf, found);
//            imshow("corners", res);
//            waitKey(50);
        } else {
            std::cout << "failed to found all chess board corners in image : " + std::to_string(i) + ".jpg" << std::endl;
        }
    }
//    cv::waitKey(0);
//    cv::destroyAllWindows();
    std::cout << successes << " useful chess boards" << std::endl;

    std::vector< std::vector< Point3f > > point_grid_pos;
    std::vector< int > point_count;
    Mat camera_matrix(3, 3, CV_32FC1, Scalar::all(0));
    Mat dist_coeffs(1, 5, CV_32FC1, Scalar::all(0));
    std::vector< Mat > rvecs;
    std::vector< Mat > tvecs;
    for (int i = 0; i < successes; i++) {
        std::vector< Point3f > buf;
        for (int j = 0; j < board_r; j++) {
            for (int k = 0; k < board_c; k++){
                Point3f pt;
                pt.x = j * square_size.height;
                pt.y = k * square_size.width;
                pt.z = 0;
                buf.push_back(pt);
            }
        }
        point_grid_pos.push_back(buf);
        point_count.push_back(board_n);
    }

    std::cout << calibrateCamera( point_grid_pos, point_pix_pos,
                                  img_size, camera_matrix,
                                  dist_coeffs, rvecs, tvecs ) << std::endl;
    std::cout << camera_matrix << std::endl << dist_coeffs << std::endl;
    return 0;
}
