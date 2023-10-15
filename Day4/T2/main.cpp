#include <iostream>
#include <Eigen/Dense>
#include <ceres/jet.h>
#include "matplotlibcpp.h"
using namespace Eigen;
using namespace ceres;
namespace plt = matplotlibcpp;

// Q : 过程噪声  R : 测量噪声

template<typename T, int N_X, int N_Z>
class EKF {
    using MatrixXX = Matrix<T, N_X, N_X>;
    using MatrixZX = Matrix<T, N_Z, N_X>;
    using MatrixXZ = Matrix<T, N_X, N_Z>;
    using MatrixZZ = Matrix<T, N_Z, N_Z>;
    using VectorX = Matrix<T, N_X, 1>;
    using VectorZ = Matrix<T, N_Z, 1>;

public:
    VectorX X, X_prior;
    MatrixXX A, P, Q;
    MatrixZZ R;
    MatrixXZ K;
    MatrixZX H;
    VectorZ Z_prior;

    explicit EKF(const VectorX &X0 = VectorX::Zero()) :
            X(X0), P(MatrixXX::Identity()), Q(MatrixXX::Identity()), R(MatrixZZ::Identity()) {}

    template<class Func>
    VectorX predict(Func &&func) {
        Jet<T, N_X> X_auto_jet[N_X];
        for (int i = 0; i < N_X; i++) {
            X_auto_jet[i].a = X[i];
            X_auto_jet[i].v[i] = 1;
        }
        Jet<T, N_X> X_prior_auto_jet[N_X];
        func(X_auto_jet, X_prior_auto_jet);
        for (int i = 0; i < N_X; i++) {
            X_prior[i] = X_prior_auto_jet[i].a;
            A.block(i, 0, 1, N_X) = X_prior_auto_jet[i].v.transpose();
        }
        P = A * P * A.transpose() + Q;
        return X_prior;
    }

    template<class Func>
    VectorX update(Func &&func, const VectorZ &Z) {
        Jet<T, N_X> X_prior_auto_jet[N_X];
        for (int i = 0; i < N_X; i++) {
            X_prior_auto_jet[i].a = X_prior[i];
            X_prior_auto_jet[i].v[i] = 1;
        }
        Jet<T, N_X> Z_prior_auto_jet[N_Z];
        func(X_prior_auto_jet, Z_prior_auto_jet);
        for (int i = 0; i < N_Z; i++) {
            Z_prior[i] = Z_prior_auto_jet[i].a;
            H.block(i, 0, 1, N_X) = Z_prior_auto_jet[i].v.transpose();
        }
        K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
        X = X_prior + K * (Z - Z_prior);
        P = (MatrixXX::Identity() - K * H) * P;
        return X;
    }
};

constexpr int Z_N = 1, X_N = 2;

struct Predict {
    double delta_t = 1;

    template<typename T>
    void operator () (const T x0[X_N], T x1[X_N]) {
        x1[0] = x0[0] + delta_t * x0[1];
        x1[1] = x0[1];
    }
} predict_;

struct Measure {
    template<typename T>
    void operator () (const T x0[X_N], T z0[Z_N]) {
        z0[0] = x0[0];
    }
} measure;

int main() {
    plt::figure_size(800, 500);
    freopen("../dollar.txt", "r", stdin);
    std::vector<double> vals;
    for (int i = 1; i <= 30; i++) {
        double x; std::cin >> x;
        vals.push_back(x);
    }
    EKF<double, X_N, Z_N> ekf(Vector2d{vals[0], 0});
    ekf.Q << 0.001, 0,
            0, 0.0001;
    ekf.R << 0.1;

    double predict_val;
    for (int i = 1; i < 30; i++) {
        Matrix<double, Z_N, 1> Z(vals[i]);
        ekf.update(measure, Z);
        predict_val = ekf.predict(predict_)[0];
    }
    std::vector<double> predicts = {vals[29]};
    for (int i = 1; i <= 10; i++) {
        std::cout << predict_val << std::endl;
        predicts.push_back(predict_val);
        Matrix<double, Z_N, 1> Z(predict_val);
        ekf.update(measure, Z);
        predict_val = ekf.predict(predict_)[0];
    }

    std::vector<int> x1, x2;
    for (int i = 1; i <= 30; i++) x1.push_back(i);
    for (int i = 30; i <= 40; i++) x2.push_back(i);
    plt::plot(x1, vals);
    plt::plot(x2, predicts, "r");
    plt::show();
    return 0;
}