#include "kalman_filter.hpp"

#include <Eigen/Dense>
#include <stdexcept>  // std::invalid_argument
#include <utility>    // std::move (optional)

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

KalmanFilter::KalmanFilter(const Vec& initial_state,
                           const Mat& initial_covariance,
                           const Mat& transition_matrix,
                           const Mat& observation_matrix,
                           const Mat& process_covariance,
                           const Mat& measurement_covariance)
{
    // Infer dimensions from inputs
    const int n = static_cast<int>(initial_state.size());
    if (initial_covariance.rows() != n || initial_covariance.cols() != n) {
        throw std::invalid_argument("initial_covariance must be n x n (square).");
    }

    // A: n x n
    if (transition_matrix.rows() != n || transition_matrix.cols() != n) {
        throw std::invalid_argument("transition_matrix A must be n x n (square).");
    }

    // H: m x n  -> m from H.rows()
    const int m = observation_matrix.rows();
    if (observation_matrix.cols() != n) {
        throw std::invalid_argument("observation_matrix H must be m x n (cols == state dim).");
    }

    // Q: n x n
    if (process_covariance.rows() != n || process_covariance.cols() != n) {
        throw std::invalid_argument("process_covariance Q must be n x n (square).");
    }

    // R: m x m
    if (measurement_covariance.rows() != m || measurement_covariance.cols() != m) {
        throw std::invalid_argument("measurement_covariance R must be m x m (square).");
    }

    x_ = initial_state;
    P_ = initial_covariance;
    A_ = transition_matrix;
    H_ = observation_matrix;
    Q_ = process_covariance;
    R_ = measurement_covariance;
}

void KalmanFilter::predict()
{
    // Predict state: x_k|k-1 = A * x_(k-1|k-1)
    x_ = A_ * x_;

    // Predict covariance: P_k|k-1 = A * P_(k-1|k-1) * A^T + Q
    P_ = A_ * P_ * A_.transpose() + Q_;
}

void KalmanFilter::update(const Vec& measurement)
{
    if (measurement.size() != H_.rows()) {
        throw std::invalid_argument("Measurement size must match observation matrix H.");
    }

    // Innovation
    Vec y = measurement - H_ * x_;

    // Innovation covariance
    Mat S = H_ * P_ * H_.transpose() + R_;
    Eigen::LLT<Mat> llt(S);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("LLT decomposition failed, S may not be positive definite.");
    }

    // Kalman gain
    Mat PHt = P_ * H_.transpose();
    Mat K = llt.solve(PHt.transpose()).transpose();

    // Update state
    x_ += K * y;

    // Update covariance (Joseph form)
    Mat I = Mat::Identity(P_.rows(), P_.cols());
    P_ = (I - K * H_) * P_ * (I - K * H_).transpose() + K * R_ * K.transpose();
}

void KalmanFilter::step(const std::optional<Vec>& measurement)
{
    // Predict the next state
    predict();

    // If a measurement is provided, update the state
    // Otherwise, we just keep the predicted state
    if (measurement.has_value()) {
        update(measurement.value());
    }
}

std::vector<Vec> KalmanFilter::filter(const std::vector<std::optional<Vec>>& measurements)
{
    std::vector<Vec> states;
    states.reserve(measurements.size());
    for (const auto& measurement : measurements) {
        step(measurement);
        states.push_back(x_);
    }
    return states;
}
