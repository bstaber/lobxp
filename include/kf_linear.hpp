#pragma once

#include <Eigen/Dense>
#include <optional>

/**
 * @brief Generic linear Kalman filter implementation.
 *
 * Supports optional control input by setting the control matrix B.
 * State-space model:
 *   x_k = A x_{k-1} + B u_k + w_{k-1},   w ~ N(0, Q)
 *   z_k = H x_k       + v_k,             v ~ N(0, R)
 * 
 * Here, we take B = 0 for simplicity.
 */
class KFLinear {
public:
    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;

    /**
     * @brief Initialize state vector and covariance.
    */
    KFLinear(const Vec& initial_state, 
                 const Mat& initial_covariance,
                 const Mat& transition_matrix,
                 const Mat& observation_matrix,
                 const Mat& process_covariance,
                 const Mat& measurement_covariance);

    /**
     * @brief Predict state without control input.
    */
    void predict();

    /**
     * @brief Update state estimate with a new measurement.
    */
    void update(const Vec& measurement);

    /**
     * @brief One-time step of the Kalman filter.
    */
    void step(const std::optional<Vec>& measurement);

    /**
     * @brief Filter a sequence of measurements.
    */
    std::vector<Vec> filter(const std::vector<std::optional<Vec>>& measurements);

    /// Accessors
    [[nodiscard]] const Vec& state()      const { return x_; }
    [[nodiscard]] const Mat& covariance() const { return P_; }

private:
    // === Model matrices ===
    Mat A_; // NOLINT(readability-identifier-naming) State transition
    Mat H_; // NOLINT(readability-identifier-naming) Observation
    Mat Q_; // NOLINT(readability-identifier-naming) Process noise covariance
    Mat R_; // NOLINT(readability-identifier-naming) Measurement noise covariance
    Mat B_; // NOLINT(readability-identifier-naming) Control matrix (optional)

    // === State variables ===
    Vec x_; // NOLINT(readability-identifier-naming) Current state
    Mat P_; // NOLINT(readability-identifier-naming) State covariance
};
