// tests/test_kalman_filter.cpp
#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <optional>
#include <vector>

#include "kf_linear.hpp"

#include <catch2/catch_approx.hpp>
using Catch::Approx;

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

TEST_CASE("Constructor validates dimensions")
{
    const int n = 2, m = 1;
    Vec x0(n); x0 << 0.0, 1.0;
    Mat P0 = Mat::Identity(n, n);
    Mat A  = Mat::Identity(n, n);
    Mat H(m, n); H << 1.0, 0.0;
    Mat Q  = 0.01 * Mat::Identity(n, n);
    Mat R  = Mat::Identity(m, m);

    SECTION("Valid dims do not throw") {
        REQUIRE_NOTHROW(KFLinear(x0, P0, A, H, Q, R));
    }

    SECTION("Bad P0 shape throws") {
        Mat badP = Mat::Identity(n+1, n+1);
        REQUIRE_THROWS_AS(KFLinear(x0, badP, A, H, Q, R), std::invalid_argument);
    }

    SECTION("Bad A shape throws") {
        Mat badA = Mat::Identity(n+1, n+1);
        REQUIRE_THROWS_AS(KFLinear(x0, P0, badA, H, Q, R), std::invalid_argument);
    }

    SECTION("Bad H shape throws") {
        Mat badH = Mat::Ones(m, n+1);
        REQUIRE_THROWS_AS(KFLinear(x0, P0, A, badH, Q, R), std::invalid_argument);
    }

    SECTION("Bad Q shape throws") {
        Mat badQ = Mat::Identity(n+1, n+1);
        REQUIRE_THROWS_AS(KFLinear(x0, P0, A, H, badQ, R), std::invalid_argument);
    }

    SECTION("Bad R shape throws") {
        Mat badR = Mat::Identity(m+1, m+1);
        REQUIRE_THROWS_AS(KFLinear(x0, P0, A, H, Q, badR), std::invalid_argument);
    }
}

TEST_CASE("Predict-only step leaves state at A*x and P -> A P A^T + Q")
{
    const int n = 2, m = 1;
    Vec x0(n); x0 << 2.0, -1.0;
    Mat P0 = (Mat(2,2) << 1.0, 0.2, 0.2, 2.0).finished();

    Mat A  = (Mat(2,2) << 1.0, 1.0,
                           0.0, 1.0).finished(); // constant-velocity model
    Mat H(m, n); H << 1.0, 0.0;
    Mat Q  = 0.1 * Mat::Identity(n, n);
    Mat R  = Mat::Identity(m, m);

    KFLinear kf(x0, P0, A, H, Q, R);

    kf.step(std::nullopt); // no measurement

    Vec x_expected = A * x0;
    Mat P_expected = A * P0 * A.transpose() + Q;

    REQUIRE(kf.state().isApprox(x_expected, 1e-12));
    REQUIRE(kf.covariance().isApprox(P_expected, 1e-12));
}

TEST_CASE("Scalar KF update matches closed-form")
{
    // 1D model: x_k = x_{k-1} + w, z_k = x_k + v
    const int n = 1, m = 1;
    Vec x0(n); x0 << 0.0;
    Mat P0(n,n); P0 << 1.0;

    Mat A(n,n); A << 1.0;
    Mat H(m,n); H << 1.0;
    Mat Q(n,n); Q << 0.25;    // process noise var
    Mat R(m,m); R << 1.0;     // measurement noise var

    KFLinear kf(x0, P0, A, H, Q, R);

    // Step 1: no measurement
    kf.step(std::nullopt);
    // Predict-only expectations
    double Pp = 1.0 + 0.25;     // P' = P0 + Q
    double xp = 0.0;            // x' = x0
    REQUIRE(kf.state()(0) == Approx(xp));
    REQUIRE(kf.covariance()(0,0) == Approx(Pp));

    // Step 2: with measurement z = 1.2
    Vec z(1); z << 1.2;
    kf.step(z);

    // Closed-form scalar update from predicted (xp, Pp+Q) actually after second predict:
    // After second predict: x' = xp, P' = Pp + Q
    double Ppred = Pp + 0.25;     // = 1.5
    double K = Ppred / (Ppred + 1.0);
    double x_post = xp + K * (1.2 - xp);
    double P_post = (1.0 - K) * Ppred;  // Joseph reduces to this in scalar case

    REQUIRE(kf.state()(0) == Approx(x_post).epsilon(1e-12));
    REQUIRE(kf.covariance()(0,0) == Approx(P_post).epsilon(1e-12));
}

TEST_CASE("Batch filter handles missing and present measurements")
{
    const int n = 2, m = 1;
    Vec x0(n); x0 << 0.0, 0.0;
    Mat P0 = Mat::Identity(n, n);
    Mat A  = (Mat(2,2) << 1.0, 1.0,
                           0.0, 1.0).finished();
    Mat H(m, n); H << 1.0, 0.0;
    Mat Q  = 0.01 * Mat::Identity(n, n);
    Mat R  = 0.25 * Mat::Identity(m, m);

    KFLinear kf(x0, P0, A, H, Q, R);

    std::vector<std::optional<Vec>> zs;
    // t=1: no measurement
    zs.push_back(std::nullopt);
    // t=2: measurement
    Vec z1(1); z1 << 0.8; zs.push_back(z1);
    // t=3: measurement
    Vec z2(1); z2 << 1.6; zs.push_back(z2);
    // t=4: no measurement
    zs.push_back(std::nullopt);

    auto states = kf.filter(zs);
    REQUIRE(states.size() == zs.size());
}

TEST_CASE("Update throws on measurement size mismatch")
{
    const int n = 2, m = 1;
    Vec x0(n); x0 << 0.0, 0.0;
    Mat P0 = Mat::Identity(n, n);
    Mat A  = Mat::Identity(n, n);
    Mat H(m, n); H << 1.0, 0.0;
    Mat Q  = Mat::Identity(n, n);
    Mat R  = Mat::Identity(m, m);

    KFLinear kf(x0, P0, A, H, Q, R);

    Vec bad_measurement(2); bad_measurement << 1.0, 2.0;
    REQUIRE_THROWS_AS(kf.update(bad_measurement), std::invalid_argument);
}
