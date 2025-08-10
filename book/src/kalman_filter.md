## Model formulation

We consider a linear dynamical system with additive Gaussian noise:

$$
x_k = A x_{k-1} + B u_k + w_k, \quad w_k \sim \mathcal{N}(0, Q)
$$
$$
z_k = H x_k + v_k, \quad v_k \sim \mathcal{N}(0, R)
$$

where:

- $x_k \in \mathbb{R}^n$ is the state vector at time step $k$,
- $u_k \in \mathbb{R}^m$ is an optional control input,
- $z_k \in \mathbb{R}^p$ is the measurement vector,
- $A \in \mathbb{R}^{n \times n}$ is the state transition matrix,
- $B \in \mathbb{R}^{n \times m}$ is the control-input matrix,
- $H \in \mathbb{R}^{p \times n}$ is the observation matrix,
- $Q \in \mathbb{R}^{n \times n}$ is the process noise covariance,
- $R \in \mathbb{R}^{p \times p}$ is the measurement noise covariance.

## Kalman filtering

The Kalman filter maintains the mean and covariance of the posterior distribution $ p(x_k \mid z_{1:k}) $ under the Gaussian assumption.

### Prediction step

Given the previous posterior $ (\hat{x}_{k-1}, P_{k-1}) $:

$$
\hat{x}^-_k = A \hat{x}_{k-1} + B u_k
$$
$$
P^-_k = A P_{k-1} A^\top + Q
$$

Here, $(\hat{x}^-_k, P^-_k)$ are the predicted state mean and covariance.

### Update step

With a new measurement $z_k$:

- Innovation (measurement residual):
$$
y_k = z_k - H \hat{x}^-_k
$$
- Innovation covariance:
$$
S_k = H P^-_k H^\top + R
$$
- Kalman gain:
$$
K_k = P^-_k H^\top S_k^{-1}
$$
- Updated mean and covariance:
$$
\hat{x}_k = \hat{x}^-_k + K_k y_k
$$
$$
P_k = (I - K_k H) P^-_k (I - K_k H)^\top + K_k R K_k^\top
$$

The filter proceeds recursively for each time step.