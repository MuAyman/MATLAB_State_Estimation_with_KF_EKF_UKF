%% Unscented Kalman Filter (UKF) Implementation for a Nonlinear System
% This example implements the UKF for a pendulum-cart system with nonlinear dynamics
% States:
%   - x1: Cart position
%   - x2: Cart velocity
%   - x3: Pendulum angle (rad)
%   - x4: Pendulum angular velocity (rad/s)
% Inputs:
%   - u1: Force applied to cart
%   - u2: External disturbance
% Measurements:
%   - z1: Noisy cart position (nonlinear function)
%   - z2: Noisy pendulum angle (nonlinear function)

clear all;
close all;
clc;

%% System Parameters
dt = 0.05;          % Time step (seconds)
T = 10;             % Total simulation time (seconds)
steps = T/dt;       % Number of time steps

% Physical system parameters
m_cart = 1.0;       % Cart mass (kg)
m_pend = 0.5;       % Pendulum mass (kg)
L = 0.5;            % Pendulum length (m)
g = 9.81;           % Gravity acceleration (m/s^2)
b = 0.1;            % Friction coefficient

% UKF Parameters
n = 4;              % Number of states
m = 2;              % Number of measurements
alpha = 1e-3;       % UKF tuning parameter (usually small, e.g., 1e-3)
beta = 2;           % UKF tuning parameter (optimal for Gaussian distributions)
kappa = 0;          % UKF tuning parameter
lambda = alpha^2 * (n + kappa) - n;  % Scaling parameter

% Calculate weights for mean and covariance
Wm = zeros(2*n + 1, 1);  % Weights for mean
Wc = zeros(2*n + 1, 1);  % Weights for covariance

Wm(1) = lambda / (n + lambda);
Wc(1) = lambda / (n + lambda) + (1 - alpha^2 + beta);

for i = 2:(2*n + 1)
    Wm(i) = 1 / (2 * (n + lambda));
    Wc(i) = 1 / (2 * (n + lambda));
end

%% Define Nonlinear System Functions

% Process model function (nonlinear state transition)
% x_k+1 = f(x_k, u_k, w_k)
function x_next = process_model(x, u, w, dt, params)
    % Extract parameters
    m_cart = params.m_cart;
    m_pend = params.m_pend;
    L = params.L;
    g = params.g;
    b = params.b;
    
    % Extract states
    pos = x(1);      % Cart position
    vel = x(2);      % Cart velocity
    theta = x(3);    % Pendulum angle
    omega = x(4);    % Pendulum angular velocity
    
    % Extract inputs
    force = u(1);    % Force on cart
    dist = u(2);     % External disturbance
    
    % Simplified nonlinear dynamics
    % Calculate acceleration terms with simplified pendulum dynamics
    sin_theta = sin(theta);
    cos_theta = cos(theta);
    
    % Total force including external disturbance
    total_force = force + dist;
    
    % Denominator term for the accelerations
    den = m_cart + m_pend * sin_theta^2;
    
    % Cart acceleration
    cart_acc = (total_force + m_pend * sin_theta * (L * omega^2 + g * cos_theta) - b * vel) / den;
    
    % Pendulum angular acceleration
    pend_acc = (-total_force * cos_theta - m_pend * L * omega^2 * sin_theta * cos_theta - ...
                (m_cart + m_pend) * g * sin_theta + b * vel * cos_theta) / (L * den);
    
    % Update states using Euler integration
    pos_next = pos + vel * dt;
    vel_next = vel + cart_acc * dt;
    theta_next = theta + omega * dt;
    omega_next = omega + pend_acc * dt;
    
    % Add process noise
    x_next = [pos_next; vel_next; theta_next; omega_next] + w;
end

% Measurement model function (nonlinear)
% z_k = h(x_k, v_k)
function z = measurement_model(x, v)
    % Nonlinear measurement functions with noise
    % Measure position with a small nonlinearity
    z1 = x(1) + 0.05 * sin(x(1)) + v(1);
    
    % Measure angle with a small nonlinearity
    z2 = x(3) + 0.05 * sin(2 * x(3)) + v(2);
    
    z = [z1; z2];
end

%% Initialize System

% Initial true state [position; velocity; angle; angular velocity]
x_true = zeros(n, steps);
x_true(:,1) = [0; 0; pi/6; 0];  % Start with pendulum at 30 degrees

% Measured state (with noise)
z = zeros(m, steps);

% UKF estimate
x_est = zeros(n, steps);
x_est(:,1) = [0; 0; 0; 0];  % Initial estimate - intentionally different from true state

% UKF error covariance
P = diag([0.5, 0.5, 0.5, 0.5]);  % Initial state uncertainty

% System and measurement noise covariances
Q = diag([0.001, 0.01, 0.001, 0.01]);  % Process noise covariance - higher for velocity states
R = diag([0.1, 0.05]);                 % Measurement noise covariance

% Generate control input - force applied to cart
t = 0:dt:T-dt;
u = zeros(2, steps);

% Force applied to cart (alternating force profile)
u(1,:) = 5 * sin(1.2 * t) .* cos(0.5 * t);

% Random disturbance
u(2,:) = 0.5 * randn(1, steps);

% Generate noise sequences
process_noise = mvnrnd(zeros(n,1), Q, steps)';
meas_noise = mvnrnd(zeros(m,1), R, steps)';

% Create structure to hold parameters for process model
params.m_cart = m_cart;
params.m_pend = m_pend;
params.L = L;
params.g = g;
params.b = b;

%% Simulation Loop
for k = 1:steps-1
    % Simulate the true system with noise
    x_true(:,k+1) = process_model(x_true(:,k), u(:,k), process_noise(:,k), dt, params);
    
    % Generate noisy measurement
    z(:,k) = measurement_model(x_true(:,k), meas_noise(:,k));
    
    % ===== UKF PREDICTION STEP =====
    
    % Calculate square root of P (using Cholesky decomposition)
    P_sqrt = chol(P)';
    
    % Calculate sigma points
    chi = zeros(n, 2*n + 1);
    chi(:,1) = x_est(:,k);  % Central sigma point
    
    % Generate sigma points spread around the mean
    for i = 1:n
        chi(:,i+1) = x_est(:,k) + sqrt(n + lambda) * P_sqrt(:,i);
        chi(:,i+1+n) = x_est(:,k) - sqrt(n + lambda) * P_sqrt(:,i);
    end
    
    % Propagate sigma points through process model
    chi_pred = zeros(n, 2*n + 1);
    
    for i = 1:(2*n + 1)
        chi_pred(:,i) = process_model(chi(:,i), u(:,k), zeros(n,1), dt, params);
    end
    
    % Calculate predicted state mean
    x_pred = zeros(n, 1);
    for i = 1:(2*n + 1)
        x_pred = x_pred + Wm(i) * chi_pred(:,i);
    end
    
    % Calculate predicted covariance
    P_pred = zeros(n, n);
    for i = 1:(2*n + 1)
        diff = chi_pred(:,i) - x_pred;
        P_pred = P_pred + Wc(i) * (diff * diff');
    end
    P_pred = P_pred + Q;  % Add process noise covariance
    
    % ===== UKF UPDATE STEP =====
    
    % Calculate measurement sigma points
    gamma = zeros(m, 2*n + 1);
    for i = 1:(2*n + 1)
        gamma(:,i) = measurement_model(chi_pred(:,i), zeros(m,1));
    end
    
    % Calculate predicted measurement mean
    z_pred = zeros(m, 1);
    for i = 1:(2*n + 1)
        z_pred = z_pred + Wm(i) * gamma(:,i);
    end
    
    % Generate measurement for this step
    z_k = measurement_model(x_true(:,k+1), meas_noise(:,k+1));
    
    % Calculate innovation covariance
    P_zz = zeros(m, m);
    for i = 1:(2*n + 1)
        diff = gamma(:,i) - z_pred;
        P_zz = P_zz + Wc(i) * (diff * diff');
    end
    P_zz = P_zz + R;  % Add measurement noise covariance
    
    % Calculate cross-correlation matrix
    P_xz = zeros(n, m);
    for i = 1:(2*n + 1)
        diff_x = chi_pred(:,i) - x_pred;
        diff_z = gamma(:,i) - z_pred;
        P_xz = P_xz + Wc(i) * (diff_x * diff_z');
    end
    
    % Calculate Kalman gain
    K = P_xz / P_zz;
    
    % Update state estimate
    x_est(:,k+1) = x_pred + K * (z_k - z_pred);
    
    % Update covariance estimate
    P = P_pred - K * P_zz * K';
    
    % Ensure P remains symmetric (numerical stability)
    P = (P + P') / 2;
end

% Final measurement
z(:,steps) = measurement_model(x_true(:,steps), meas_noise(:,steps));

%% Calculate Error Metrics
% Compute error for raw measurements
% For position
pos_error_raw = sqrt(mean((z(1,:) - x_true(1,:)).^2));
% For angle
angle_error_raw = sqrt(mean((z(2,:) - x_true(3,:)).^2));

% Compute error for UKF estimates
% For position
pos_error_ukf = sqrt(mean((x_est(1,:) - x_true(1,:)).^2));
% For angle
angle_error_ukf = sqrt(mean((x_est(3,:) - x_true(3,:)).^2));

% Total RMS error
total_error_raw = sqrt(mean((z(1,:) - x_true(1,:)).^2) + mean((z(2,:) - x_true(3,:)).^2));
total_error_ukf = sqrt(mean((x_est(1,:) - x_true(1,:)).^2) + mean((x_est(3,:) - x_true(3,:)).^2));

% Calculate improvement percentage
pos_improvement = 100 * (pos_error_raw - pos_error_ukf) / pos_error_raw;
angle_improvement = 100 * (angle_error_raw - angle_error_ukf) / angle_error_raw;
total_improvement = 100 * (total_error_raw - total_error_ukf) / total_error_raw;

% Display results
fprintf('Position RMS Error (Raw measurements): %.4f\n', pos_error_raw);
fprintf('Position RMS Error (UKF): %.4f\n', pos_error_ukf);
fprintf('Position Improvement: %.2f%%\n\n', pos_improvement);

fprintf('Angle RMS Error (Raw measurements): %.4f rad\n', angle_error_raw);
fprintf('Angle RMS Error (UKF): %.4f rad\n', angle_error_ukf);
fprintf('Angle Improvement: %.2f%%\n\n', angle_improvement);

fprintf('Total RMS Error (Raw measurements): %.4f\n', total_error_raw);
fprintf('Total RMS Error (UKF): %.4f\n', total_error_ukf);
fprintf('Total Improvement: %.2f%%\n', total_improvement);

%% Plotting
time = 0:dt:T-dt;

% Create one figure with 4 subplots for all states
figure('Name', 'UKF Estimation Results', 'Position', [100, 100, 1000, 800]);

% Subplot 1: Cart Position
subplot(2, 2, 1);
plot(time, x_true(1,:), 'b-', 'LineWidth', 2);
hold on;
plot(time, z(1,:), 'r', 'MarkerSize', 4);
plot(time, x_est(1,:), 'k--', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Position (m)');
title('Cart Position vs Time');
legend('True', 'Noisy', 'Filtered');

% Subplot 2: Cart Velocity
subplot(2, 2, 2);
plot(time, x_true(2,:), 'b-', 'LineWidth', 2);
hold on;
% No direct measurement for velocity
plot(time, x_est(2,:), 'k--', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Velocity (m/s)');
title('Cart Velocity vs Time');
legend('True', 'Filtered');

% Subplot 3: Pendulum Angle
subplot(2, 2, 3);
plot(time, x_true(3,:), 'b-', 'LineWidth', 2);
hold on;
plot(time, z(2,:), 'r', 'MarkerSize', 4);
plot(time, x_est(3,:), 'k--', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Angle (rad)');
title('Pendulum Angle vs Time');
legend('True', 'Noisy', 'Filtered');

% Subplot 4: Pendulum Angular Velocity
subplot(2, 2, 4);
plot(time, x_true(4,:), 'b-', 'LineWidth', 2);
hold on;
% No direct measurement for angular velocity
plot(time, x_est(4,:), 'k--', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Angular Velocity (rad/s)');
title('Pendulum Angular Velocity vs Time');
legend('True', 'Filtered');