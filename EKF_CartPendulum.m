% Extended Kalman Filter Implementation for Cart-Pendulum System
% This script models a nonlinear cart-pendulum system with noise
% and applies an EKF to estimate the state

clear all;
close all;
clc;

%% System Parameters
m_cart = 1.0;      % Mass of the cart [kg]
m_pend = 0.5;      % Mass of the pendulum [kg]
L = 1.0;           % Length of the pendulum [m]
g = 9.81;          % Gravitational acceleration [m/s^2]
b = 0.1;           % Damping coefficient [N·s/m]

%% Simulation Parameters
dt = 0.01;         % Time step [s]
t_final = 10;      % Simulation duration [s]
t = 0:dt:t_final;  % Time vector
num_steps = length(t);

%% Process and Measurement Noise
% Process noise covariance (affects state evolution)
Q = diag([0.01, 0.05, 0.01, 0.05]);

% Measurement noise covariance
R = diag([1, 1]);  % Noise in position and angle measurements

% Generate random process and measurement noise sequences
process_noise = mvnrnd(zeros(4,1), Q, num_steps)';
meas_noise = mvnrnd(zeros(2,1), R, num_steps)';

%% Initial State and Estimates
% True initial state [x, x_dot, theta, theta_dot]
% x: cart position, theta: pendulum angle (0 = upright position)
x_true = [0; 0; pi/6; 0];  % Start with pendulum at 30 degrees

% Initial state estimate for EKF
x_est = [0; 0; 0; 0];      % Initial guess (inaccurate)

% Initial error covariance matrix for EKF
P = diag([0.5, 0.5, 0.5, 0.5]);

%% Simulation and EKF Implementation
% Preallocate memory for results
x_true_history = zeros(4, num_steps);
x_est_history = zeros(4, num_steps);
y_meas_history = zeros(2, num_steps);
P_diag_history = zeros(4, num_steps);

% External force applied to cart (optional - add a pulse force)
F = zeros(num_steps, 1);
F(200:300) = 5;  % Apply 5N force between t=2s and t=3s

% Set initial values
x_true_history(:, 1) = x_true;
x_est_history(:, 1) = x_est;
y_meas_history(:, 1) = [x_true(1); x_true(3)] + meas_noise(:, 1);
P_diag_history(:, 1) = diag(P);

% Main simulation loop
for k = 1:num_steps-1
    %% 1. Propagate true state (with process noise)
    current_F = F(k);
    x_true = nonlinear_dynamics(x_true, current_F, m_cart, m_pend, L, g, b, dt) + process_noise(:, k);
    x_true_history(:, k+1) = x_true;
    
    %% 2. Generate noisy measurement
    y_meas = [x_true(1); x_true(3)] + meas_noise(:, k+1);  % Measure position and angle
    y_meas_history(:, k+1) = y_meas;
    
    %% 3. EKF Prediction Step
    % Predict state
    x_pred = nonlinear_dynamics(x_est, current_F, m_cart, m_pend, L, g, b, dt);
    
    % Calculate Jacobian matrix (linearization of dynamics around current estimate)
    A = jacobian_matrix(x_est, current_F, m_cart, m_pend, L, g, b, dt);
    
    % Predict error covariance
    P_pred = A * P * A' + Q;
    
    %% 4. EKF Update Step
    % Measurement model Jacobian (linear for this case: we measure position and angle directly)
    H = [1, 0, 0, 0;   % Position measurement
         0, 0, 1, 0];  % Angle measurement
    
    % Predicted measurement
    y_pred = [x_pred(1); x_pred(3)];
    
    % Innovation (measurement residual)
    innovation = y_meas - y_pred;
    
    % Innovation covariance
    S = H * P_pred * H' + R;
    
    % Kalman gain
    K = P_pred * H' / S;
    
    % Update state estimate
    x_est = x_pred + K * innovation;
    
    % Update error covariance
    P = (eye(4) - K * H) * P_pred;
    
    % Store results
    x_est_history(:, k+1) = x_est;
    P_diag_history(:, k+1) = diag(P);
end

%% Plotting Results
% Plot 1: Cart position
figure('Position', [100, 100, 1200, 800]);
subplot(2, 2, 1);
plot(t, x_true_history(1, :), 'b-', 'LineWidth', 1.5);
hold on;
plot(t, y_meas_history(1, :), 'r-', 'MarkerSize', 2);
plot(t, x_est_history(1, :), 'k-', 'LineWidth', 1.5);
hold off;
title('Cart Position');
xlabel('Time [s]');
ylabel('Position [m]');
legend('True', 'Measured', 'EKF Estimate');
grid on;

% Plot 2: Cart velocity
subplot(2, 2, 2);
plot(t, x_true_history(2, :), 'b-', 'LineWidth', 1.5);
hold on;
plot(t, x_est_history(2, :), 'k-', 'LineWidth', 1.5);
hold off;
title('Cart Velocity');
xlabel('Time [s]');
ylabel('Velocity [m/s]');
legend('True', 'EKF Estimate');
grid on;

% Plot 3: Pendulum angle
subplot(2, 2, 3);
plot(t, x_true_history(3, :), 'b-', 'LineWidth', 1.5);
hold on;
plot(t, y_meas_history(2, :), 'r-', 'MarkerSize', 2);
plot(t, x_est_history(3, :), 'k-', 'LineWidth', 1.5);
hold off;
title('Pendulum Angle');
xlabel('Time [s]');
ylabel('Angle [rad]');
legend('True', 'Measured', 'EKF Estimate');
grid on;

% Plot 4: Pendulum angular velocity
subplot(2, 2, 4);
plot(t, x_true_history(4, :), 'b-', 'LineWidth', 1.5);
hold on;
plot(t, x_est_history(4, :), 'k-', 'LineWidth', 1.5);
hold off;
title('Pendulum Angular Velocity');
xlabel('Time [s]');
ylabel('Angular Velocity [rad/s]');
legend('True', 'EKF Estimate');
grid on;

% % Plot 5: Estimation error and 3-sigma bounds (separate figure)
% figure('Position', [100, 100, 1200, 800]);
% for i = 1:4
%     subplot(2, 2, i);
%     err = x_true_history(i, :) - x_est_history(i, :);
%     sigma_bounds = 3 * sqrt(P_diag_history(i, :));
% 
%     plot(t, err, 'b-', 'LineWidth', 1.2);
%     hold on;
%     plot(t, sigma_bounds, 'r--', t, -sigma_bounds, 'r--', 'LineWidth', 1);
%     hold off;
% 
%     labels = {'Position Error [m]', 'Velocity Error [m/s]', ...
%              'Angle Error [rad]', 'Angular Velocity Error [rad/s]'};
%     title(['Estimation Error - ' labels{i}(1:end-4)]);
%     xlabel('Time [s]');
%     ylabel(labels{i});
%     legend('Error', '±3σ Bounds');
%     grid on;
% end
%
% % Plot 6: Applied force (separate figure)
% figure('Position', [100, 100, 600, 400]);
% plot(t, F, 'LineWidth', 1.5);
% title('Applied Force');
% xlabel('Time [s]');
% ylabel('Force [N]');
% grid on;

%% Helper Functions

% Nonlinear Dynamics Function
function x_next = nonlinear_dynamics(x, F, m_cart, m_pend, L, g, b, dt)
    % State: [x, x_dot, theta, theta_dot]
    % Extract state variables
    x_pos = x(1);
    x_vel = x(2);
    theta = x(3);
    theta_dot = x(4);
    
    % Calculate accelerations based on nonlinear dynamics
    M = m_cart + m_pend;
    
    % Compute denominator term (used in both equations)
    den = M * L - m_pend * L * cos(theta)^2;
    
    % Cart acceleration
    x_acc = (F + m_pend * L * theta_dot^2 * sin(theta) - b * x_vel + ...
            m_pend * g * sin(theta) * cos(theta)) / den;
    
    % Pendulum angular acceleration
    theta_acc = (-F * cos(theta) - m_pend * L * theta_dot^2 * sin(theta) * cos(theta) + ...
                b * x_vel * cos(theta) - M * g * sin(theta)) / den;
    
    % Euler integration for simplicity
    x_next = zeros(4, 1);
    x_next(1) = x_pos + dt * x_vel;
    x_next(2) = x_vel + dt * x_acc;
    x_next(3) = theta + dt * theta_dot;
    x_next(4) = theta_dot + dt * theta_acc;
end

% Jacobian Matrix Calculation (linearization of dynamics)
function A = jacobian_matrix(x, F, m_cart, m_pend, L, g, b, dt)
    % For this complex system, we'll use numerical differentiation
    % to approximate the Jacobian
    
    % Small perturbation for numerical differentiation
    eps = 1e-6;
    
    % Base state evolution
    x_next_base = nonlinear_dynamics(x, F, m_cart, m_pend, L, g, b, dt);
    
    % Initialize Jacobian
    A = zeros(4, 4);
    
    % Compute each column of the Jacobian
    for i = 1:4
        x_perturbed = x;
        x_perturbed(i) = x_perturbed(i) + eps;
        
        x_next_perturbed = nonlinear_dynamics(x_perturbed, F, m_cart, m_pend, L, g, b, dt);
        
        A(:, i) = (x_next_perturbed - x_next_base) / eps;
    end
end