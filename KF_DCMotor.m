%% DC Motor Position Control with Improved Kalman Filter
% This example models a DC motor with voltage input and position output
clear; clc; close all;
clear KalmanFilter  % Clear persistent variables

%% System Parameters
J = 0.01;      % Motor inertia [kg*m^2]
b = 0.1;       % Damping coefficient [N*m*s/rad]
K = 0.01;      % Motor torque constant [N*m/A]
R = 1;         % Armature resistance [Ohm]

%% State Space Model
% States: x = [theta; omega] (position and angular velocity)
% Input: u = voltage [V]
% Output: y = theta (position measurement only)

% Continuous-time model
A_c = [0 1; 0 -b/J];        % System matrix
B_c = [0; K/(J*R)];         % Input matrix
C_c = [1 0];                % Output matrix (measuring position only)

% Discretization
dt = 0.01;                   % Time step [s]
t_final = 15;                 % Final time [s]
t = 0:dt:t_final;            % Time vector
N = length(t);               % Number of time steps

% Discrete-time model using zero-order hold (ZOH)
sys_c = ss(A_c, B_c, C_c, 0);
sys_d = c2d(sys_c, dt, 'zoh');

% Extract discrete matrices
A = sys_d.A;
B = sys_d.B;
C = sys_d.C;

%% Noise Parameters - ADJUSTED TO IMPROVE POSITION FILTERING
% Process noise (uncertainty in the model)
Q = diag([0.0001, 0.01]);  % REDUCED position process noise by 10x

% Measurement noise (uncertainty in the sensor)
R = 0.1;                     % INCREASED measurement noise by 10x

%% Input Signal - Step followed by sine wave
u = zeros(1, N);
% Step input at t = 0.5s
step_start = round(0.5/dt);
u(step_start:end) = 1;
% Add sine wave at t = 2.5s
sine_start = round(2.5/dt);
for i = sine_start:N
    u(i) = 1 + 0.5*sin(2*pi*(i-sine_start)*dt); % Sine wave around 1V
end

%% Simulation Arrays
x_true = zeros(2, N);       % True state [theta; omega]
x_noisy = zeros(2, N);      % Noisy state
y_noisy = zeros(1, N);      % Noisy measurement
x_est = zeros(2, N);        % Estimated state from Kalman filter
K_gain = zeros(2, N);       % Kalman gain

% Initial state covariance (for Kalman filter)
P0 = diag([0.1, 0.1]);      % Initial uncertainty in position and velocity

%% Simulation Loop
for k = 1:N-1
    % True system (no noise)
    x_true(:, k+1) = A * x_true(:, k) + B * u(k);
    
    % Noisy system
    w = mvnrnd([0 0], Q)';             % Process noise
    v = sqrt(R) * randn();             % Measurement noise
    
    x_noisy(:, k+1) = A * x_noisy(:, k) + B * u(k) + w;
    y_noisy(k+1) = C * x_noisy(:, k+1) + v;
    
    % Kalman filter estimation
    [x_est(:, k+1), ~, K_gain(:, k+1)] = ...
        KalmanFilter(y_noisy(k+1), u(k), A, B, C, Q, R, P0, k+1);
end

%% Calculate RMS Error
rms_noisy = sqrt(mean((x_true(1,:) - x_noisy(1,:)).^2));
rms_filtered = sqrt(mean((x_true(1,:) - x_est(1,:)).^2));
fprintf('Position RMS Error:\n');
fprintf('Noisy: %f\n', rms_noisy);
fprintf('Filtered: %f\n', rms_filtered);
fprintf('Improvement: %f%%\n', 100*(1-rms_filtered/rms_noisy));

%% Plotting - SIMPLIFIED
% Figure 1: States
figure('Position', [100, 100, 900, 600]);

% Position plot
subplot(2, 1, 1);
plot(t, x_true(1, :), 'b', 'LineWidth', 2, 'DisplayName', 'True');
hold on;
plot(t, x_noisy(1, :), 'r', 'LineWidth', 1, 'DisplayName', 'Noisy');
plot(t, x_est(1, :), 'k', 'LineWidth', 1.5, 'DisplayName', 'Filtered');
xlabel('Time [s]');
ylabel('Position [rad]');
title('DC Motor Position');
legend('Location', 'best');
grid on;

% Velocity plot
subplot(2, 1, 2);
plot(t, x_true(2, :), 'b', 'LineWidth', 2, 'DisplayName', 'True');
hold on;
plot(t, x_noisy(2, :), 'r', 'LineWidth', 1, 'DisplayName', 'Noisy');
plot(t, x_est(2, :), 'k', 'LineWidth', 1.5, 'DisplayName', 'Filtered');
xlabel('Time [s]');
ylabel('Velocity [rad/s]');
title('DC Motor Velocity');
legend('Location', 'best');
grid on;

%% Kalman Filter Function
function [xhat, Pstate, Kgain] = KalmanFilter(y, u, A, B, C, Q, R, P0, timestep)
    persistent x P initialized
    
    % Initialize state and covariance on first call
    if isempty(initialized) || timestep == 2
        x = zeros(size(A, 1), 1);  % Initialize state to zero
        P = P0;                    % Initialize covariance
        initialized = 1;
    end
    
    % Prediction step (a priori)
    x_pred = A*x + B*u;
    P_pred = A*P*A' + Q;
    
    % Update step (a posteriori)
    S = C*P_pred*C' + R;          % Innovation covariance
    K = P_pred*C' / S;            % Kalman gain
    
    % Update state estimate
    x = x_pred + K*(y - C*x_pred);
    
    % Update covariance using Joseph form for better numerical stability
    I = eye(size(A));
    P = (I - K*C)*P_pred*(I - K*C)' + K*R*K';
    
    % Return values
    xhat = x;
    Pstate = P;
    Kgain = K;
end