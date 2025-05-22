%% Kalman Filter Implementation for Drone Position Tracking
% Model: 2D drone movement with position and velocity states
% States: [x_pos, x_vel, y_pos, y_vel]
% Inputs: [x_accel, y_accel, external_disturbance]
% Measurements: noisy position readings [x_pos, y_pos]

clear all;
close all;
clc;

%% System Parameters
dt = 0.1;         % Time step (seconds)
T = 20;           % Total simulation time (seconds)
steps = T/dt;     % Number of time steps

% State vector: [x_pos, x_vel, y_pos, y_vel]
% State transition matrix (physics model)
A = [1 dt 0 0;    % x_pos = x_pos + x_vel*dt
     0 1  0 0;    % x_vel = x_vel
     0 0  1 dt;   % y_pos = y_pos + y_vel*dt
     0 0  0 1];   % y_vel = y_vel

% Control input matrix [x_accel, y_accel, disturbance]
B = [0.5*dt^2 0        0.1*dt^2;  % Effect on x_pos
     dt       0        0.05;      % Effect on x_vel
     0        0.5*dt^2 0.1*dt^2;  % Effect on y_pos
     0        dt       0.05];     % Effect on y_vel

% Measurement matrix (we can only measure positions, not velocities)
C = [1 0 0 0;     % Measure x_pos
     0 0 1 0];    % Measure y_pos

% Process noise covariance
Q = diag([0.01, 0.05, 0.01, 0.05]);  % Process noise - higher for velocity states

% Measurement noise covariance
R = diag([1, 1]);                % Measurement noise for position readings

%% Generate Control Inputs
% Create a complex but smooth path using sinusoids with different frequencies
t = 0:dt:T-dt;

% Generate sinusoidal control inputs for x and y accelerations
u_x = 0.5*sin(0.5*t) + 0.3*cos(0.2*t);
u_y = 0.4*sin(0.3*t) + 0.2*sin(0.7*t);

% Add a random external disturbance 
disturbance = 0.1*randn(1, steps);

% Combine all inputs
u = [u_x; u_y; disturbance];

%% Initialize System
% True initial state
x_true = zeros(4, steps);
x_true(:,1) = [0; 0; 0; 0];  % Start at origin with zero velocity

% Measured state (with noise)
z = zeros(2, steps);

% Kalman filter estimate
x_est = zeros(4, steps);
x_est(:,1) = [0; 0; 0; 0];  % Initial guess

% Kalman filter error covariance
P = eye(4);  % Initial uncertainty in state estimate

%% Simulation
% Generate process noise
w = mvnrnd(zeros(4,1), Q, steps)';

% Generate measurement noise
v = mvnrnd(zeros(2,1), R, steps)';

% Run simulation
for k = 1:steps-1
    % True system evolution (with process noise)
    x_true(:,k+1) = A * x_true(:,k) + B * u(:,k) + w(:,k);
    
    % Noisy measurements
    z(:,k) = C * x_true(:,k) + v(:,k);
    
    % Kalman Filter Implementation
    % Step 1: Prediction
    x_pred = A * x_est(:,k) + B * u(:,k);
    P_pred = A * P * A' + Q;
    
    % Step 2: Update
    K = P_pred * C' * inv(C * P_pred * C' + R);  % Kalman gain
    
    % Measurement for current step
    z_k = C * x_true(:,k+1) + v(:,k+1);  % Use k+1 to align with the next state
    
    % Update state estimate and covariance
    x_est(:,k+1) = x_pred + K * (z_k - C * x_pred);
    P = (eye(4) - K * C) * P_pred;
end

% Final measurement for the last step
z(:,steps) = C * x_true(:,steps) + v(:,steps);

%% Calculate Error Metrics
% Position error RMS
pos_error_raw = sqrt(mean((z(1,:) - x_true(1,:)).^2 + (z(2,:) - x_true(3,:)).^2));
pos_error_kf = sqrt(mean((x_est(1,:) - x_true(1,:)).^2 + (x_est(3,:) - x_true(3,:)).^2));

fprintf('Position RMS Error (Raw measurements): %.4f\n', pos_error_raw);
fprintf('Position RMS Error (Kalman filter): %.4f\n', pos_error_kf);
fprintf('Improvement: %.2f%%\n', 100*(pos_error_raw - pos_error_kf)/pos_error_raw);

%% Plotting
time = 0:dt:T-dt;

% Figure 1: Position Trajectory (2D)
figure('Name', 'Drone Position Trajectory', 'Position', [100, 100, 800, 600]);
plot(x_true(1,:), x_true(3,:), 'g-', 'LineWidth', 2);
hold on;
plot(z(1,:), z(2,:), 'r', 'MarkerSize', 4);
plot(x_est(1,:), x_est(3,:), 'b-', 'LineWidth', 1.5);
grid on;
xlabel('X Position');
ylabel('Y Position');
title('Drone Position Trajectory');
legend('True Position', 'Noisy Measurements', 'Kalman Filter Estimate');
axis equal;

% Figure 2: Position vs Time
figure('Name', 'Position vs Time', 'Position', [100, 100, 1000, 800]);

% X Position subplot
subplot(2,1,1);
plot(time, x_true(1,:), 'g-', 'LineWidth', 2);
hold on;
plot(time, z(1,:), 'r', 'MarkerSize', 4);
plot(time, x_est(1,:), 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('X Position');
title('X Position vs Time');
legend('True Position', 'Noisy Measurements', 'Kalman Filter Estimate');

% Y Position subplot
subplot(2,1,2);
plot(time, x_true(3,:), 'g-', 'LineWidth', 2);
hold on;
plot(time, z(2,:), 'r', 'MarkerSize', 4);
plot(time, x_est(3,:), 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Y Position');
title('Y Position vs Time');
legend('True Position', 'Noisy Measurements', 'Kalman Filter Estimate');

% Figure 3: Velocity vs Time
figure('Name', 'Velocity vs Time', 'Position', [100, 100, 1000, 800]);

% X Velocity subplot
subplot(2,1,1);
plot(time, x_true(2,:), 'g-', 'LineWidth', 2);
hold on;
plot(time, x_est(2,:), 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('X Velocity');
title('X Velocity vs Time');
legend('True Velocity', 'Kalman Filter Estimate');

% Y Velocity subplot
subplot(2,1,2);
plot(time, x_true(4,:), 'g-', 'LineWidth', 2);
hold on;
plot(time, x_est(4,:), 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Y Velocity');
title('Y Velocity vs Time');
legend('True Velocity', 'Kalman Filter Estimate');
