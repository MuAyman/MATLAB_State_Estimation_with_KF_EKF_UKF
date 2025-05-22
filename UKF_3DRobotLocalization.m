% Unscented Kalman Filter Implementation for 3D Robot Localization with Range Measurements
% This script models a 3D robot with nonlinear dynamics and nonlinear range measurements
% to fixed beacons, then applies a UKF to estimate the state

clear all;
close all;
clc;

%% System Parameters
% Robot parameters
mass = 2.0;          % Mass of robot [kg]
I_z = 0.5;           % Moment of inertia [kg*m^2]
dt = 0.1;            % Time step [s]
t_final = 15;        % Simulation duration [s]
t = 0:dt:t_final;    % Time vector
num_steps = length(t);

% Beacon locations (fixed positions in 3D space)
beacons = [
    10, 10, -2;      % Beacon 1: [x, y, z]
    -5, 8, 1;        % Beacon 2
    0, -7, 3;        % Beacon 3
    12, -3, 0;       % Beacon 4
    -8, -9, -1       % Beacon 5
];
num_beacons = size(beacons, 1);

%% UKF Parameters
n = 9;               % State dimension [x, y, z, vx, vy, vz, roll, pitch, yaw]
m = num_beacons;     % Measurement dimension (range to each beacon)

% UKF specific parameters
alpha = 0.1;         % Spread parameter
beta = 2;            % Distribution parameter (2 is optimal for Gaussian)
kappa = 0;           % Secondary scaling parameter
lambda = alpha^2 * (n + kappa) - n;  % Scaling parameter

% Calculate weights
Wm = zeros(2*n + 1, 1);  % Weights for mean
Wc = zeros(2*n + 1, 1);  % Weights for covariance
Wm(1) = lambda / (n + lambda);
Wc(1) = lambda / (n + lambda) + (1 - alpha^2 + beta);
for i = 2:(2*n + 1)
    Wm(i) = 1 / (2 * (n + lambda));
    Wc(i) = 1 / (2 * (n + lambda));
end

%% Process and Measurement Noise
% Process noise covariance (affects state evolution)
Q_continuous = diag([
    0.2, 0.2, 0.1,              % Position noise [m^2/s^3]
    0.05, 0.05, 0.02,           % Velocity noise [m^2/s^3]
    0.01, 0.01, 0.03            % Attitude noise [rad^2/s]
]);
Q = Q_continuous * dt;          % Discrete time process noise

% Make sure Q is square and proper size
if size(Q, 1) ~= n || size(Q, 2) ~= n
    Q = diag(diag(Q));  % Extract diagonal and create proper square matrix
    if length(diag(Q)) < n
        % Pad if necessary
        Q = diag([diag(Q); zeros(n-length(diag(Q)), 1)]);
    elseif length(diag(Q)) > n
        % Truncate if necessary
        Q = diag(diag(Q, 0),(1:n));
    end
end

% Range measurement noise (varies with distance)
R_base = 0.25;                  % Base variance [m^2]
R_factor = 0.01;                % Additional variance factor based on distance

%% Initial State and Estimates
% True initial state [x, y, z, vx, vy, vz, roll, pitch, yaw]
x_true = [0; 0; 0; 1; 0.5; 0.2; 0; 0; pi/4];

% Initial state estimate for UKF (with some error)
x_est = [0.5; -0.5; 0.2; 0.8; 0.3; 0; 0.1; 0.1; pi/3];

% Initial error covariance matrix for UKF
P = diag([1, 1, 0.5, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2]);

%% Generate Control Input (Complex 3D trajectory)
% Pre-generate control inputs for the whole trajectory
u = zeros(6, num_steps);  % [Fx, Fy, Fz, Mx, My, Mz]

% Create a complex trajectory with changing accelerations and rotations
for k = 1:num_steps
    t_cur = t(k);
    
    % Sinusoidal forces in x and y directions
    u(1, k) = 2 * sin(0.5 * t_cur);         % Force in x
    u(2, k) = 1.5 * cos(0.3 * t_cur);       % Force in y
    
    % Varying force in z direction
    if t_cur < 5
        u(3, k) = 0.5;                      % Lift up
    elseif t_cur < 10
        u(3, k) = -0.3;                     % Down
    else
        u(3, k) = 0.1 * sin(t_cur);         % Oscillate
    end
    
    % Moments causing rotation
    u(4, k) = 0.2 * sin(0.7 * t_cur);       % Roll moment
    u(5, k) = 0.2 * cos(0.5 * t_cur);       % Pitch moment
    u(6, k) = 0.3 * sin(0.2 * t_cur);       % Yaw moment
end

%% Generate Process Noise
% Generate process noise manually using randn
process_noise = zeros(n, num_steps);
for k = 1:num_steps
    % Generate independent standard normal samples
    noise_std = sqrt(diag(Q));
    process_noise(:, k) = noise_std .* randn(n, 1);
end

%% Simulation and UKF Implementation
% Preallocate memory for results
x_true_history = zeros(n, num_steps);
x_est_history = zeros(n, num_steps);
P_diag_history = zeros(n, num_steps);
y_meas_history = zeros(m, num_steps);

% Set initial values
x_true_history(:, 1) = x_true;
x_est_history(:, 1) = x_est;
P_diag_history(:, 1) = diag(P);

% Pre-compute sigma point scaling constant
gamma = sqrt(n + lambda);

% Main simulation loop
for k = 1:num_steps-1
    % Current control input
    u_k = u(:, k);
    
    %% 1. Propagate true state (with process noise)
    x_true = robot_dynamics(x_true, u_k, mass, I_z, dt) + process_noise(:, k);
    x_true_history(:, k+1) = x_true;
    
    %% 2. Generate noisy measurement (range to beacons)
    ranges_true = zeros(num_beacons, 1);
    for i = 1:num_beacons
        % Calculate true range to beacon i
        beacon_pos = beacons(i, :)';
        ranges_true(i) = norm(x_true(1:3) - beacon_pos);
        
        % Add distance-dependent noise
        range_var = R_base + R_factor * ranges_true(i);
        ranges_true(i) = ranges_true(i) + sqrt(range_var) * randn;
    end
    y_meas = ranges_true;
    y_meas_history(:, k+1) = y_meas;
    
    %% 3. UKF Prediction Step
    % Generate sigma points
    sigma_points = generate_sigma_points(x_est, P, gamma);
    
    % Propagate sigma points through nonlinear dynamics
    sigma_points_pred = zeros(size(sigma_points));
    for i = 1:(2*n + 1)
        sigma_points_pred(:, i) = robot_dynamics(sigma_points(:, i), u_k, mass, I_z, dt);
    end
    
    % Calculate predicted state and covariance
    x_pred = zeros(n, 1);
    for i = 1:(2*n + 1)
        x_pred = x_pred + Wm(i) * sigma_points_pred(:, i);
    end
    
    P_pred = Q;  % Initialize with process noise
    for i = 1:(2*n + 1)
        diff = sigma_points_pred(:, i) - x_pred;
        % Normalize angular differences to [-pi, pi]
        diff(7:9) = wrapToPi(diff(7:9));
        P_pred = P_pred + Wc(i) * (diff * diff');
    end
    
    %% 4. UKF Update Step
    % Generate new sigma points around predicted state
    sigma_points = generate_sigma_points(x_pred, P_pred, gamma);
    
    % Propagate sigma points through measurement function
    y_sigma = zeros(m, 2*n + 1);
    for i = 1:(2*n + 1)
        % Calculate ranges to all beacons for this sigma point
        for j = 1:num_beacons
            beacon_pos = beacons(j, :)';
            y_sigma(j, i) = norm(sigma_points(1:3, i) - beacon_pos);
        end
    end
    
    % Calculate predicted measurement and innovation covariance
    y_pred = zeros(m, 1);
    for i = 1:(2*n + 1)
        y_pred = y_pred + Wm(i) * y_sigma(:, i);
    end
    
    % Measurement covariance (distance-dependent)
    R = diag(R_base + R_factor * y_pred);
    
    % Innovation covariance
    S = R;  % Initialize with measurement noise
    for i = 1:(2*n + 1)
        diff = y_sigma(:, i) - y_pred;
        S = S + Wc(i) * (diff * diff');
    end
    
    % Cross correlation
    Pxy = zeros(n, m);
    for i = 1:(2*n + 1)
        diff_x = sigma_points(:, i) - x_pred;
        diff_x(7:9) = wrapToPi(diff_x(7:9));  % Normalize angles
        diff_y = y_sigma(:, i) - y_pred;
        Pxy = Pxy + Wc(i) * (diff_x * diff_y');
    end
    
    % Kalman gain
    K = Pxy / (S + 1e-8*eye(size(S)));  % Add small regularization for numerical stability
    
    % Update state estimate
    innovation = y_meas - y_pred;
    x_est = x_pred + K * innovation;
    
    % Update error covariance
    P = P_pred - K * S * K';
    
    % Ensure P remains symmetric
    P = (P + P') / 2;
    
    % Ensure P is positive definite (add small regularization if needed)
    [V, D] = eig(P);
    if any(diag(D) < 0)
        % Fix negative eigenvalues if they exist
        D_fixed = max(D, 1e-6 * eye(size(D)));
        P = V * D_fixed * V';
    end
    
    % Store results
    x_est_history(:, k+1) = x_est;
    P_diag_history(:, k+1) = diag(P);
end

%% Plotting Results
% Plot 3D trajectory
figure('Position', [100, 100, 1000, 800]);
plot3(x_true_history(1, :), x_true_history(2, :), x_true_history(3, :), 'b-', 'LineWidth', 2);
hold on;
plot3(x_est_history(1, :), x_est_history(2, :), x_est_history(3, :), 'r--', 'LineWidth', 2);
scatter3(beacons(:, 1), beacons(:, 2), beacons(:, 3), 100, 'g', 'filled');
for i = 1:num_beacons
    text(beacons(i, 1), beacons(i, 2), beacons(i, 3), ['  Beacon ', num2str(i)]);
end
hold off;
title('3D Robot Trajectory');
xlabel('X Position [m]');
ylabel('Y Position [m]');
zlabel('Z Position [m]');
legend('True Trajectory', 'UKF Estimate', 'Beacons', 'Location', 'best');
grid on;
view(45, 30);  % Set view angle

% Plot positions
figure('Position', [100, 100, 1200, 800]);
subplot(3, 3, 1);
plot(t, x_true_history(1, :), 'b-', 'LineWidth', 1.5);
hold on;
plot(t, x_est_history(1, :), 'r--', 'LineWidth', 1.5);
hold off;
title('X Position');
xlabel('Time [s]');
ylabel('Position [m]');
legend('True', 'UKF Estimate');
grid on;

subplot(3, 3, 2);
plot(t, x_true_history(2, :), 'b-', 'LineWidth', 1.5);
hold on;
plot(t, x_est_history(2, :), 'r--', 'LineWidth', 1.5);
hold off;
title('Y Position');
xlabel('Time [s]');
ylabel('Position [m]');
legend('True', 'UKF Estimate');
grid on;

subplot(3, 3, 3);
plot(t, x_true_history(3, :), 'b-', 'LineWidth', 1.5);
hold on;
plot(t, x_est_history(3, :), 'r--', 'LineWidth', 1.5);
hold off;
title('Z Position');
xlabel('Time [s]');
ylabel('Position [m]');
legend('True', 'UKF Estimate');
grid on;

% Plot velocities
subplot(3, 3, 4);
plot(t, x_true_history(4, :), 'b-', 'LineWidth', 1.5);
hold on;
plot(t, x_est_history(4, :), 'r--', 'LineWidth', 1.5);
hold off;
title('X Velocity');
xlabel('Time [s]');
ylabel('Velocity [m/s]');
legend('True', 'UKF Estimate');
grid on;

subplot(3, 3, 5);
plot(t, x_true_history(5, :), 'b-', 'LineWidth', 1.5);
hold on;
plot(t, x_est_history(5, :), 'r--', 'LineWidth', 1.5);
hold off;
title('Y Velocity');
xlabel('Time [s]');
ylabel('Velocity [m/s]');
legend('True', 'UKF Estimate');
grid on;

subplot(3, 3, 6);
plot(t, x_true_history(6, :), 'b-', 'LineWidth', 1.5);
hold on;
plot(t, x_est_history(6, :), 'r--', 'LineWidth', 1.5);
hold off;
title('Z Velocity');
xlabel('Time [s]');
ylabel('Velocity [m/s]');
legend('True', 'UKF Estimate');
grid on;

% Plot attitudes (roll, pitch, yaw)
subplot(3, 3, 7);
plot(t, x_true_history(7, :), 'b-', 'LineWidth', 1.5);
hold on;
plot(t, x_est_history(7, :), 'r--', 'LineWidth', 1.5);
hold off;
title('Roll Angle');
xlabel('Time [s]');
ylabel('Angle [rad]');
legend('True', 'UKF Estimate');
grid on;

subplot(3, 3, 8);
plot(t, x_true_history(8, :), 'b-', 'LineWidth', 1.5);
hold on;
plot(t, x_est_history(8, :), 'r--', 'LineWidth', 1.5);
hold off;
title('Pitch Angle');
xlabel('Time [s]');
ylabel('Angle [rad]');
legend('True', 'UKF Estimate');
grid on;

subplot(3, 3, 9);
plot(t, x_true_history(9, :), 'b-', 'LineWidth', 1.5);
hold on;
plot(t, x_est_history(9, :), 'r--', 'LineWidth', 1.5);
hold off;
title('Yaw Angle');
xlabel('Time [s]');
ylabel('Angle [rad]');
legend('True', 'UKF Estimate');
grid on;

% Plot estimation errors and 3-sigma bounds
figure('Position', [100, 100, 1200, 800]);
state_labels = {'X Position [m]', 'Y Position [m]', 'Z Position [m]', ...
                'X Velocity [m/s]', 'Y Velocity [m/s]', 'Z Velocity [m/s]', ...
                'Roll [rad]', 'Pitch [rad]', 'Yaw [rad]'};
            
for i = 1:n
    subplot(3, 3, i);
    err = x_true_history(i, :) - x_est_history(i, :);
    % Wrap angle errors to [-pi, pi]
    if i >= 7
        err = wrapToPi(err);
    end
    sigma_bounds = 3 * sqrt(P_diag_history(i, :));
    
    plot(t, err, 'b-', 'LineWidth', 1.2);
    hold on;
    plot(t, sigma_bounds, 'r--', t, -sigma_bounds, 'r--', 'LineWidth', 1);
    hold off;
    
    title(['Error - ' state_labels{i}(1:end-4)]);
    xlabel('Time [s]');
    ylabel(['Error in ' state_labels{i}]);
    legend('Error', '±3σ Bounds');
    grid on;
end

% Plot range measurements to each beacon
figure('Position', [100, 100, 1000, 800]);
subplot(2, 1, 1);
for i = 1:num_beacons
    plot(t, y_meas_history(i, :), 'LineWidth', 1.5, 'DisplayName', ['Beacon ' num2str(i)]);
    hold on;
end
hold off;
title('Range Measurements to Beacons');
xlabel('Time [s]');
ylabel('Range [m]');
legend('Location', 'best');
grid on;

% Plot control inputs
subplot(2, 1, 2);
plot(t, u(1, :), 'LineWidth', 1.5, 'DisplayName', 'Force X');
hold on;
plot(t, u(2, :), 'LineWidth', 1.5, 'DisplayName', 'Force Y');
plot(t, u(3, :), 'LineWidth', 1.5, 'DisplayName', 'Force Z');
plot(t, u(4, :), 'LineWidth', 1.5, 'DisplayName', 'Moment Roll');
plot(t, u(5, :), 'LineWidth', 1.5, 'DisplayName', 'Moment Pitch');
plot(t, u(6, :), 'LineWidth', 1.5, 'DisplayName', 'Moment Yaw');
hold off;
title('Control Inputs');
xlabel('Time [s]');
ylabel('Force [N] / Moment [N·m]');
legend('Location', 'best');
grid on;

%% Helper Functions

% Generate sigma points using the given state, covariance, and scaling parameter
function sigma_points = generate_sigma_points(x, P, gamma)
    n = length(x);
    sigma_points = zeros(n, 2*n + 1);
    
    % Calculate matrix square root of P using Cholesky decomposition
    % Ensure P is symmetric positive definite
    P = (P + P') / 2;  % Ensure symmetry
    
    % Add small regularization if needed
    [V, D] = eig(P);
    if any(diag(D) < 0)
        D = max(D, 1e-6 * eye(size(D)));
        P = V * D * V';
    end
    
    try
        S = chol(P, 'lower');  % Lower triangular Cholesky factor
    catch
        % If Cholesky fails, use SVD as alternative
        [U, D, ~] = svd(P);
        S = U * sqrt(D);
    end
    
    % Set sigma points
    sigma_points(:, 1) = x;
    for i = 1:n
        sigma_points(:, i+1) = x + gamma * S(:, i);
        sigma_points(:, i+n+1) = x - gamma * S(:, i);
    end
end

% Robot dynamics function
function x_next = robot_dynamics(x, u, mass, I_z, dt)
    % State: [x, y, z, vx, vy, vz, roll, pitch, yaw]
    % Control: [Fx, Fy, Fz, Mx, My, Mz]
    
    % Extract state components
    pos = x(1:3);       % [x, y, z]
    vel = x(4:6);       % [vx, vy, vz]
    att = x(7:9);       % [roll, pitch, yaw]
    
    % Extract control components
    F = u(1:3);         % [Fx, Fy, Fz]
    M = u(4:6);         % [Mx, My, Mz]
    
    % Calculate rotation matrix (body to inertial)
    roll = att(1);
    pitch = att(2);
    yaw = att(3);
    
    % Rotation matrix (ZYX Euler angles)
    Rx = [1, 0, 0; 0, cos(roll), -sin(roll); 0, sin(roll), cos(roll)];
    Ry = [cos(pitch), 0, sin(pitch); 0, 1, 0; -sin(pitch), 0, cos(pitch)];
    Rz = [cos(yaw), -sin(yaw), 0; sin(yaw), cos(yaw), 0; 0, 0, 1];
    R = Rz * Ry * Rx;  % Inertial to body
    
    % Simple dynamics (ignoring complex aerodynamic effects)
    % Position update
    pos_next = pos + dt * vel;
    
    % Velocity update (F = ma)
    vel_next = vel + dt * (R' * F / mass);
    
    % Attitude update (simplified - assuming small angles)
    % For a real robot, quaternions would be better
    att_rates = [M(1)/I_z; M(2)/I_z; M(3)/I_z];  % Simplified inertia
    att_next = att + dt * att_rates;
    
    % Normalize angles to [-pi, pi]
    att_next = wrapToPi(att_next);
    
    % Assemble next state
    x_next = [pos_next; vel_next; att_next];
end