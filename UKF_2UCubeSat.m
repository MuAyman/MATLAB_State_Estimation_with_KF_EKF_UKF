% UKF Implementation for 2U CubeSat with Multiple Subsystems
% This script implements an Unscented Kalman Filter for a complex nonlinear CubeSat model
% with multiple states, inputs, and measurements, including noise and disturbances.

clear;
close all;
clc;

%% CubeSat Parameters
% 2U CubeSat dimensions and properties
cubesat.mass = 2.66;        % kg
cubesat.inertia = diag([0.0135, 0.0135, 0.0045]); % kg·m^2 (Principal moments of inertia)
cubesat.dim = [0.1, 0.1, 0.2];  % m (2U CubeSat dimensions)

% Simulation parameters
dt = 0.1;                  % Time step (s)
T = 1000;                  % Total simulation time (s)
N = T/dt;                  % Number of steps
time = 0:dt:T-dt;          % Time vector

%% State Vector Definition (13 states)
% x = [position(3); velocity(3); quaternion(4); angular_velocity(3)]
% position: [x; y; z] in ECI frame (m)
% velocity: [vx; vy; vz] in ECI frame (m/s)
% quaternion: [q1; q2; q3; q4] (attitude quaternion, scalar last)
% angular_velocity: [wx; wy; wz] in body frame (rad/s)

% Initial state (random orbit around Earth and random attitude)
altitude = 500e3;  % Initial altitude (500 km)
radius_earth = 6371e3;  % Earth radius (m)
orbital_radius = radius_earth + altitude;
orbital_velocity = sqrt(3.986e14/orbital_radius);  % Circular orbit velocity

% Initial state
x_true = zeros(13, N);
x_true(:,1) = [
    orbital_radius; 0; 0;                 % Initial position [x; y; z]
    0; orbital_velocity; 0;               % Initial velocity [vx; vy; vz]
    0; 0; 0; 1;                           % Initial quaternion [q1; q2; q3; q4] (identity orientation)
    0.001; 0.001; 0.001                   % Initial angular velocity [wx; wy; wz] (slight rotation)
];

% Control inputs [Fx; Fy; Fz; Tx; Ty; Tz] (external force and torque)
u = zeros(6, N);

%% Process and Measurement Noise Parameters
% Process noise (state disturbances)
Q_pos = 0.01^2 * eye(3);         % Position process noise (m^2)
Q_vel = 0.005^2 * eye(3);        % Velocity process noise (m^2/s^2)
Q_quat = 0.001^2 * eye(4);      % Quaternion process noise
Q_ang_vel = 0.002^2 * eye(3);   % Angular velocity process noise (rad^2/s^2)
Q = blkdiag(Q_pos, Q_vel, Q_quat, Q_ang_vel);

% Measurement noise
R_pos = 5^2 * eye(3);           % GPS position noise (m^2)
R_mag = 1e-5 * eye(3);          % Magnetometer noise (T^2)
R_gyro = 0.01^2 * eye(3);       % Gyroscope noise (rad^2/s^2)
R_sun = 0.05^2 * eye(3);        % Sun sensor noise (unit vector)
R = blkdiag(R_pos, R_mag, R_gyro, R_sun);

% Number of measurements: GPS(3) + Magnetometer(3) + Gyroscope(3) + Sun sensor(3) = 12
m_dim = 12;

%% System simulation with true dynamics and added noise
x_noisy = zeros(13, N);
x_noisy(:,1) = x_true(:,1);
z_noisy = zeros(m_dim, N);

% Earth parameters
mu = 3.986e14;  % Earth's gravitational parameter (m^3/s^2)
B_earth = [1.8e-5; 0; 3e-5];  % Earth's magnetic field vector at reference point (T)
sun_direction = [1; 0; 0];  % Simplified constant sun direction (would vary in reality)

% External disturbances (e.g., atmospheric drag, solar pressure)
disturb_amp_force = 1e-5;   % Force disturbance amplitude (N)
disturb_amp_torque = 1e-7;  % Torque disturbance amplitude (N·m)

% Solar panel power calculation parameters
solar_panel_area = 0.01 * 0.3 * 2;  % Solar panel area (m^2)
solar_efficiency = 0.3;            % Solar panel efficiency
solar_irradiance = 1366;           % Solar irradiance (W/m^2)

% Generate true states and measurements
for k = 1:N-1
    % Get current state
    pos = x_true(1:3, k);
    vel = x_true(4:6, k);
    quat = x_true(7:10, k);
    omega = x_true(11:13, k);
    
    % External disturbances (random walk)
    disturb_force = disturb_amp_force * randn(3,1);
    disturb_torque = disturb_amp_torque * randn(3,1);
    
    % Control inputs (could be from a controller in a more complex model)
    u(:,k) = [disturb_force; disturb_torque];
    
    % Compute dynamics (nonlinear state update)
    % 1. Position and velocity update (orbital mechanics)
    r_norm = norm(pos);
    grav_accel = -mu * pos / r_norm^3;  % Gravitational acceleration
    pos_next = pos + vel * dt;
    vel_next = vel + (grav_accel + u(1:3,k)/cubesat.mass) * dt;
    
    % 2. Attitude quaternion update
    omega_matrix = [
        0, -omega(3), omega(2);
        omega(3), 0, -omega(1);
        -omega(2), omega(1), 0
    ];
    quat_dot = 0.5 * [
        -omega_matrix, omega;
        -omega', 0
    ] * quat;
    quat_next = quat + quat_dot * dt;
    quat_next = quat_next / norm(quat_next);  % Normalize quaternion
    
    % 3. Angular velocity update (Euler's equation)
    I = cubesat.inertia;
    omega_dot = I \ (u(4:6,k) + disturb_torque - cross(omega, I * omega));
    omega_next = omega + omega_dot * dt;
    
    % Combine next state
    x_true(:, k+1) = [pos_next; vel_next; quat_next; omega_next];
    
    % Add process noise to create noisy state
    x_noisy(:, k+1) = x_true(:, k+1) + sqrt(dt) * sqrtm(Q) * randn(13, 1);
    x_noisy(7:10, k+1) = x_noisy(7:10, k+1) / norm(x_noisy(7:10, k+1));  % Normalize quaternion
    
    % Generate measurements
    % 1. Position from GPS (with noise)
    z_pos = x_true(1:3, k+1);
    
    % 2. Magnetometer reading (depends on position and attitude)
    % Convert quaternion to rotation matrix
    q0 = quat(4); q1 = quat(1); q2 = quat(2); q3 = quat(3);
    R_q = [
        1-2*(q2^2+q3^2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2);
        2*(q1*q2+q0*q3), 1-2*(q1^2+q3^2), 2*(q2*q3-q0*q1);
        2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1^2+q2^2)
    ];
    
    % Simple magnetic field model (depends on position)
    r_unit = pos / norm(pos);
    scale_factor = (orbital_radius/norm(pos))^3;
    B_eci = scale_factor * (3 * dot(r_unit, [0;0;1]) * r_unit - [0;0;1]) .* B_earth;  % Element-wise multiplication
    z_mag = R_q' * B_eci;  % Magnetic field in body frame
    
    % 3. Gyroscope reading (angular velocity)
    z_gyro = omega;
    
    % 4. Sun sensor reading
    sun_body = R_q' * sun_direction;  % Sun direction in body frame
    sun_body = sun_body / norm(sun_body);
    z_sun = sun_body;
    
    % Combine measurements and add noise
    z_true = [z_pos; z_mag; z_gyro; z_sun];
    z_noisy(:, k+1) = z_true + sqrtm(R) * randn(m_dim, 1);
    
    % Normalize sun sensor measurement after adding noise
    z_noisy(10:12, k+1) = z_noisy(10:12, k+1) / norm(z_noisy(10:12, k+1));
end

%% UKF Parameters
n = 13;  % State dimension
m = 12;  % Measurement dimension
p = 6;   % Input dimension

alpha = 1e-3;  % Spread parameter
beta = 2;      % Prior knowledge parameter (2 is optimal for Gaussian)
kappa = 0;     % Secondary spread parameter
lambda = alpha^2 * (n + kappa) - n;  % Scaling parameter

% Calculate weights
Wm = ones(2*n+1, 1) / (2*(n+lambda));
Wm(1) = lambda / (n + lambda);
Wc = Wm;
Wc(1) = Wc(1) + (1 - alpha^2 + beta);

%% UKF Implementation
x_ukf = zeros(13, N);
P_ukf = eye(13);  % Initial error covariance

% Initialize UKF state with noisy initial state
x_ukf(:,1) = x_noisy(:,1);

% Add small regularization to ensure positive definiteness
reg_value = 1e-8;

for k = 1:N-1
    % Current state and covariance
    x = x_ukf(:,k);
    
    % Normalize quaternion part of state
    x(7:10) = x(7:10) / norm(x(7:10));
    
    % 1. Generate sigma points
    % Ensure P_ukf is positive definite
    P_reg = P_ukf + reg_value * eye(n);
    
    % Try to compute Cholesky decomposition, use SVD as fallback
    try
        sP = chol((n+lambda)*P_reg, 'lower');  % Square root of covariance
    catch
        % Fallback to SVD-based square root if Cholesky fails
        [U, S, ~] = svd(P_reg);
        sP = U * sqrt(S) * sqrt(n+lambda);
    end
    
    X = zeros(n, 2*n+1);
    X(:,1) = x;
    for i = 1:n
        X(:,i+1) = x + sP(:,i);
        X(:,i+n+1) = x - sP(:,i);
        
        % Normalize quaternion components of sigma points
        X(7:10,i+1) = X(7:10,i+1) / norm(X(7:10,i+1));
        X(7:10,i+n+1) = X(7:10,i+n+1) / norm(X(7:10,i+n+1));
    end
    
    % 2. Propagate sigma points through process model
    Y = zeros(n, 2*n+1);
    for i = 1:2*n+1
        % Extract state components
        pos_i = X(1:3,i);
        vel_i = X(4:6,i);
        quat_i = X(7:10,i);
        omega_i = X(11:13,i);
        
        % Orbit dynamics
        r_norm = norm(pos_i);
        grav_accel = -mu * pos_i / r_norm^3;
        pos_next = pos_i + vel_i * dt;
        vel_next = vel_i + (grav_accel + u(1:3,k)/cubesat.mass) * dt;
        
        % Attitude dynamics (quaternion)
        omega_matrix = [
            0, -omega_i(3), omega_i(2);
            omega_i(3), 0, -omega_i(1);
            -omega_i(2), omega_i(1), 0
        ];
        quat_dot = 0.5 * [
            -omega_matrix, omega_i;
            -omega_i', 0
        ] * quat_i;
        quat_next = quat_i + quat_dot * dt;
        quat_next = quat_next / norm(quat_next);  % Normalize quaternion
        
        % Angular velocity dynamics
        I = cubesat.inertia;
        omega_dot = I \ (u(4:6,k) - cross(omega_i, I * omega_i));
        omega_next = omega_i + omega_dot * dt;
        
        % Store propagated sigma point
        Y(:,i) = [pos_next; vel_next; quat_next; omega_next];
    end
    
    % 3. Calculate predicted mean and covariance
    x_pred = zeros(n, 1);
    for i = 1:2*n+1
        x_pred = x_pred + Wm(i) * Y(:,i);
    end
    
    % Normalize quaternion part of predicted state
    x_pred(7:10) = x_pred(7:10) / norm(x_pred(7:10));
    
    P_pred = Q;  % Start with process noise
    for i = 1:2*n+1
        % Apply quaternion difference properly
        if dot(Y(7:10,i), x_pred(7:10)) < 0
            Y(7:10,i) = -Y(7:10,i);  % Flip quaternion to avoid double-covering
        end
        
        state_diff = Y(:,i) - x_pred;
        P_pred = P_pred + Wc(i) * state_diff * state_diff';
    end
    
    % Ensure P_pred is symmetric
    P_pred = (P_pred + P_pred') / 2;
    
    % Add small regularization if needed
    [~, p_flag] = chol(P_pred);
    if p_flag > 0
        P_pred = P_pred + reg_value * eye(n);
    end
    
    % 4. Propagate sigma points through measurement model
    Z = zeros(m, 2*n+1);
    for i = 1:2*n+1
        % Extract state components
        pos_i = Y(1:3,i);
        quat_i = Y(7:10,i);
        omega_i = Y(11:13,i);
        
        % Convert quaternion to rotation matrix
        q0 = quat_i(4); q1 = quat_i(1); q2 = quat_i(2); q3 = quat_i(3);
        R_q = [
            1-2*(q2^2+q3^2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2);
            2*(q1*q2+q0*q3), 1-2*(q1^2+q3^2), 2*(q2*q3-q0*q1);
            2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1^2+q2^2)
        ];
        
        % GPS measurement
        z_pos = pos_i;
        
        % Magnetometer measurement
        r_unit = pos_i / norm(pos_i);
        scale_factor = (orbital_radius/norm(pos_i))^3;
        B_eci = scale_factor * (3 * dot(r_unit, [0;0;1]) * r_unit - [0;0;1]) .* B_earth;  % Element-wise multiplication
        z_mag = R_q' * B_eci;
        
        % Gyroscope measurement
        z_gyro = omega_i;
        
        % Sun sensor measurement
        sun_body = R_q' * sun_direction;
        sun_body = sun_body / norm(sun_body);
        z_sun = sun_body;
        
        % Store predicted measurement
        Z(:,i) = [z_pos; z_mag; z_gyro; z_sun];
    end
    
    % 5. Calculate predicted measurement mean and covariance
    z_pred = zeros(m, 1);
    for i = 1:2*n+1
        z_pred = z_pred + Wm(i) * Z(:,i);
    end
    
    % Normalize sun direction component
    z_pred(10:12) = z_pred(10:12) / norm(z_pred(10:12));
    
    Pzz = R;  % Start with measurement noise
    for i = 1:2*n+1
        meas_diff = Z(:,i) - z_pred;
        Pzz = Pzz + Wc(i) * meas_diff * meas_diff';
    end
    
    % Ensure Pzz is symmetric and positive definite
    Pzz = (Pzz + Pzz') / 2;
    [~, p_flag] = chol(Pzz);
    if p_flag > 0
        Pzz = Pzz + reg_value * eye(m);
    end
    
    % 6. Calculate cross-correlation matrix
    Pxz = zeros(n, m);
    for i = 1:2*n+1
        % Apply quaternion difference properly for state difference
        y_diff = Y(:,i) - x_pred;
        if dot(Y(7:10,i), x_pred(7:10)) < 0
            y_diff(7:10) = -y_diff(7:10);
        end
        
        z_diff = Z(:,i) - z_pred;
        Pxz = Pxz + Wc(i) * y_diff * z_diff';
    end
    
    % 7. Kalman gain with safeguard against matrix inversion issues
    try
        K = Pxz / Pzz;
    catch
        % Fallback using pseudoinverse if regular inversion fails
        K = Pxz * pinv(Pzz);
    end
    
    % 8. Update state and covariance
    x_ukf(:,k+1) = x_pred + K * (z_noisy(:,k+1) - z_pred);
    P_ukf = P_pred - K * Pzz * K';
    
    % Ensure P_ukf remains symmetric and positive definite
    P_ukf = (P_ukf + P_ukf') / 2;
    
    % Add small regularization if needed
    [~, p_flag] = chol(P_ukf);
    if p_flag > 0
        P_ukf = P_ukf + reg_value * eye(n);
    end
    
    % Normalize quaternion
    x_ukf(7:10,k+1) = x_ukf(7:10,k+1) / norm(x_ukf(7:10,k+1));
end

%% Performance Evaluation
% Calculate RMS error for raw measurements (treated as if they were state estimates)
rms_error_raw = zeros(1, 6);  % Position (3) and Angular velocity (3)

% Calculate RMS errors for position and angular velocity
for i = 1:3
    rms_error_raw(i) = sqrt(mean((z_noisy(i,:) - x_true(i,:)).^2));
    rms_error_raw(i+3) = sqrt(mean((z_noisy(i+6,:) - x_true(i+10,:)).^2));
end

% Calculate RMS error for UKF
rms_error_ukf = zeros(1, 6);
for i = 1:3
    rms_error_ukf(i) = sqrt(mean((x_ukf(i,:) - x_true(i,:)).^2));
    rms_error_ukf(i+3) = sqrt(mean((x_ukf(i+10,:) - x_true(i+10,:)).^2));
end

% Calculate improvement percentage
improvement = 100 * (1 - rms_error_ukf ./ rms_error_raw);

% Display results
fprintf('\n----- Performance Evaluation -----\n');
fprintf('RMS Error for Raw Measurements:\n');
fprintf('  Position XYZ: [%.2f, %.2f, %.2f] meters\n', rms_error_raw(1), rms_error_raw(2), rms_error_raw(3));
fprintf('  Angular Velocity: [%.4f, %.4f, %.4f] rad/s\n', rms_error_raw(4), rms_error_raw(5), rms_error_raw(6));

fprintf('\nRMS Error for UKF Estimates:\n');
fprintf('  Position XYZ: [%.2f, %.2f, %.2f] meters\n', rms_error_ukf(1), rms_error_ukf(2), rms_error_ukf(3));
fprintf('  Angular Velocity: [%.4f, %.4f, %.4f] rad/s\n', rms_error_ukf(4), rms_error_ukf(5), rms_error_ukf(6));

fprintf('\nImprovement Percentage:\n');
fprintf('  Position XYZ: [%.2f%%, %.2f%%, %.2f%%]\n', improvement(1), improvement(2), improvement(3));
fprintf('  Angular Velocity: [%.2f%%, %.2f%%, %.2f%%]\n', improvement(4), improvement(5), improvement(6));

%% Plotting Results
% Only plot a portion of the data for clarity
plot_range = 1:100:N; % Plot every 100th point

% Position Plots (XYZ)
figure('Name', 'Position Estimates', 'Position', [100, 100, 1200, 800]);
states = {'X Position (m)', 'Y Position (m)', 'Z Position (m)'};
for i = 1:3
    subplot(3, 1, i);
    plot(time(plot_range), x_true(i,plot_range), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(time(plot_range), z_noisy(i,plot_range), 'r-', 'MarkerSize', 5);
    plot(time(plot_range), x_ukf(i,plot_range), 'k-', 'LineWidth', 1);
    grid on;
    xlabel('Time (s)');
    ylabel(states{i});
    title(sprintf('%s: Pure, Noisy, and Filtered', states{i}));
    legend('True', 'Noisy', 'UKF Filtered');
end

% Velocity Plots (XYZ)
figure('Name', 'Velocity Estimates', 'Position', [150, 150, 1200, 800]);
states = {'X Velocity (m/s)', 'Y Velocity (m/s)', 'Z Velocity (m/s)'};
for i = 1:3
    subplot(3, 1, i);
    plot(time(plot_range), x_true(i+3,plot_range), 'b-', 'LineWidth', 1.5);
    hold on;
    % No direct velocity measurement in our sensor model
    plot(time(plot_range), x_ukf(i+3,plot_range), 'k-', 'LineWidth', 1);
    grid on;
    xlabel('Time (s)');
    ylabel(states{i});
    title(sprintf('%s: Pure and Filtered', states{i}));
    legend('True', 'UKF Filtered');
end

% Angular Velocity Plots
figure('Name', 'Angular Velocity Estimates', 'Position', [200, 200, 1200, 800]);
states = {'X Angular Velocity (rad/s)', 'Y Angular Velocity (rad/s)', 'Z Angular Velocity (rad/s)'};
for i = 1:3
    subplot(3, 1, i);
    plot(time(plot_range), x_true(i+10,plot_range), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(time(plot_range), z_noisy(i+6,plot_range), 'r-', 'MarkerSize', 5);
    plot(time(plot_range), x_ukf(i+10,plot_range), 'k-', 'LineWidth', 1);
    grid on;
    xlabel('Time (s)');
    ylabel(states{i});
    title(sprintf('%s: Pure, Noisy, and Filtered', states{i}));
    legend('True', 'Noisy', 'UKF Filtered');
end

% Plot 3D orbit trajectory
figure('Name', '3D Orbit Trajectory', 'Position', [250, 250, 800, 800]);
plot3(x_true(1,plot_range), x_true(2,plot_range), x_true(3,plot_range), 'b-', 'LineWidth', 2);
hold on;
plot3(x_ukf(1,plot_range), x_ukf(2,plot_range), x_ukf(3,plot_range), 'k-', 'LineWidth', 1);
plot3(z_noisy(1,plot_range), z_noisy(2,plot_range), z_noisy(3,plot_range), 'r-', 'MarkerSize', 5);

% Draw Earth (simplified)
[X,Y,Z] = sphere(50);
X = X * radius_earth;
Y = Y * radius_earth;
Z = Z * radius_earth;
surf(X, Y, Z, 'FaceColor', 'blue', 'EdgeColor', 'none', 'FaceAlpha', 0.3);

grid on; box on;
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('Orbit Trajectory: True, Noisy, and Filtered');
legend('True Trajectory', 'UKF Filtered', 'Noisy Measurements', 'Earth');
axis equal;
view(30, 30);