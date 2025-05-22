clear; clc; close all;

%% Initial values
m = 1;
k = 5;
b = 0.5;

dt = 0.1;
t = 0:dt:20;

Q = diag([0.001, 0.01]);  % process disturbance (2 inputs)
wd = 0.5*randn(2,size(t,2));
R = 0.5;  % sensor noise (1 sensed output)

%% Continuous system model
A = [0 1;
    -k/m -b/m];
B = [0; 1/m];
C = [1 0];

%% Discretize the system
sys = ss(A,B,C,0);
sysd = c2d(sys, dt);  % Discrete-time system

Ad = sysd.A;
Bd = sysd.B;
Cd = sysd.C;

%% Input signal
u = zeros(1, length(t));
u(10:12) = 5;  % small pulse

%% Simulation loop
x = zeros(2, length(t));
y = zeros(2, length(t));         % measuring both states
xnoisy = zeros(2, length(t));
ynoisy = zeros(1, length(t));
xhat = zeros(2, length(t));
Pstate = zeros(2, 2, length(t));
K = zeros(2, length(t));
w = zeros(2, length(t));         % process noise w_k
v = zeros(1, length(t));         % measurement noise v_k

for i = 2:size(t,2)
    % model without noise & disturbance
    x(:,i) = Ad*x(:,i-1) + Bd*u(i);
    y(:,i) = [1 0; 0 1]*x(:,i);

    % model with noise & disturbance
    w(:,i) = mvnrnd([0 0], Q)';
    v(i) = sqrt(R)*randn();
    xnoisy(:,i) = Ad*xnoisy(:,i-1) + Bd*u(i) + wd(:,i);
    ynoisy(i) = Cd*xnoisy(:,i) + v(i);

    [xhat(:,i), Pstate(:,:,i), K(:,i)] = KalmanFilter(ynoisy(i),u(i),Ad,Bd,Cd,Q,R);
end

%% Plotting
% Figure 1: States
figure('Position', [100, 150, 1100, 700]);

% Position plot
subplot(2, 1, 1);
plot(t, x(1, :), 'b', 'LineWidth', 2, 'DisplayName', 'True');
hold on;
plot(t, xnoisy(1, :), 'r', 'LineWidth', 1, 'DisplayName', 'Noisy');
plot(t, xhat(1, :), 'k', 'LineWidth', 1.5, 'DisplayName', 'Filtered');
xlabel('Time [s]');
ylabel('Position [m]');
title('Position');
legend('Location', 'best');
grid on;

% Velocity plot
subplot(2, 1, 2);
plot(t, x(2, :), 'b', 'LineWidth', 2, 'DisplayName', 'True');
hold on;
plot(t, xnoisy(2, :), 'r', 'LineWidth', 1, 'DisplayName', 'Noisy');
plot(t, xhat(2, :), 'k', 'LineWidth', 1.5, 'DisplayName', 'Filtered');
xlabel('Time [s]');
ylabel('Velocity [m/s]');
title('Velocity');
legend('Location', 'best');
grid on;

function [xhat, Pstate, K] = KalmanFilter(y,u,A,B,C,Q,R)

persistent x P intial

if isempty(intial)
    x = zeros(size(A,1), 1);
    K = zeros(size(A,1), 1);
    P = eye(size(A,1));
    intial = 1;
end

% predection step
x = A*x + B*u;
P = A*P*A' + Q;

% update step
K = P*C' / (C*P*C' + R);
x = x + K*(y - C*x);
P = (eye(size(A,1)) - K*C) * P;

xhat = x;
Pstate = P;

end