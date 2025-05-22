clc; close all; clear;

%% initialization 
m = 1;
M = 5;
L = 2;
g = -10;
d = 1;

s = -1; % pendulum up s=1

%% system ss model
A = [0 1 0 0;
    0 -d/M -m*g/M 0;
    0 0 0 1;
    0 -s*d/M*L -s*(M+m)*g/M*L 0];

B = [0; 1/M; 0; s/M*L];

C = [1 0 0 0];

D = zeros(size(C,1),size(B,2));

%% Augmented sys model
Vd = 0.1*eye(4);        % states disturbance
Vn = 1;                 % sensor noise

BF = [B Vd 0*B];        % B_aug = Bu + Vd + 0*Vn

DD = [0 0 0 0 0 Vn];    % D = 0*u + 0*Vd + Vn  

sysC = ss(A,BF,C, DD);  % Sys with noise & distubance

sysFullOutput = ss(A,BF,eye(4),zeros(4,size(BF,2))); % sys without noise

%% SImulating linearized sys + disturbance + noise in down mode
dt = 0.01;
t = dt:dt:50;

uDIST = randn(4,size(t,2));
uNOISE = randn(size(t));

u = 0*t;
u(100:120) = 100;
u(1500:1520) = -100;

uAUG = [u; uDIST; uNOISE];


[y,tt] = lsim(sysC,uAUG,t);
plot(tt,y);

%% Kalman Filter (Mine)

sys = ss(A,B,C,0);
sysd = c2d(sys,dt);
Ad = sysd.A;
Bd = sysd.B;
Cd = sysd.C;

x = zeros(4,size(t,2));
P = zeros(4,4,size(t,2));
K = zeros(4,size(t,2));

P(:,:,1) = 1 * eye(4);  % Initial uncertainty

for i = 2:size(t,2)

    % Predict
x(:,i) = Ad*x(:,i-1) + Bd*u(i);
P(:,:,i) = Ad*P(:,:,i-1)*Ad' + Vd;

    % Kalman Gain
K(:,i) = P(:,:,i)*C' * inv(C*P(:,:,i)*C' + Vn);

    % Update
x(:,i) = x(:,i) + K(:,i)*(y(i) - C*x(:,i));
P(:,:,i) = (eye(4) - K(:,i)*C)*P(:,:,i);

end

% figure;
% plot(t, K'); legend('K1','K2','K3','K4'); title('Kalman Gains');


%% ploting the x state without noise
% sysFullOutputd = c2d(sysFullOutput,dt);
[xtrue,t] = lsim(sysFullOutput,uAUG,t);
x = x';
hold on
plot(t,xtrue(:,1),'b','LineWidth',2.0);

%% KF filtering the noise of x state
% [x,t] = lsim(sysKF, [u; y'],t);
% figure
plot(t,x(:,1),'k--','LineWidth',2.0);

%% KF estimation of the other states
figure
plot(t,xtrue,'-',t,x,'--','LineWidth',2.0)

%% 
figure;
plot(t, K(1,:), t, K(2,:), t, K(3,:), t, K(4,:));
title('Kalman Gain Evolution'); legend('K1','K2','K3','K4');
