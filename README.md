# MATLAB State Estimation with Kalman Filters (KF, EKF, UKF)

This repository contains MATLAB implementations of state estimation algorithms (Kalman Filter, Extended Kalman Filter, Unscented Kalman Filter) applied to a variety of dynamic systems. Each script demonstrates the use of these filters for different physical models, including simulation of noisy measurements, process disturbances, and filter performance evaluation.

## Table of Contents
- [1. KF_DronePositionTracking.m](#1-kf_dronepositiontrackingm)
- [3. KF_MassSpringDamper.m](#3-kf_massspringdamperm)
- [4. KF_DCMotor.m](#4-kf_dcmotorm)
- [5. EKF_CartPendulum.m](#5-ekf_cartpendulumm)
- [6. UKF_CartPendulum.m](#6-ukf_cartpendulumm)
- [7. UKF_2UCubeSat.m](#7-ukf_2ucubesatm)
- [8. UKF_3DRobotLocalization.m](#8-ukf_3drobotlocalizationm)

---

## 1. KF_DronePositionTracking.m
**Description:**
- Implements a Kalman Filter for tracking the position of a drone.
- **States:** Drone position and velocity (typically in 2D or 3D: [x, y, z, vx, vy, vz]).
- **Inputs:** Control accelerations or forces applied to the drone.
- **Measurements:** Noisy position measurements from sensors (e.g., GPS).
- Demonstrates how the Kalman Filter can estimate the true position from noisy data.
- Useful for understanding basic KF concepts in a 2D/3D tracking context.

## 2. KF_CartPendulumContinous.m
**Description:**
- Applies the Kalman Filter to a linearized cart-pendulum (inverted pendulum) system.
- **States:** [cart position, cart velocity, pendulum angle, pendulum angular velocity].
- **Inputs:** Force applied to the cart.
- **Measurements:** Noisy cart position measurement.
- Simulates the system with process disturbances and sensor noise.
- Shows how the filter estimates all system states from noisy measurements.
- Includes plots comparing true, measured, and estimated states, as well as Kalman gain evolution.

## 3. KF_MassSpringDamper.m
**Description:**
- Demonstrates Kalman Filtering for a mass-spring-damper system.
- **States:** [position, velocity] of the mass.
- **Inputs:** External force applied to the mass.
- **Measurements:** Noisy position measurement.
- Simulates both the true and noisy system responses to an input force.
- The filter estimates position and velocity from noisy position measurements.
- Includes a custom Kalman filter function and visualizes the improvement over noisy data.

## 4. KF_DCMotor.m
**Description:**
- Implements a Kalman Filter for a DC motor position control system.
- **States:** [rotor position, angular velocity].
- **Inputs:** Voltage applied to the motor.
- **Measurements:** Noisy position (rotor angle) measurement.
- Models the motor's dynamics and simulates both process and measurement noise.
- The filter estimates position and velocity from noisy position measurements.
- Includes RMS error analysis and plots to show the effectiveness of filtering.

## 5. EKF_CartPendulum.m
**Description:**
- Implements an Extended Kalman Filter (EKF) for a nonlinear cart-pendulum system.
- **States:** [cart position, cart velocity, pendulum angle, pendulum angular velocity].
- **Inputs:** Force applied to the cart.
- **Measurements:** Noisy measurements of cart position and pendulum angle.
- Simulates the true nonlinear dynamics and noisy measurements.
- The EKF linearizes the system at each step and estimates all states.
- Includes detailed plots for all states and estimation error analysis.

## 6. UKF_CartPendulum.m
**Description:**
- Applies the Unscented Kalman Filter (UKF) to the nonlinear cart-pendulum system.
- **States:** [cart position, cart velocity, pendulum angle, pendulum angular velocity].
- **Inputs:** Force applied to the cart and external disturbance.
- **Measurements:** Noisy cart position and pendulum angle (nonlinear measurement model).
- Uses the unscented transform to handle nonlinearities in both process and measurement models.
- Simulates process and measurement noise, and compares UKF estimates to true and noisy measurements.
- Provides error metrics and visualizations for all states.

## 7. UKF_2UCubeSat.m
**Description:**
- Implements a UKF for a 2U CubeSat with multiple subsystems.
- **States:** [position (3), velocity (3), attitude quaternion (4), angular velocity (3)] — 13 states in total.
- **Inputs:** External forces and torques acting on the satellite.
- **Measurements:** GPS position, magnetometer, gyroscope, and sun sensor readings (all noisy).
- Simulates orbital motion, attitude dynamics, and sensor measurements with noise and disturbances.
- The UKF estimates the full 13-dimensional state vector.
- Includes RMS error analysis and plots for position, velocity, and angular velocity.

## 8. UKF_3DRobotLocalization.m
**Description:**
- Demonstrates UKF-based localization for a 3D robot using nonlinear range measurements to fixed beacons.
- **States:** [x, y, z, vx, vy, vz, roll, pitch, yaw] — position, velocity, and orientation.
- **Inputs:** Forces and moments applied to the robot (6D control input).
- **Measurements:** Noisy range measurements to multiple fixed beacons (nonlinear function of position).
- Models the robot's 3D motion and orientation, and simulates noisy range measurements.
- The UKF estimates position, velocity, and orientation (roll, pitch, yaw).
- Visualizes the true and estimated 3D trajectories, as well as state evolution over time.

---

## Usage
- Open any script in MATLAB and run it to see the simulation and filtering results.
- Each script is self-contained and includes comments for clarity.
- Plots are generated to compare true, noisy, and filtered states.

## License
This repository is licensed under the MIT License.
