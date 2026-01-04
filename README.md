# CKATool fork

This repository is a fork of the original CKATool: https://github.com/khaira/ckatool.

## Changes in this fork

- Make Rerun an optional dependency, allowing the tool to be used as a library for offline computation of kinematics
- Modify the jerk computation by computing derivatives per iteration
- Add the trunk angle computation to the neck object

## Installation

- Download this repository
- Install it with:
```sh
pip install -e .
```

## Offline usage example

```python
import numpy
import ckatool

# Set the coordinates for all iterations
iteration = [...]
timestamp = [...]
neck_x = [...]
neck_y = [...]
neck_z = [...]
...

# Set the side (right or left) for shoulder, elbow and wrist
side = ...

# Create the CKATool objects
neck = ckatool.Neck(timestamp, neck_x, neck_y, neck_z, iteration)
hip = ckatool.Hip(timestamp, hip_x, hip_y, hip_z, iteration)
shoulder = ckatool.Shoulder(timestamp, shoulder_x, shoulder_y, shoulder_z, iteration, side)
elbow = ckatool.Elbow(timestamp, elbow_x, elbow_y, elbow_z, iteration, side)
wrist = ckatool.Wrist(timestamp, wrist_x, wrist_y, wrist_z, iteration, side)
end_effector = ckatool.EndEffector(timestamp, end_effector_x, end_effector_y, end_effector_z, iteration)
target = ckatool.Target(timestamp, target_x, target_y, target_z, iteration)

# Set the reference vector for the trunk angle
# The reference vector depends on the coordinate system
# Example for MediaPipe, a vertical vector is [0, -1, 0]
reference_vector = ...

# Compute the kinematics
# Use the calculate_* functions instead of the visualise_* functions
neck.calculate_trunk_angle(hip, reference_vector)

shoulder.calculate_speed_profile(elbow, neck, hip)
shoulder.calculate_acceleration_profile()
shoulder.calculate_zero_crossings()
shoulder.count_number_of_velocity_peaks()
shoulder.calculate_ratio_mean_peak_velocity()
shoulder.calculate_mean_velocity()
shoulder.calculate_peak_velocity()
shoulder.calculate_sparc()
shoulder.calculate_jerk()
shoulder.calculate_movement_time()
shoulder.calculate_percentage_time_to_peak_velocity()

elbow.calculate_speed_profile(wrist, shoulder)
elbow.calculate_acceleration_profile()
elbow.calculate_zero_crossings()
elbow.count_number_of_velocity_peaks()
elbow.calculate_ratio_mean_peak_velocity()
elbow.calculate_mean_velocity()
elbow.calculate_peak_velocity()
elbow.calculate_sparc()
elbow.calculate_jerk()
elbow.calculate_movement_time()
elbow.calculate_percentage_time_to_peak_velocity()

wrist.calculate_speed_profile()
wrist.calculate_acceleration_profile()
wrist.calculate_zero_crossings()
wrist.count_number_of_velocity_peaks()
wrist.calculate_ratio_mean_peak_velocity()
wrist.calculate_mean_velocity()
wrist.calculate_peak_velocity()
wrist.calculate_sparc()
wrist.calculate_jerk()
wrist.calculate_movement_time()
wrist.calculate_percentage_time_to_peak_velocity()

end_effector.calculate_target_error_distance(target)
end_effector.calculate_hand_path_ratio(target)

# Get the kinematics for one iteration
iteration_number = ...

shoulder_number_of_velocity_peaks = shoulder.number_of_velocity_peaks[iteration_number]
elbow_number_of_velocity_peaks = elbow.number_of_velocity_peaks[iteration_number]
wrist_number_of_velocity_peaks = wrist.number_of_velocity_peaks[iteration_number]

shoulder_ratio_mean_peak_velocity = shoulder.ratio_mean_peak_velocity[iteration_number]
elbow_ratio_mean_peak_velocity = elbow.ratio_mean_peak_velocity[iteration_number]
wrist_ratio_mean_peak_velocity = wrist.ratio_mean_peak_velocity[iteration_number]

shoulder_mean_velocity = shoulder.mean_velocity[iteration_number]
elbow_mean_velocity = elbow.mean_velocity[iteration_number]
wrist_mean_velocity = wrist.mean_velocity[iteration_number]

shoulder_peak_velocity = shoulder.peak_velocity[iteration_number]
elbow_peak_velocity = elbow.peak_velocity[iteration_number]
wrist_peak_velocity = wrist.peak_velocity[iteration_number]

shoulder_sparc = shoulder.sparc[iteration_number]
elbow_sparc = elbow.sparc[iteration_number]
wrist_sparc = wrist.sparc[iteration_number]

shoulder_jerk = shoulder.jerk[iteration_number]
elbow_jerk = elbow.jerk[iteration_number]
wrist_jerk = wrist.jerk[iteration_number]

shoulder_movement_time = shoulder.movement_time[iteration_number]
elbow_movement_time = elbow.movement_time[iteration_number]
wrist_movement_time = wrist.movement_time[iteration_number]

shoulder_percentage_time_to_peak_velocity = shoulder.percentage_time_to_peak_velocity[iteration_number]
elbow_percentage_time_to_peak_velocity = elbow.percentage_time_to_peak_velocity[iteration_number]
wrist_percentage_time_to_peak_velocity = wrist.percentage_time_to_peak_velocity[iteration_number]

trunk_rom = numpy.nanmax(neck.trunk_angle) - numpy.nanmin(neck.trunk_angle)
shoulder_rom = numpy.nanmax(shoulder.angle) - numpy.nanmin(shoulder.angle)
elbow_rom = numpy.nanmax(elbow.angle) - numpy.nanmin(elbow.angle)

target_error_distance = end_effector.target_error_distance[iteration_number]
hand_path_ratio = end_effector.hand_path_ratio[iteration_number]
```