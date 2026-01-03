import argparse

import polars as pl
import rerun as rr
import rerun.blueprint as rrb

from .lib.limb import (
    Elbow,
    EndEffector,
    Hip,
    Neck,
    Shoulder,
    Target,
    Wrist,
    calculate_percentage_time_to_peak_velocity,
    count_velocity_peaks_per_iteration,
    visualise_barchart_per_iteration,
    visualise_max_acceleration,
    visualise_max_angle,
    visualise_max_speed_angular,
    visualise_max_zero_crossings,
    visualise_speed_profile_iteration,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tool for analyzing upper limb movement using 3D motion tracking data."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input file")
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "CKATool")
    vs = Visualiser(args.input)
    rr.send_blueprint(vs.default_viewport())

    vs.process()
    rr.script_teardown(args)

    return 0


class Visualiser:
    def __init__(self, input_file: str):
        self._df = pl.read_csv(
            input_file,
            schema_overrides={
                "iteration": pl.Int32,
            },
        )

    def process(self):
        ts = self._df["timestamp"]
        neck = None
        iteration = self._df["iteration"]
        if "neck_x" in self._df.columns:
            neck = Neck(ts, self._df["neck_x"], self._df["neck_y"], self._df["neck_z"], iteration)
        hip = None
        if "hip_x" in self._df.columns:
            hip = Hip(ts, self._df["hip_x"], self._df["hip_y"], self._df["hip_z"], iteration)
        left_shoulder = None
        if "left_shoulder_x" in self._df.columns:
            left_shoulder = Shoulder(
                ts,
                self._df["left_shoulder_x"],
                self._df["left_shoulder_y"],
                self._df["left_shoulder_z"],
                iteration,
                "left",
            )
        right_shoulder = None
        if "right_shoulder_x" in self._df.columns:
            right_shoulder = Shoulder(
                ts,
                self._df["right_shoulder_x"],
                self._df["right_shoulder_y"],
                self._df["right_shoulder_z"],
                iteration,
                "right",
            )
        right_shoulder = None
        if "shoulder_x" in self._df.columns:
            right_shoulder = Shoulder(
                ts,
                self._df["shoulder_x"],
                self._df["shoulder_y"],
                self._df["shoulder_z"],
                iteration,
                "right",
            )
        left_elbow = None
        if "left_elbow_x" in self._df.columns:
            left_elbow = Elbow(
                ts, self._df["left_elbow_x"], self._df["left_elbow_y"], self._df["left_elbow_z"], iteration, "left"
            )
        right_elbow = None
        if "right_elbow_x" in self._df.columns:
            right_elbow = Elbow(
                ts, self._df["right_elbow_x"], self._df["right_elbow_y"], self._df["right_elbow_z"], iteration, "right"
            )
        right_elbow = None
        if "elbow_x" in self._df.columns:
            right_elbow = Elbow(
                ts, self._df["elbow_x"], self._df["elbow_y"], self._df["elbow_z"], iteration, "right"
            )
        left_wrist = None
        if "left_wrist_x" in self._df.columns:
            left_wrist = Wrist(
                ts, self._df["left_wrist_x"], self._df["left_wrist_y"], self._df["left_wrist_z"], iteration, "left"
            )
        right_wrist = None
        if "right_wrist_x" in self._df.columns:
            right_wrist = Wrist(
                ts, self._df["right_wrist_x"], self._df["right_wrist_y"], self._df["right_wrist_z"], iteration, "right"
            )
        right_wrist = None
        if "wrist_x" in self._df.columns:
            right_wrist = Wrist(
                ts, self._df["wrist_x"], self._df["wrist_y"], self._df["wrist_z"], iteration, "right"
            )

        # if there is target points data and endeffector
        target = None
        if "target_x" in self._df.columns:
            target = Target(ts, self._df["target_x"], self._df["target_y"], self._df["target_z"], iteration)
        end_effector = None
        if "end_effector_x" in self._df.columns:
            end_effector = EndEffector(
                ts, self._df["end_effector_x"], self._df["end_effector_y"], self._df["end_effector_z"], iteration
            )

        if neck:
            neck.visualise_3d_data()
        if hip:
            hip.visualise_3d_data()
        if left_shoulder:
            left_shoulder.visualise_3d_data(neck, hip)
            left_shoulder.visualise_speed_profile(left_elbow, neck, hip)
            left_shoulder.visualise_acceleration_profile()
            left_shoulder.visualise_zero_crossings()
            left_shoulder.count_number_of_velocity_peaks()
            left_shoulder.calculate_ratio_mean_peak_velocity()
            left_shoulder.calculate_mean_velocity()
            left_shoulder.calculate_peak_velocity()
            left_shoulder.calculate_sparc()
            left_shoulder.calculate_jerk()
            left_shoulder.calculate_movement_time()
            left_shoulder.calculate_percentage_time_to_peak_velocity()
        if right_shoulder:
            right_shoulder.visualise_3d_data(neck, hip)
            right_shoulder.visualise_speed_profile(right_elbow, neck, hip)
            right_shoulder.visualise_acceleration_profile()
            right_shoulder.visualise_zero_crossings()
            right_shoulder.count_number_of_velocity_peaks()
            right_shoulder.calculate_ratio_mean_peak_velocity()
            right_shoulder.calculate_mean_velocity()
            right_shoulder.calculate_peak_velocity()
            right_shoulder.calculate_sparc()
            right_shoulder.calculate_jerk()
            right_shoulder.calculate_movement_time()
            right_shoulder.calculate_percentage_time_to_peak_velocity()
        if left_elbow:
            left_elbow.visualise_3d_data(left_shoulder)
            left_elbow.visualise_speed_profile(left_wrist, left_shoulder)
            left_elbow.visualise_acceleration_profile()
            left_elbow.visualise_zero_crossings()
            left_elbow.count_number_of_velocity_peaks()
            left_elbow.calculate_ratio_mean_peak_velocity()
            left_elbow.calculate_mean_velocity()
            left_elbow.calculate_peak_velocity()
            left_elbow.calculate_sparc()
            left_elbow.calculate_jerk()
            left_elbow.calculate_movement_time()
            left_elbow.calculate_percentage_time_to_peak_velocity()
        if right_elbow:
            right_elbow.visualise_3d_data(right_shoulder)
            right_elbow.visualise_speed_profile(right_wrist, right_shoulder)
            right_elbow.visualise_acceleration_profile()
            right_elbow.visualise_zero_crossings()
            right_elbow.count_number_of_velocity_peaks()
            right_elbow.calculate_ratio_mean_peak_velocity()
            right_elbow.calculate_mean_velocity()
            right_elbow.calculate_peak_velocity()
            right_elbow.calculate_sparc()
            right_elbow.calculate_jerk()
            right_elbow.calculate_movement_time()
            right_elbow.calculate_percentage_time_to_peak_velocity()
        if left_wrist:
            left_wrist.visualise_3d_data(left_elbow)
            left_wrist.visualise_speed_profile()
            left_wrist.visualise_acceleration_profile()
            left_wrist.visualise_zero_crossings()
            left_wrist.count_number_of_velocity_peaks()
            left_wrist.calculate_ratio_mean_peak_velocity()
            left_wrist.calculate_mean_velocity()
            left_wrist.calculate_peak_velocity()
            left_wrist.calculate_sparc()
            left_wrist.calculate_jerk()
            left_wrist.calculate_movement_time()
            left_wrist.calculate_percentage_time_to_peak_velocity()
        if right_wrist:
            right_wrist.visualise_3d_data(right_elbow)
            right_wrist.visualise_speed_profile()
            right_wrist.visualise_acceleration_profile()
            right_wrist.visualise_zero_crossings()
            right_wrist.count_number_of_velocity_peaks()
            right_wrist.calculate_ratio_mean_peak_velocity()
            right_wrist.calculate_mean_velocity()
            right_wrist.calculate_peak_velocity()
            right_wrist.calculate_sparc()
            right_wrist.calculate_jerk()
            right_wrist.calculate_movement_time()
            right_wrist.calculate_percentage_time_to_peak_velocity()
        if target:
            target.visualise_3d_data()
        if end_effector:
            end_effector.visualise_3d_data()
            end_effector.calculate_target_error_distance(target)
            end_effector.calculate_hand_path_ratio(target)
            visualise_barchart_per_iteration(
                end_effector.target_error_distance, ts, iteration, "target_error_distance", "", end_effector.object_name
            )
            visualise_barchart_per_iteration(
                end_effector.hand_path_ratio, ts, iteration, "hand_path_ratio", "", end_effector.object_name
            )

        visualise_max_angle(
            right_shoulder,
            left_shoulder,
            right_elbow,
            left_elbow,
            ts,
            iteration,
        )
        visualise_max_speed_angular(
            right_shoulder,
            left_shoulder,
            right_elbow,
            left_elbow,
            ts,
            iteration,
        )
        visualise_max_acceleration(
            right_wrist, left_wrist, right_shoulder, left_shoulder, right_elbow, left_elbow, ts, iteration
        )
        visualise_max_zero_crossings(
            right_wrist, left_wrist, right_shoulder, left_shoulder, right_elbow, left_elbow, ts, iteration
        )
        visualise_speed_profile_iteration(right_wrist, left_wrist, ts, iteration)

        for limb in [left_wrist, right_wrist, left_shoulder, right_shoulder, left_elbow, right_elbow]:
            if limb and list(limb.speed_profile) != [0]:
                count_velocity_peaks_per_iteration(limb.speed_profile, ts, iteration)
                calculate_percentage_time_to_peak_velocity(limb.speed_profile, ts, iteration)
                visualise_barchart_per_iteration(
                    limb.mean_velocity,
                    ts,
                    iteration,
                    "mean_velocity",
                    limb.side,
                    limb.object_name,
                )
                visualise_barchart_per_iteration(
                    limb.peak_velocity,
                    ts,
                    iteration,
                    "peak_velocity",
                    limb.side,
                    limb.object_name,
                )
                visualise_barchart_per_iteration(
                    limb.ratio_mean_peak_velocity,
                    ts,
                    iteration,
                    "ratio_mean_peak_velocity",
                    limb.side,
                    limb.object_name,
                )
                visualise_barchart_per_iteration(
                    limb.number_of_velocity_peaks,
                    ts,
                    iteration,
                    "number_of_velocity_peaks",
                    limb.side,
                    limb.object_name,
                )
                visualise_barchart_per_iteration(
                    limb.zero_crossings, ts, iteration, "zero_crossings_bar", limb.side, limb.object_name
                )
                visualise_barchart_per_iteration(limb.sparc, ts, iteration, "sparc", limb.side, limb.object_name)
                visualise_barchart_per_iteration(limb.jerk, ts, iteration, "jerk", limb.side, limb.object_name)
                visualise_barchart_per_iteration(
                    limb.movement_time, ts, iteration, "movement_time", limb.side, limb.object_name
                )
                visualise_barchart_per_iteration(
                    limb.percentage_time_to_peak_velocity,
                    ts,
                    iteration,
                    "time_to_peak_velocity",
                    limb.side,
                    limb.object_name,
                )

    def default_viewport(self) -> rrb.BlueprintLike:
        view = rrb.Vertical(
            contents=[
                rrb.Horizontal(
                    contents=[
                        rrb.Spatial3DView(
                            name="3D Scene",
                            origin="3d/",
                        ),  # background=[100, 149, 237]
                        rrb.Spatial3DView(name="Trajectory", origin="trajectory"),
                        rrb.TimeSeriesView(name="Range of Motion (degree)", origin="angle"),
                        rrb.TimeSeriesView(name="Speed - Movement Time (s)", origin="movement_time"),
                    ]
                ),
                rrb.Horizontal(
                    contents=[
                        rrb.TimeSeriesView(name="Speed Profile Linear", origin="speed_profile_linear"),
                        rrb.TimeSeriesView(name="Speed Profile Angular", origin="speed_profile_angular"),
                        rrb.TimeSeriesView(name="Speed - Mean Velocity", origin="mean_velocity"),
                        rrb.TimeSeriesView(name="Speed - Peak Velocity", origin="peak_velocity"),
                    ]
                ),
                rrb.Horizontal(
                    contents=[
                        rrb.TimeSeriesView(
                            name="Control Strategy - Time to Peak Velocity (%)", origin="time_to_peak_velocity"
                        ),
                        rrb.TimeSeriesView(
                            name="Smoothness - Ratio Mean and Peak Velocity", origin="ratio_mean_peak_velocity"
                        ),
                        rrb.TimeSeriesView(
                            name="Smoothness - Number of Peak Velocity", origin="number_of_velocity_peaks"
                        ),
                        rrb.TimeSeriesView(name="Smoothness - Zero Crossing Acceleration", origin="zero_crossings_bar"),
                    ]
                ),
                rrb.Horizontal(
                    contents=[
                        rrb.TimeSeriesView(name="Smoothness - Jerk", origin="jerk"),
                        rrb.TimeSeriesView(name="Smoothness - SPARC", origin="sparc"),
                        rrb.TimeSeriesView(name="Accuracy - Target Error Distance", origin="target_error_distance"),
                        rrb.TimeSeriesView(name="Efficiency - Hand Path Ratio", origin="hand_path_ratio"),
                    ]
                ),
            ]
        )

        # hide some panels
        return rrb.Blueprint(
            view,
            rrb.TimePanel(state="collapsed"),
            rrb.SelectionPanel(state="collapsed"),
            rrb.BlueprintPanel(state="collapsed"),
        )


if __name__ == "__main__":
    main()
