from __future__ import annotations

from math import isnan

import numpy as np
import polars as pl
from scipy.signal import argrelextrema, butter, filtfilt

from .smoothness import sampling_frequency_from_timestamp, sparc

try: import rerun as rr
except ImportError: rr = None


def _rr(): 
    if rr is None: raise RuntimeError("Rerun is not installed: visualisation is unavailable.")
    return rr


class Object:
    def __init__(self, ts: pl.Series, x: pl.Series, y: pl.Series, z: pl.Series, iteration: pl.Series):
        self.df = pl.DataFrame({"timestamp": ts, "x": x, "y": y, "z": z, "iteration": iteration})

        self.point_radii = np.array([0.02] * self.df.height)
        self.line_radii = np.array([0.01] * self.df.height)

        # note: for Arm-Coda, use the radii below and comment the above
        # self.point_radii = np.array([9.0] * self.df.height)
        # self.line_radii = np.array([7.0] * self.df.height)

    @property
    def xyz_coordinates(self) -> pl.DataFrame:
        return self.df["x", "y", "z"]

    @property
    def object_name(self) -> str:
        return self.__class__.__name__.lower()


class Neck(Object):
    def visualise_3d_data(self):
        _rr().send_columns(
            f"3d/points/{self.object_name}",
            indexes=[_rr().TimeColumn("record_time", timestamp=self.df["timestamp"])],
            columns=[
                *_rr().archetypes.Points3D.columns(positions=self.xyz_coordinates, radii=self.point_radii),
            ],
        )


class Hip(Object):
    def visualise_3d_data(self):
        _rr().send_columns(
            f"3d/points/{self.object_name}",
            indexes=[_rr().TimeColumn("record_time", timestamp=self.df["timestamp"])],
            columns=[
                *_rr().archetypes.Points3D.columns(positions=self.xyz_coordinates, radii=self.point_radii),
            ],
        )


class Shoulder(Object):
    def __init__(self, ts: pl.Series, x: pl.Series, y: pl.Series, z: pl.Series, iteration: pl.Series, side: str):
        super().__init__(ts, x, y, z, iteration)
        self.side = side
        self.movement_time = {}
        self.mean_velocity = {}
        self.peak_velocity = {}
        self.ratio_mean_peak_velocity = {}
        self.number_of_velocity_peaks = {}
        self.zero_crossings = {}
        self.sparc = {}
        self.jerk = {}
        self.percentage_time_to_peak_velocity = {}

    @property
    def speed_profile(self) -> np.ndarray:
        return self.df["speed_profile"].to_numpy() if "speed_profile" in self.df.columns else [0]

    @property
    def acceleration_profile(self) -> np.ndarray:
        return self.df["acceleration_profile"].to_numpy() if "acceleration_profile" in self.df.columns else [0]

    @property
    def angle(self) -> np.ndarray:
        return self.df["angle"].to_numpy() if "angle" in self.df.columns else [0]

    def visualise_speed_profile(self, elbow: Elbow | None, neck: Neck | None, hip: Hip | None):
        """
        Calculates and visualise speed profile.
        Updates the DataFrame with a new column "speed_profile" containing the speed values.
        """
        if elbow is None or neck is None or hip is None:
            print(f"Cannot visualise speed profile for {self.side} shoulder, missing required limbs.")
            return

        # shoulder angle
        angle = self._compute_angles(
            elbow.xyz_coordinates.to_numpy(),
            self.xyz_coordinates.to_numpy(),
            neck.xyz_coordinates.to_numpy(),
            hip.xyz_coordinates.to_numpy(),
        )
        _rr().send_columns(
            f"angle/{self.side}\ {self.object_name}",
            indexes=[_rr().TimeColumn("record_time", timestamp=self.df["timestamp"])],
            columns=[
                *_rr().archetypes.Scalars.columns(scalars=angle),
            ],
        )

        # speed profile
        speed_profile = calculate_speed_profile(angle.reshape(-1, 1), self.df["timestamp"].to_numpy())
        _rr().send_columns(
            f"speed_profile_angular/{self.side}\ {self.object_name}",
            indexes=[_rr().TimeColumn("record_time", timestamp=self.df["timestamp"])],
            columns=[
                *_rr().archetypes.Scalars.columns(scalars=speed_profile),
            ],
        )

        self.df.insert_column(-1, pl.Series("speed_profile", speed_profile))
        self.df.insert_column(-1, pl.Series("angle", angle))

    def visualise_3d_data(self, neck: Neck | None, hip: Hip | None):
        _rr().send_columns(
            f"3d/points/{self.object_name}",
            indexes=[_rr().TimeColumn("record_time", timestamp=self.df["timestamp"])],
            columns=[*_rr().archetypes.Points3D.columns(positions=self.xyz_coordinates, radii=self.point_radii)],
        )

        # draw lines to neck and hip if they exist
        if neck:
            self._draw_line(f"3d/{self.side}/lines/neck_{self.object_name}", self.xyz_coordinates, neck.xyz_coordinates)
        if hip:
            self._draw_line(f"3d/{self.side}/lines/hip_{self.object_name}", self.xyz_coordinates, hip.xyz_coordinates)

    def _draw_line(self, entity_path: str, shoulder: pl.DataFrame, other_limb: pl.DataFrame):
        color = [0x00000000] * self.df.height
        line = np.hstack((shoulder, other_limb)).reshape((self.df.height, 2, 3))
        _rr().send_columns(
            entity_path,
            indexes=[_rr().TimeColumn("record_time", timestamp=self.df["timestamp"])],
            columns=[*_rr().archetypes.LineStrips3D.columns(strips=line, colors=color, radii=self.line_radii)],
        )

    def _compute_angles(
        self,
        elbow: np.ndarray,
        shoulder: np.ndarray,
        neck: np.ndarray,
        hip: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the shoulder angle for timeseries data using:
        elbow → shoulder → shifted-hip (hip′),
        where hip′ is calculated by shifting the original hip position
        with the same vector as neck → shoulder.

        Arguments:
            elbow: (N, 3) ndarray of elbow positions
            shoulder: (N, 3) ndarray of shoulder positions
            neck: (N, 3) ndarray of neck positions
            hip: (N, 3) ndarray of hip positions

        Returns:
            angles: (N,) ndarray of shoulder angles in degrees
        """
        # Step 1: Compute the translation vector neck → shoulder
        neck_to_shoulder = shoulder - neck

        # Step 2: Shift the hip to create hip′
        shifted_hip = hip + neck_to_shoulder

        # Step 3: Loop through rows and compute angle at shoulder
        num_frames = elbow.shape[0]
        angles = np.zeros(num_frames, dtype=float)

        for i in range(num_frames):
            angles[i] = calculate_angle(elbow[i], shoulder[i], shifted_hip[i])

        return angles  # shape (N,)

    def visualise_acceleration_profile(self):
        _visualise_acceleration_profile(self)

    def visualise_zero_crossings(self):
        _calculate_zero_crossings(self)
        _visualise_zero_crossings_trace(self)

    def count_number_of_velocity_peaks(self):
        _count_velocity_peaks(self)

    def calculate_mean_velocity(self):
        _calculate_mean_velocity(self)

    def calculate_peak_velocity(self):
        _calculate_peak_velocity(self)

    def calculate_ratio_mean_peak_velocity(self):
        _calculate_ratio_mean_peak_velocity(self)

    def calculate_movement_time(self):
        _calculate_movement_time(self)

    def calculate_sparc(self):
        _calculate_sparc(self)

    def calculate_jerk(self):
        _calculate_jerk(self)

    def calculate_percentage_time_to_peak_velocity(self):
        _calculate_percentage_time_to_peak_velocity(self)


class Elbow(Object):
    def __init__(self, ts: pl.Series, x: pl.Series, y: pl.Series, z: pl.Series, iteration: pl.Series, side: str):
        super().__init__(ts, x, y, z, iteration)
        self.side = side
        self.movement_time = {}
        self.mean_velocity = {}
        self.peak_velocity = {}
        self.ratio_mean_peak_velocity = {}
        self.number_of_velocity_peaks = {}
        self.zero_crossings = {}
        self.sparc = {}
        self.jerk = {}
        self.percentage_time_to_peak_velocity = {}

    @property
    def speed_profile(self) -> np.ndarray:
        return self.df["speed_profile"].to_numpy() if "speed_profile" in self.df.columns else [0]

    @property
    def acceleration_profile(self) -> np.ndarray:
        return self.df["acceleration_profile"].to_numpy() if "acceleration_profile" in self.df.columns else [0]

    @property
    def angle(self) -> np.ndarray:
        return self.df["angle"].to_numpy() if "angle" in self.df.columns else [0]

    def visualise_3d_data(self, shoulder: Shoulder | None):
        _rr().send_columns(
            f"3d/points/{self.object_name}",
            indexes=[_rr().TimeColumn("record_time", timestamp=self.df["timestamp"])],
            columns=[*_rr().archetypes.Points3D.columns(positions=self.xyz_coordinates, radii=self.point_radii)],
        )

        # draw lines to shoulder
        if shoulder:
            self._draw_line(
                f"3d/{self.side}/lines/{self.object_name}_shoulder", self.xyz_coordinates, shoulder.xyz_coordinates
            )

    def _draw_line(self, entity_path: str, elbow: pl.DataFrame, shoulder: pl.DataFrame):
        color = [0xAAAAAAAA] * self.df.height
        line = np.hstack((elbow, shoulder)).reshape((self.df.height, 2, 3))
        _rr().send_columns(
            entity_path,
            indexes=[_rr().TimeColumn("record_time", timestamp=self.df["timestamp"])],
            columns=[*_rr().archetypes.LineStrips3D.columns(strips=line, colors=color, radii=self.line_radii)],
        )

    def visualise_speed_profile(self, wrist: Wrist | None, shoulder: Shoulder | None):
        """
        Calculates and visualise speed profile.
        Updates the DataFrame with a new column "speed_profile" containing the speed values.
        """
        if wrist is None or shoulder is None:
            print(f"Cannot visualise speed profile for {self.side} {self.object_name}, missing required limbs.")
            return

        # elbow angle
        angle = self._compute_angles(
            wrist.xyz_coordinates.to_numpy(), self.xyz_coordinates.to_numpy(), shoulder.xyz_coordinates.to_numpy()
        )
        _rr().send_columns(
            f"angle/{self.side}\ {self.object_name}",
            indexes=[_rr().TimeColumn("record_time", timestamp=self.df["timestamp"])],
            columns=[
                *_rr().archetypes.Scalars.columns(scalars=angle),
            ],
        )

        # speed profile
        speed_profile = calculate_speed_profile(angle.reshape(-1, 1), self.df["timestamp"].to_numpy())
        _rr().send_columns(
            f"speed_profile_angular/{self.side}\ {self.object_name}",
            indexes=[_rr().TimeColumn("record_time", timestamp=self.df["timestamp"])],
            columns=[
                *_rr().archetypes.Scalars.columns(scalars=speed_profile),
            ],
        )

        self.df.insert_column(-1, pl.Series("speed_profile", speed_profile))
        self.df.insert_column(-1, pl.Series("angle", angle))

    def _compute_angles(
        self,
        wrist: np.ndarray,
        elbow: np.ndarray,
        shoulder: np.ndarray,
    ) -> np.ndarray:
        # Loop through rows and compute angle at elbow
        num_frames = elbow.shape[0]
        angles = np.zeros(num_frames, dtype=float)

        for i in range(num_frames):
            angles[i] = calculate_angle(wrist[i], elbow[i], shoulder[i])

        return angles  # shape (N,)

    def visualise_acceleration_profile(self):
        _visualise_acceleration_profile(self)

    def visualise_zero_crossings(self):
        _calculate_zero_crossings(self)
        _visualise_zero_crossings_trace(self)

    def count_number_of_velocity_peaks(self):
        _count_velocity_peaks(self)

    def calculate_mean_velocity(self):
        _calculate_mean_velocity(self)

    def calculate_peak_velocity(self):
        _calculate_peak_velocity(self)

    def calculate_ratio_mean_peak_velocity(self):
        _calculate_ratio_mean_peak_velocity(self)

    def calculate_movement_time(self):
        _calculate_movement_time(self)

    def calculate_sparc(self):
        _calculate_sparc(self)

    def calculate_jerk(self):
        _calculate_jerk(self)

    def calculate_percentage_time_to_peak_velocity(self):
        _calculate_percentage_time_to_peak_velocity(self)


class Wrist(Object):
    def __init__(self, ts: pl.Series, x: pl.Series, y: pl.Series, z: pl.Series, iteration: pl.Series, side: str):
        super().__init__(ts, x, y, z, iteration)
        self.side = side
        self.movement_time = {}
        self.mean_velocity = {}
        self.peak_velocity = {}
        self.ratio_mean_peak_velocity = {}
        self.number_of_velocity_peaks = {}
        self.zero_crossings = {}
        self.sparc = {}
        self.jerk = {}
        self.percentage_time_to_peak_velocity = {}

    def visualise_3d_data(self, elbow: Elbow | None):
        # draw 3D points
        _rr().send_columns(
            f"3d/points/{self.object_name}",
            indexes=[_rr().TimeColumn("record_time", timestamp=self.df["timestamp"])],
            columns=[*_rr().archetypes.Points3D.columns(positions=self.xyz_coordinates, radii=self.point_radii)],
        )

        # draw lines to elbow
        if elbow:
            self._draw_line(
                f"3d/{self.side}/lines/{self.object_name}_elbow", self.xyz_coordinates, elbow.xyz_coordinates
            )

        trajectory_color = [0xAAAAAAAA] * self.df.height
        position_color = [0x00000000] * self.df.height
        # create connections between consecutive rows
        wrist_positions = self.xyz_coordinates.to_numpy()
        connections = np.hstack((wrist_positions[:-1], wrist_positions[1:])).reshape(-1, 2, 3)
        _rr().log(
            f"trajectory/{self.side}/{self.object_name}_static",
            _rr().LineStrips3D(strips=connections, colors=trajectory_color, radii=self.line_radii / 2),
            static=True,
        )
        _rr().send_columns(
            f"trajectory/{self.side}/{self.object_name}",
            indexes=[_rr().TimeColumn("record_time", timestamp=self.df["timestamp"])],
            columns=[
                *_rr().archetypes.Points3D.columns(
                    positions=self.xyz_coordinates, radii=self.line_radii * 2, colors=position_color
                )
            ],
        )

    def _draw_line(self, entity_path: str, wrist: pl.DataFrame, elbow: pl.DataFrame):
        color = [0xAAAAAAAA] * self.df.height
        line = np.hstack((wrist, elbow)).reshape((self.df.height, 2, 3))
        _rr().send_columns(
            entity_path,
            indexes=[_rr().TimeColumn("record_time", timestamp=self.df["timestamp"])],
            columns=[*_rr().archetypes.LineStrips3D.columns(strips=line, colors=color, radii=self.line_radii)],
        )

    def visualise_speed_profile(self):
        """
        Calculates and visualise speed profile (linear)
        Updates the DataFrame with a new column "speed_profile" containing the speed values.
        """
        speed_profile = calculate_speed_profile(self.xyz_coordinates.to_numpy(), self.df["timestamp"].to_numpy())
        self.df.insert_column(-1, pl.Series("speed_profile", speed_profile))
        _rr().send_columns(
            f"speed_profile_linear/{self.side}\ {self.object_name}",
            indexes=[_rr().TimeColumn("record_time", timestamp=self.df["timestamp"])],
            columns=[
                *_rr().archetypes.Scalars.columns(scalars=self.df["speed_profile"]),
            ],
        )

    def visualise_acceleration_profile(self):
        _visualise_acceleration_profile(self)

    def visualise_zero_crossings(self):
        _calculate_zero_crossings(self)
        _visualise_zero_crossings_trace(self)

    def count_number_of_velocity_peaks(self):
        _count_velocity_peaks(self)

    def calculate_mean_velocity(self):
        _calculate_mean_velocity(self)

    def calculate_peak_velocity(self):
        _calculate_peak_velocity(self)

    def calculate_ratio_mean_peak_velocity(self):
        _calculate_ratio_mean_peak_velocity(self)

    def calculate_movement_time(self):
        _calculate_movement_time(self)

    def calculate_sparc(self):
        _calculate_sparc(self)

    def calculate_jerk(self):
        _calculate_jerk(self)

    def calculate_percentage_time_to_peak_velocity(self):
        _calculate_percentage_time_to_peak_velocity(self)

    @property
    def speed_profile(self) -> np.ndarray:
        return self.df["speed_profile"].to_numpy() if "speed_profile" in self.df.columns else [0]

    @property
    def acceleration_profile(self) -> np.ndarray:
        return self.df["acceleration_profile"].to_numpy() if "acceleration_profile" in self.df.columns else [0]


def _visualise_acceleration_profile(cls):
    """
    Calculates and visualise acceleration profile.
    Updates the DataFrame with a new column "acceleration_profile" containing the acceleration values.
    """
    if "speed_profile" not in cls.df.columns:
        print(f"Cannot visualise acceleration profile for {cls.side} {cls.object_name}, speed profile is missing.")
        return

    acceleration = calculate_acceleration_profile(cls.df["speed_profile"].to_numpy(), cls.df["timestamp"].to_numpy())
    cls.df.insert_column(-1, pl.Series("acceleration_profile", acceleration))
    _rr().send_columns(
        f"acceleration_profile/{cls.side}\ {cls.object_name}\ (linear)",
        indexes=[_rr().TimeColumn("record_time", timestamp=cls.df["timestamp"])],
        columns=[
            *_rr().archetypes.Scalars.columns(scalars=acceleration),
        ],
    )


def _calculate_zero_crossings(cls):
    if "acceleration_profile" not in cls.df.columns:
        print(f"Cannot visualise zero crossings for {cls.side} {cls.object_name}, acceleration profile is missing.")
        return

    crossings = calculate_zero_crossings_per_iteration(
        cls.df["acceleration_profile"].to_numpy(), cls.df["iteration"].to_numpy()
    )
    cls.zero_crossings = crossings


def _visualise_zero_crossings_trace(cls):
    if "acceleration_profile" not in cls.df.columns:
        print(
            f"Cannot visualise zero crossings trace for {cls.side} {cls.object_name}, acceleration profile is missing."
        )
        return

    counter_trace = zero_crossing_counter_trace(
        cls.df["acceleration_profile"].to_numpy(), cls.df["iteration"].to_numpy()
    )
    cls.df.insert_column(-1, pl.Series("zero_crossing_counter_trace", counter_trace))
    _rr().send_columns(
        f"zero_crossings_trace/{cls.side}\ {cls.object_name}\ (linear)",
        indexes=[_rr().TimeColumn("record_time", timestamp=cls.df["timestamp"])],
        columns=[
            *_rr().archetypes.Scalars.columns(scalars=counter_trace),
        ],
    )


def _count_velocity_peaks(cls):
    if "speed_profile" not in cls.df.columns:
        print(f"Cannot count velocity peak for {cls.side} {cls.object_name}, speed profile is missing.")
        return

    number_of_peak = count_velocity_peaks_per_iteration(
        cls.df["speed_profile"].to_numpy(), cls.df["timestamp"].to_numpy(), cls.df["iteration"].to_numpy()
    )
    cls.number_of_velocity_peaks = number_of_peak


def _calculate_mean_velocity(cls):
    if "speed_profile" not in cls.df.columns:
        print(f"Cannot calculate mean velocity for {cls.side} {cls.object_name}, speed profile is missing.")
        return

    mean_velocities = calculate_mean_velocity_per_iteration(
        cls.df["speed_profile"].to_numpy(), cls.df["iteration"].to_numpy()
    )

    cls.mean_velocity = mean_velocities


def _calculate_peak_velocity(cls):
    if "speed_profile" not in cls.df.columns:
        print(f"Cannot calculate peak velocity for {cls.side} {cls.object_name}, speed profile is missing.")
        return

    peak_velocities = get_peak_velocity(cls.df["speed_profile"].to_numpy(), cls.df["iteration"].to_numpy())
    cls.peak_velocity = peak_velocities


def _calculate_ratio_mean_peak_velocity(cls):
    if "speed_profile" not in cls.df.columns:
        print(
            f"Cannot calculate ratio mean and peak velocity for {cls.side} {cls.object_name}, speed profile is missing."
        )
        return

    mean_velocities = calculate_mean_velocity_per_iteration(
        cls.df["speed_profile"].to_numpy(), cls.df["iteration"].to_numpy()
    )
    peak_velocities = get_peak_velocity(cls.df["speed_profile"].to_numpy(), cls.df["iteration"].to_numpy())
    ratio_mean_peak = calculate_ratio_mean_peak_velocity(mean_velocities, peak_velocities)

    cls.ratio_mean_peak_velocity = ratio_mean_peak


def _calculate_movement_time(cls):
    if "speed_profile" not in cls.df.columns:
        print(f"Cannot calculate movement time for {cls.side} {cls.object_name}, speed profile is missing.")
        return

    movement_time = calculate_movement_time_per_iteration(
        cls.df["timestamp"].to_numpy(), cls.df["iteration"].to_numpy()
    )

    cls.movement_time = movement_time


def _calculate_sparc(cls):
    if "speed_profile" not in cls.df.columns:
        print(f"Cannot calculate sparc for {cls.side} {cls.object_name}, speed profile is missing.")
        return

    sparc_values = calculate_sparc_per_iteration(
        cls.df["timestamp"], cls.df["speed_profile"].to_numpy(), cls.df["iteration"].to_numpy()
    )
    cls.sparc = sparc_values


def _calculate_jerk(cls):
    if "speed_profile" not in cls.df.columns:
        print(f"Cannot calculate log dimensionless jerk for {cls.side} {cls.object_name}, speed profile is missing.")
        return

    x = cls.df["x"].to_numpy()
    y = cls.df["y"].to_numpy()
    z = cls.df["z"].to_numpy()
    jerk_value = calculate_jerk_per_iteration(cls.df["timestamp"].to_numpy(), x, y, z, cls.df["iteration"].to_numpy())

    cls.jerk = jerk_value


def _calculate_percentage_time_to_peak_velocity(cls):
    if "speed_profile" not in cls.df.columns:
        print(f"Cannot calculate time to peak velocity for {cls.side} {cls.object_name}, speed profile is missing.")
        return

    percentage_time = calculate_percentage_time_to_peak_velocity(
        cls.df["speed_profile"].to_numpy(), cls.df["timestamp"].to_numpy(), cls.df["iteration"].to_numpy()
    )

    cls.percentage_time_to_peak_velocity = percentage_time


class Target(Object):
    def visualise_3d_data(self):
        _rr().send_columns(
            f"3d/points/target",
            indexes=[_rr().TimeColumn("record_time", timestamp=self.df["timestamp"])],
            columns=[
                *_rr().archetypes.Points3D.columns(positions=self.xyz_coordinates, radii=self.point_radii),
            ],
        )


class EndEffector(Object):
    def __init__(self, ts: pl.Series, x: pl.Series, y: pl.Series, z: pl.Series, iteration: pl.Series):
        super().__init__(ts, x, y, z, iteration)
        self.target_error_distance = {}
        self.hand_path_ratio = {}

    def visualise_3d_data(self):
        _rr().send_columns(
            f"3d/points/end_effector",
            indexes=[_rr().TimeColumn("record_time", timestamp=self.df["timestamp"])],
            columns=[
                *_rr().archetypes.Points3D.columns(positions=self.xyz_coordinates, radii=self.point_radii),
            ],
        )

    def calculate_target_error_distance(self, target: Target | None):
        if target is None:
            print("Target is None, cannot calculate target error distance.")
            return {}

        errors = {}
        # Filter out rows with missing iteration values
        data = self.df.filter(pl.col("iteration").is_not_nan())

        # Group by iteration
        for iter_label in data["iteration"].unique():
            # Filter for this iteration
            ee_df = self.df.filter(pl.col("iteration") == iter_label)
            target_df = target.df.filter(pl.col("iteration") == iter_label)
            if ee_df.height == 0 or target_df.height == 0:
                continue

            # Calculate the Euclidean distance between end effector and target positions
            distances = np.linalg.norm(
                ee_df.select(["x", "y", "z"]).to_numpy() - target_df.select(["x", "y", "z"]).to_numpy(), axis=1
            )
            mean_distance = np.nanmean(distances)
            errors[iter_label] = mean_distance

        self.target_error_distance = errors

    def calculate_hand_path_ratio(self, target: Target | None):
        if target is None:
            print("Target is None, cannot calculate hand path ratio.")
            return {}

        ratios = {}
        # Filter out rows with missing iteration values
        data = self.df.filter(pl.col("iteration").is_not_nan())

        # Group by iteration
        for iter_label in data["iteration"].unique():
            ee_df = self.df.filter(pl.col("iteration") == iter_label)
            target_df = target.df.filter(pl.col("iteration") == iter_label)
            if ee_df.height == 0 or target_df.height == 0:
                continue

            # Find target movement segments (where target position changes)
            target_positions = target_df.select(["x", "y", "z"]).to_numpy()
            target_timestamps = target_df["timestamp"].to_numpy()
            # Find indices where target moves
            move_indices = np.where(np.any(np.diff(target_positions, axis=0) != 0, axis=1))[0]
            # Always include the first index as a start
            segment_starts = np.insert(move_indices + 1, 0, 0)
            # End at the next move or the end
            segment_ends = np.append(move_indices, len(target_positions) - 1)

            total_true_path = 0.0
            total_euclidean = 0.0

            for start, end in zip(segment_starts, segment_ends):
                t_start = target_timestamps[start]
                t_end = target_timestamps[end]
                # Get end effector positions in this window
                ee_window = (
                    ee_df.filter((pl.col("timestamp") >= t_start) & (pl.col("timestamp") <= t_end))
                    .select(["x", "y", "z"])
                    .to_numpy()
                )
                if ee_window.shape[0] < 2:
                    continue
                # True path
                true_path = np.sum(np.linalg.norm(np.diff(ee_window, axis=0), axis=1))
                # Euclidean distance
                euclidean = np.linalg.norm(ee_window[-1] - ee_window[0])
                total_true_path += true_path
                total_euclidean += euclidean

            ratio = total_true_path / total_euclidean if total_euclidean != 0 else np.nan
            ratios[iter_label] = ratio

        self.hand_path_ratio = ratios


def calculate_speed_profile(data: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """
    Calculate the speed (linear or angular) profile from time-series data.

    Parameters:
    - data: np.ndarray of shape (n, d), where d = 3 for 3D positions or d = 1 for joint angles.
    - timestamps: np.ndarray of shape (n,), containing the time for each sample.

    Returns:
    - speeds: np.ndarray of shape (n,), containing the smoothed speed between samples.
    """

    # Compute differences
    data_diff = np.diff(data, axis=0)
    time_diff = np.diff(timestamps)

    # Avoid division by zero
    time_diff[time_diff == 0] = np.nan

    # Compute speed (magnitude of change per unit time)
    magnitudes = np.linalg.norm(data_diff, axis=1)
    speeds = magnitudes / time_diff

    # Insert 0 to match original length
    speeds = np.insert(speeds, 0, 0)

    # Apply low-pass Butterworth filter
    mean_dt = np.nanmean(time_diff)
    if np.isnan(mean_dt) or mean_dt == 0:
        return speeds  # skip filtering if sampling rate is invalid

    sampling_rate = 1.0 / mean_dt
    cutoff_frequency = 5  # Hz
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_frequency / nyquist

    if 0 < normalized_cutoff < 1:
        b, a = butter(N=4, Wn=normalized_cutoff, btype="low", analog=False)
        speeds = filtfilt(b, a, speeds)

    return speeds


def calculate_angle(limb1: np.ndarray, limb2: np.ndarray, limb3: np.ndarray) -> float:
    # Ensure all inputs are 1D vectors of shape (3,)
    a = np.squeeze(np.asarray(limb1))
    b = np.squeeze(np.asarray(limb2))
    c = np.squeeze(np.asarray(limb3))

    # Compute vectors
    ba = a - b
    bc = c - b

    # Compute cosine of angle
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Avoid division by zero
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle_rad = np.arccos(cosine_angle)
    return float(np.degrees(angle_rad))


def calculate_acceleration_profile(speeds: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """
    Calculate the acceleration profile from a time series of speeds.

    Parameters:
    - speeds: np.ndarray of shape (n,), containing the speed at each timestamp.
    - timestamps: np.ndarray of shape (n,), containing the time for each speed.

    Returns:
    - accelerations: np.ndarray of shape (n,), containing the acceleration between consecutive speeds.
    """
    # Calculate differences in speeds
    speed_differences = np.diff(speeds)

    # Calculate time differences
    time_differences = np.diff(timestamps)

    # Avoid division by zero
    time_differences[time_differences == 0] = np.nan

    # Calculate accelerations
    accelerations = speed_differences / time_differences

    # Pad the accelerations array to match the shape of timestamps
    accelerations = np.insert(accelerations, 0, 0)

    return accelerations


def visualise_max_speed_angular(
    right_shoulder: Shoulder | None,
    left_shoulder: Shoulder | None,
    right_elbow: Elbow | None,
    left_elbow: Elbow | None,
    timestamp: pl.Series,
    iterations: pl.Series,
):
    """
    Visualise the maximum speed of the angular joints.
    """
    right_shoulder_speeds = right_shoulder.speed_profile if right_shoulder else [0]
    left_shoulder_speeds = left_shoulder.speed_profile if left_shoulder else [0]
    right_elbow_speeds = right_elbow.speed_profile if right_elbow else [0]
    left_elbow_speeds = left_elbow.speed_profile if left_elbow else [0]
    max_speed_angular = max(
        np.nanmax(right_shoulder_speeds),
        np.nanmax(left_shoulder_speeds),
        np.nanmax(right_elbow_speeds),
        np.nanmax(left_elbow_speeds),
    )
    # max_speed = np.nanmax(right_wrist)
    iteration_scalars_speeds_angular = [0 if rep is None else max_speed_angular for rep in iterations]
    _rr().send_columns(
        f"speed_profile_angular/iteration",
        indexes=[_rr().TimeColumn("record_time", timestamp=timestamp)],
        columns=[
            *_rr().archetypes.Scalars.columns(scalars=iteration_scalars_speeds_angular),
            *_rr().archetypes.SeriesLines.columns(colors=[0x00FFFFFF] * len(timestamp), widths=[2] * len(timestamp)),
        ],
    )


def calculate_movement_time_per_iteration(timestamps: np.ndarray, iterations: np.ndarray) -> dict:
    """
    Calculate the movement time for each submovement (iteration).

    Parameters:
    - timestamps: pl.Series of timestamps (in seconds or milliseconds).
    - iterations: pl.Series of iteration labels (can contain None).

    Returns:
    - dict: Mapping from iteration label to movement time.
    """
    data = pl.DataFrame({"timestamp": timestamps, "iteration": iterations}).filter(pl.col("iteration").is_not_nan())

    # Group by iteration and calculate movement time
    movement_times = data.group_by("iteration").agg(
        [(pl.col("timestamp").max() - pl.col("timestamp").min()).alias("movement_time")]
    )

    # Convert to dictionary
    return dict(zip(movement_times["iteration"], movement_times["movement_time"]))


def calculate_sparc_per_iteration(ts: np.ndarray, speeds: np.ndarray, iterations: np.ndarray) -> dict:
    data = pl.DataFrame({"timestamp": ts, "speed": speeds, "iteration": iterations})

    data = data.filter(pl.col("iteration").is_not_nan())

    result = {}
    for group in data.group_by("iteration"):
        df = group[1]
        fs = sampling_frequency_from_timestamp(df["timestamp"].to_numpy())
        sparc_value, _, _ = sparc(df["speed"].to_numpy(), fs)
        result[df["iteration"][0]] = sparc_value

    return result


def calculate_jerk_per_iteration(
    ts: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, iterations: np.ndarray
) -> dict:
    data = pl.DataFrame({"timestamp": ts, "x": x, "y": y, "z": z, "iteration": iterations})

    data = data.filter(pl.col("iteration").is_not_nan())

    result = {}
    for group in data.group_by("iteration"):
        df = group[1]
        dt = np.mean(np.diff(df["timestamp"]))

        # Compute third derivatives (jerk) for each axis
        jerk_x = np.gradient(np.gradient(np.gradient(df["x"], dt), dt), dt)
        jerk_y = np.gradient(np.gradient(np.gradient(df["y"], dt), dt), dt)
        jerk_z = np.gradient(np.gradient(np.gradient(df["z"], dt), dt), dt)

        # Magnitude of jerk vector
        jerk_mag_squared = jerk_x**2 + jerk_y**2 + jerk_z**2
        jerk_integral = np.sum(jerk_mag_squared) * dt

        # Duration and path length
        movement_time = df["timestamp"][-1] - df["timestamp"][0]
        diffs = np.diff(np.vstack((df["x"], df["y"], df["z"])), axis=1)
        segment_lengths = np.linalg.norm(diffs, axis=0)
        path_length = np.sum(segment_lengths)

        # Handle zero division cases
        if movement_time == 0 or path_length == 0:
            result[df["iteration"][0]] = 0
            continue

        dimensionless_jerk_value = jerk_integral * movement_time**5 / (path_length**2)
        log_dimensionless_jerk_value = -np.log(dimensionless_jerk_value)
        result[df["iteration"][0]] = log_dimensionless_jerk_value

    return result


def calculate_mean_velocity_per_iteration(speeds: np.ndarray, iterations: pl.Series) -> dict:
    data = pl.DataFrame({"speed": speeds, "iteration": iterations}).filter(pl.col("iteration").is_not_nan())

    # Group by iteration and calculate mean velocity
    mean_velocities = data.group_by("iteration").agg([pl.col("speed").mean().alias("mean_velocity")])

    # Convert to dictionary
    return dict(zip(mean_velocities["iteration"], mean_velocities["mean_velocity"]))


def calculate_ratio_mean_peak_velocity(mean_velocities: dict, peak_velocities: dict) -> dict:
    """
    Calculate the ratio of mean velocity to peak velocity for each iteration.

    Parameters:
    - mean_velocities: dict of mean velocities per iteration.
    - peak_velocities: dict of peak velocities per iteration.

    Returns:
    - dict: Mapping from iteration label to the ratio of mean to peak velocity.
    """
    ratio = {}
    for iteration in mean_velocities:
        if iteration in peak_velocities and peak_velocities[iteration] != 0:
            ratio[iteration] = mean_velocities[iteration] / peak_velocities[iteration]
        else:
            ratio[iteration] = None  # Handle cases where peak velocity is zero or missing
    return ratio


def get_peak_velocity(speeds: np.ndarray, iterations: np.ndarray) -> dict:
    """
    Calculate the peak velocity for each iteration.
    This function computes the maximum speed for each iteration group and returns it as a dictionary.
    """
    data = pl.DataFrame({"speed": speeds, "iteration": iterations}).filter(pl.col("iteration").is_not_nan())

    # Group by iteration and calculate peak velocity
    peak_velocities = data.group_by("iteration").agg([pl.col("speed").max().alias("peak_velocity")])

    # Convert to dictionary
    return dict(zip(peak_velocities["iteration"], peak_velocities["peak_velocity"]))


def calculate_percentage_time_to_peak_velocity(speeds: np.ndarray, ts: np.ndarray, iterations: np.ndarray) -> dict:
    """
    Calculate the percentage time to peak velocity for each iteration.
    """
    data = pl.DataFrame({"speed": speeds, "timestamp": ts, "iteration": iterations}).filter(
        pl.col("iteration").is_not_nan()
    )

    # Sort by iteration and timestamp to ensure proper order
    data = data.sort(["iteration", "timestamp"])

    # Group by iteration and compute time to peak velocity
    result = {}

    for group in data.group_by("iteration"):
        df = group[1]
        peak_idx = df["speed"].arg_max()
        peak_time = df["timestamp"][peak_idx]
        start_time = df["timestamp"][0]
        end_time = df["timestamp"][-1]

        if end_time > start_time:
            tpv_percent = ((peak_time - start_time) / (end_time - start_time)) * 100
            result[df["iteration"][0]] = tpv_percent
        else:
            result[df["iteration"][0]] = None

    return result


def detect_peaks_in_speed_profile(speeds: np.ndarray, ts: np.ndarray, threshold=0, min_interval_ms=150):
    """
    Detect peaks in a scalar speed profile based on threshold and minimum time interval.

    Parameters:
    - speeds: speed_profile
    - time: 1D array of timestamps
    - threshold: minimum amplitude difference from previous minimum (mm/s). Recommended to be 20 based on the reference.
    - min_interval_ms: minimum time between peaks (milliseconds).

    Returns:
    - int: Number of valid velocity peaks.
    """
    v = np.asarray(speeds)
    t = np.asarray(ts)
    peak_indices = argrelextrema(v, np.greater)[0]

    valid_peaks = []
    last_peak_time = -np.inf

    for idx in peak_indices:
        if idx <= 0:
            continue

        # Find previous local minimum before the current peak
        local_min = np.min(v[:idx])
        amp_diff = v[idx] - local_min
        time_diff = (t[idx] - last_peak_time) * 1000  # convert to ms

        if amp_diff > threshold and time_diff >= min_interval_ms:
            valid_peaks.append(idx)
            last_peak_time = t[idx]

    return valid_peaks


def count_velocity_peaks_per_iteration(speeds: np.ndarray, ts: np.ndarray, iterations: np.ndarray) -> dict:
    """
    Count number of valid velocity peaks per iteration group.

    Parameters:
    - speed_profile: List or array of speed values.
    - timestamp: List or array of timestamps (in seconds).
    - iteration: List or array of iteration labels.

    Returns:
    - dict: {iteration_value: number_of_peaks}
    """
    data = pl.DataFrame({"speed": speeds, "timestamp": ts, "iteration": iterations}).filter(
        pl.col("iteration").is_not_nan()
    )
    data = data.sort(["iteration", "timestamp"])

    result = {}

    for group in data.group_by("iteration"):
        df = group[1]
        speed = df["speed"].to_numpy()
        time = df["timestamp"].to_numpy()
        num_peaks = len(detect_peaks_in_speed_profile(speed, time))
        result[df["iteration"][0]] = num_peaks

    return result


def calculate_zero_crossings_per_iteration(accelerations: np.ndarray, iterations: np.ndarray) -> dict:
    """
    Count zero crossings in the acceleration profile per valid iteration.

    Parameters:
    - accelerations: np.ndarray of acceleration values.
    - iterations: np.ndarray of iteration labels (can contain None).

    Returns:
    - dict: Mapping from iteration label to zero-crossing count.
    """
    data = pl.DataFrame({"acceleration": accelerations, "iteration": iterations}).filter(
        pl.col("iteration").is_not_nan()
    )

    # Group by iteration and collect acceleration values
    grouped = data.group_by("iteration").agg([pl.col("acceleration").alias("acc_list")])

    # Compute zero crossings
    zero_crossings = {}
    for iteration, acc_list in grouped.iter_rows():
        acc_array = np.array(acc_list)
        crossings = np.where(np.diff(np.sign(acc_array)) != 0)[0]
        zero_crossings[iteration] = len(crossings)

    return zero_crossings


def zero_crossing_counter_trace(accelerations: np.ndarray, iterations: np.ndarray) -> np.ndarray:
    """
    Generate an array with a running count of zero crossings per valid iteration.

    Parameters:
    - accelerations: np.ndarray of acceleration values.
    - iterations: np.ndarray of iteration labels (can contain np.nan).

    Returns:
    - np.ndarray of same length with zero crossing counter.
    """
    zero_crossing_trace = np.zeros_like(accelerations, dtype=int)
    current_count = 0
    last_sign = None
    last_iteration = None

    for i, (acc, iter_label) in enumerate(zip(accelerations, iterations)):
        # Reset if iteration is NaN
        if iter_label is None or isnan(iter_label):  # Check for NaN
            current_count = 0
            last_sign = None
            last_iteration = None
            zero_crossing_trace[i] = 0
            continue

        # If new iteration starts, reset tracking
        if last_iteration != iter_label:
            current_count = 0
            last_sign = np.sign(acc)
            last_iteration = iter_label
            zero_crossing_trace[i] = current_count
            continue

        current_sign = np.sign(acc)
        if last_sign is not None and current_sign != 0 and current_sign != last_sign:
            current_count += 1

        zero_crossing_trace[i] = current_count
        last_sign = current_sign

    return zero_crossing_trace


def visualise_speed_profile_iteration(
    wrist_a: Wrist | None, wrist_b: Wrist | None, timestamp: pl.Series, iterations: pl.Series
):
    """
    Draws the speed profile iteration for the wrist. Requires another wrist instance to compare against.
    """
    if not wrist_a and not wrist_b:
        print("Both wrist instances are None, cannot visualise speed profile iteration.")
        return

    speed_profile_a = wrist_a.speed_profile if wrist_a else [0]
    speed_profile_b = wrist_b.speed_profile if wrist_b else [0]

    max_speed_linear = max(np.nanmax(speed_profile_a), np.nanmax(speed_profile_b))
    iteration_scalars_speeds_linear = [0 if rep is None else max_speed_linear for rep in iterations]
    _rr().send_columns(
        f"speed_profile_linear/iteration",
        indexes=[_rr().TimeColumn("record_time", timestamp=timestamp)],
        columns=[
            *_rr().archetypes.Scalars.columns(scalars=iteration_scalars_speeds_linear),
            *_rr().archetypes.SeriesLines.columns(colors=[0x00FFFFFF] * len(timestamp), widths=[2] * len(timestamp)),
        ],
    )


def visualise_max_angle(
    right_shoulder: Shoulder | None,
    left_shoulder: Shoulder | None,
    right_elbow: Elbow | None,
    left_elbow: Elbow | None,
    timestamp: pl.Series,
    iterations: pl.Series,
):
    """
    Visualise the maximum angle of the angular joints.
    """
    right_shoulder_angle = right_shoulder.angle if right_shoulder else [0]
    left_shoulder_angle = left_shoulder.angle if left_shoulder else [0]
    right_elbow_angle = right_elbow.angle if right_elbow else [0]
    left_elbow_angle = left_elbow.angle if left_elbow else [0]
    max_angle = max(
        np.nanmax(right_shoulder_angle),
        np.nanmax(left_shoulder_angle),
        np.nanmax(right_elbow_angle),
        np.nanmax(left_elbow_angle),
    )
    iteration_scalars_angle = [0 if rep is None else max_angle for rep in iterations]
    _rr().send_columns(
        f"angle/iteration",
        indexes=[_rr().TimeColumn("record_time", timestamp=timestamp)],
        columns=[
            *_rr().archetypes.Scalars.columns(scalars=iteration_scalars_angle),
            *_rr().archetypes.SeriesLines.columns(colors=[0x00FFFFFF] * len(timestamp), widths=[2] * len(timestamp)),
        ],
    )


def visualise_max_acceleration(
    right_wrist: Wrist | None,
    left_wrist: Wrist | None,
    right_shoulder: Shoulder | None,
    left_shoulder: Shoulder | None,
    right_elbow: Elbow | None,
    left_elbow: Elbow | None,
    timestamp: pl.Series,
    iterations: pl.Series,
):
    right_wrist_acc = right_wrist.acceleration_profile if right_wrist else [0]
    left_wrist_acc = left_wrist.acceleration_profile if left_wrist else [0]
    right_shoulder_acc = right_shoulder.acceleration_profile if right_shoulder else [0]
    left_shoulder_acc = left_shoulder.acceleration_profile if left_shoulder else [0]
    right_elbow_acc = right_elbow.acceleration_profile if right_elbow else [0]
    left_elbow_acc = left_elbow.acceleration_profile if left_elbow else [0]
    max_acceleration = max(
        np.nanmax(right_wrist_acc),
        np.nanmax(left_wrist_acc),
        np.nanmax(right_shoulder_acc),
        np.nanmax(left_shoulder_acc),
        np.nanmax(right_elbow_acc),
        np.nanmax(left_elbow_acc),
    )
    iteration_scalars_angle = [0 if rep is None else max_acceleration for rep in iterations]
    _rr().send_columns(
        f"acceleration_profile/iteration",
        indexes=[_rr().TimeColumn("record_time", timestamp=timestamp)],
        columns=[
            *_rr().archetypes.Scalars.columns(scalars=iteration_scalars_angle),
            *_rr().archetypes.SeriesLines.columns(colors=[0x00FFFFFF] * len(timestamp), widths=[2] * len(timestamp)),
        ],
    )


def visualise_max_zero_crossings(
    right_wrist: Wrist | None,
    left_wrist: Wrist | None,
    right_shoulder: Shoulder | None,
    left_shoulder: Shoulder | None,
    right_elbow: Elbow | None,
    left_elbow: Elbow | None,
    timestamp: pl.Series,
    iterations: pl.Series,
):
    zero_cross_right_wrist = (
        [0] if not right_wrist or not right_wrist.zero_crossings else right_wrist.zero_crossings.values()
    )
    zero_cross_left_wrist = (
        [0] if not left_wrist or not left_wrist.zero_crossings else left_wrist.zero_crossings.values()
    )
    zero_cross_right_shoulder = (
        [0] if not right_shoulder or not right_shoulder.zero_crossings else right_shoulder.zero_crossings.values()
    )
    zero_cross_left_shoulder = (
        [0] if not left_shoulder or not left_shoulder.zero_crossings else left_shoulder.zero_crossings.values()
    )
    zero_cross_right_elbow = (
        [0] if not right_elbow or not right_elbow.zero_crossings else right_elbow.zero_crossings.values()
    )
    zero_cross_left_elbow = (
        [0] if not left_elbow or not left_elbow.zero_crossings else left_elbow.zero_crossings.values()
    )
    max_crossing = max(
        np.nanmax(np.array(list(zero_cross_right_wrist))),
        np.nanmax(np.array(list(zero_cross_left_wrist))),
        np.nanmax(np.array(list(zero_cross_right_shoulder))),
        np.nanmax(np.array(list(zero_cross_left_shoulder))),
        np.nanmax(np.array(list(zero_cross_right_elbow))),
        np.nanmax(np.array(list(zero_cross_left_elbow))),
    )
    iteration_scalars_crossing = [0 if rep is None else max_crossing for rep in iterations]
    _rr().send_columns(
        f"zero_crossings_trace/iteration",
        indexes=[_rr().TimeColumn("record_time", timestamp=timestamp)],
        columns=[
            *_rr().archetypes.Scalars.columns(scalars=iteration_scalars_crossing),
            *_rr().archetypes.SeriesLines.columns(colors=[0x00FFFFFF] * len(timestamp), widths=[2] * len(timestamp)),
        ],
    )


def visualise_barchart_per_iteration(
    data: dict,
    timestamp: pl.Series,
    iterations: pl.Series,
    origin_prefix: str,
    side: str,
    name: str,
):
    iteration_scalars_crossing = [data.get(rep, 0) if rep else 0 for rep in iterations]
    _rr().send_columns(
        f"{origin_prefix}/{side}\ {name}",
        indexes=[_rr().TimeColumn("record_time", timestamp=timestamp)],
        columns=[
            *_rr().archetypes.Scalars.columns(scalars=iteration_scalars_crossing),
            *_rr().archetypes.SeriesLines.columns(widths=[2] * len(timestamp)),
        ],
    )
