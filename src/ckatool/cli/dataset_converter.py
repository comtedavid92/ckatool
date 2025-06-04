import argparse
import csv
import json
import math
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from math import inf, isnan
from pathlib import Path

import orjson


class Converter(ABC):
    @abstractmethod
    def run(self, input_dir: str): ...


class ArmCodaConverter(Converter):
    def __init__(self):
        self._important_points = {
            "left_shoulder": [10],
            "right_shoulder": [8],
            "left_elbow": [17],
            "right_elbow": [16],
            "hip": [11],
            "left_wrist": [28, 29],
            "right_wrist": [21, 20],
            "neck": [10, 8],
        }
        self._rows_per_second = 100

    def run(self, input_dir: str):
        inputs = []
        for root, dirs, filenames in os.walk(input_dir):
            for name in filenames:
                if name.endswith("csv"):
                    inputs.append(Path(root) / name)

        for file in inputs:
            json_pair = file.with_suffix(".json")

            output_loc = Path(f"{file.parent}_output") / file.name
            output_loc.parent.mkdir(exist_ok=True, parents=True)

            with (
                open(file, "r", newline="") as data_file,
                open(json_pair, "r") as meta_file,
                open(output_loc, "w", newline="") as out_file,
            ):
                # get iteration indices from JSON metadata file
                meta = json.load(meta_file)

                # write header for result CSV
                fieldnames = ["iteration", "timestamp"]
                for key in self._important_points.keys():
                    for axis in ["x", "y", "z"]:
                        fieldnames.append(f"{key}_{axis}")
                writer = csv.DictWriter(out_file, fieldnames=fieldnames)
                writer.writeheader()

                # starting time for generating timestamps
                starting_second = math.floor(time.time())

                row_count = 0
                reader = csv.DictReader(data_file)
                for i, row in enumerate(reader):
                    ts = starting_second + (i / self._rows_per_second)
                    buffer = {"timestamp": ts}

                    for limb, points in self._important_points.items():
                        for axis in ["x", "y", "z"]:
                            if len(points) == 1:
                                buffer[f"{limb}_{axis}"] = row[f"marker_{points[0]}_{axis}"]
                            else:
                                mid_point = (
                                    float(row[f"marker_{points[0]}_{axis}"]) + float(row[f"marker_{points[1]}_{axis}"])
                                ) / 2
                                buffer[f"{limb}_{axis}"] = mid_point

                    for label, threshold in meta["Movement_label"].items():
                        # label looks like: Iteration_1
                        iteration_num = int(label.split("_")[1])
                        start = threshold[0]
                        end = inf if isnan(threshold[1]) else threshold[1]
                        if start <= row_count < end:
                            buffer["iteration"] = iteration_num

                    writer.writerow(buffer)
                    row_count += 1


class ArmGazeConverter(Converter):
    def __init__(self):
        self._important_points = {
            "left_shoulder": "oppositeShoulderVirtPos",  # the shoulder data is flipped in the 3D Arm Gaze dataset
            "right_shoulder": "shouCustPos",
            "right_elbow": "elbCustPos",
            "hip": "trunkVirtPos",
            "right_wrist": "wriCustPos",
            "neck": "neckVirtPos",
            "target": "tgtPos",
            "end_effector": "endEffCustPos",
        }

    def run(self, input_dir: str):
        inputs = []
        for root, dirs, filenames in os.walk(input_dir):
            for name in filenames:
                if name.endswith("json"):
                    inputs.append(Path(root) / name)

        for file in inputs:
            output_loc = Path(f"{file.parent}_output") / file.with_suffix(".csv").name
            output_loc.parent.mkdir(exist_ok=True, parents=True)

            with (
                open(file, "r", newline="") as source_file,
                open(output_loc, "w", newline="") as out_file,
            ):
                # write header for result CSV
                fieldnames = ["iteration", "timestamp"]
                for key in self._important_points.keys():
                    for axis in ["x", "y", "z"]:
                        fieldnames.append(f"{key}_{axis}")
                writer = csv.DictWriter(out_file, fieldnames=fieldnames)
                writer.writeheader()

                data: dict[str, list[dict]] = orjson.loads(source_file.read())

                prev_neutral_posture = True
                iteration = 0
                for row in data["samples"]:
                    buffer = {"timestamp": row["timestamp"]}
                    for limb, location in self._important_points.items():
                        for axis in ["x", "y", "z"]:
                            buffer[f"{limb}_{axis}"] = row[location][axis]

                    if prev_neutral_posture and not row["returnInitNeutralPosture"]:
                        # change from True to False -> starts new iteration
                        iteration += 1

                    prev_neutral_posture = row["returnInitNeutralPosture"]
                    if not row["returnInitNeutralPosture"]:
                        # inside iteration
                        buffer["iteration"] = iteration

                    writer.writerow(buffer)


class IntelliRehabConverter(Converter):
    def __init__(self):
        self._important_points = {
            "right_shoulder": "ShoulderLeft",  # kinect data is flipped
            "left_shoulder": "ShoulderRight",
            "right_elbow": "ElbowLeft",
            "left_elbow": "ElbowRight",
            "hip": "SpineMid",
            "right_wrist": "WristLeft",
            "left_wrist": "WristRight",
            "neck": "SpineShoulder",
        }
        self._rows_per_second = 30

    def run(self, input_dir: str):
        inputs = []
        for root, dirs, filenames in os.walk(input_dir):
            for name in filenames:
                if name.endswith("txt"):
                    inputs.append(Path(root) / name)

        # run in parallel because there are many files
        with ProcessPoolExecutor() as exe:
            exe.submit(self._convert_each_file, inputs)

    def _convert_each_file(self, inputs: list[Path]):
        for file in inputs:
            output_loc = Path(f"{file.parent}_output") / file.with_suffix(".csv").name
            output_loc.parent.mkdir(exist_ok=True, parents=True)

            with (
                open(file, "r", newline="") as source_file,
                open(output_loc, "w", newline="") as out_file,
            ):
                # write header for result CSV
                fieldnames = ["iteration", "timestamp"]
                for key in self._important_points.keys():
                    for axis in ["x", "y", "z"]:
                        fieldnames.append(f"{key}_{axis}")
                writer = csv.DictWriter(out_file, fieldnames=fieldnames)
                writer.writeheader()

                # starting time for generating timestamps
                starting_second = math.floor(time.time())

                # source file does not have header/fieldnames, list below taken from paper
                limbs = [
                    "SpineBase",
                    "SpineMid",
                    "Neck",
                    "Head",
                    "ShoulderLeft",
                    "ElbowLeft",
                    "WristLeft",
                    "HandLeft",
                    "ShoulderRight",
                    "ElbowRight",
                    "WristRight",
                    "HandRight",
                    "HipLeft",
                    "KneeLeft",
                    "AnkleLeft",
                    "FootLeft",
                    "HipRight",
                    "KneeRight",
                    "AnkleRight",
                    "FootRight",
                    "SpineShoulder",
                    "HandTipLeft",
                    "ThumbLeft",
                    "HandTipRight",
                    "ThumbRight",
                ]
                fieldnames = [f"{limb}_{axis}" for limb in limbs for axis in ["x", "y", "z"]]
                reader = csv.DictReader(source_file, fieldnames)
                for i, row in enumerate(reader):
                    ts = starting_second + (i / self._rows_per_second)
                    buffer = {"timestamp": ts}
                    for limb, location in self._important_points.items():
                        for axis in ["x", "y", "z"]:
                            buffer[f"{limb}_{axis}"] = row[f"{location}_{axis}"]

                    buffer["iteration"] = self._get_iteration_from_filename(file.name)
                    writer.writerow(buffer)

    def _get_iteration_from_filename(self, name: str) -> str:
        # name's format:
        # SubjectID_DateID_GestureLabel_RepetitionNo_CorrectLabel_Position.txt
        return name.split("_")[3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    args = parser.parse_args()

    match args.source:
        case "armcoda":
            converter = ArmCodaConverter()
        case "3darmgaze":
            converter = ArmGazeConverter()
        case "intellirehab":
            converter = IntelliRehabConverter()
        case _:
            raise ValueError(f"unregistered source {args.source}")

    converter.run(args.input_dir)
