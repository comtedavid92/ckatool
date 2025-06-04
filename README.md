# CKATool: A Clinical Kinematic Analysis Toolbox for Upper Limb Rehabilitation 

Tool for analyzing upper limb movement using 3D motion tracking data.

## Preparing environment

[Install uv](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2) with method of your choice.
Verify it is installed properly from terminal:

```sh
uv --version
```

Install required dependencies to run this project:

```sh
cd path/to/project
uv sync
```

## Convert dataset

We support converting dataset from [Arm-CODA](https://doi.org/10.5201/ipol.2024.494), [3D-ARM-Gaze](https://doi.org/10.1038/s41597-024-03765-4), and [IntelliRehabDS](https://doi.org/10.3390/data6050046).
Different dataset should be put in different folders.
For example dataset Arm-CODA in folder `armcoda`, 3D-ARM-Gaze in folder `3darmgaze`, and IntelliRehabDS in folder `intellirehab`.

To convert a dataset:

```sh
uv run python ./src/ckatool/cli/dataset_converter.py --input ./dataset/3darmgaze --source 3darmgaze 
```

The result would be available on `path/to/3darmgaze_output`.

## Running the tool

To run:

```sh
uv run main -i ./dataset/3darmgaze_output/s1_test_RNP_after_targets_pairs.csv
```

Note:
1. To vizualize Arm-CODA dataset, open `src/ckatool/lib/limb.py` and set `radii` accordingly.
2. In this repository, we attach some sample data for your convenience. They still belong to their respective datasets and are used for testing purposes only.
