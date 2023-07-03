import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import pyvista as pv


def rpy_to_vector(rpy_array: np.ndarray):
    rolls = rpy_array[:, 0]
    pitches = rpy_array[:, 1]
    yaws = rpy_array[:, 2]
    x = np.cos(yaws)
    y = np.sin(yaws)
    z = np.tan(-pitches) * np.sqrt(x**2 + y**2)
    return np.vstack([x, y, z]).T


def vector_to_rpy(vector_array: np.ndarray):
    x = vector_array[:, 0]
    y = vector_array[:, 1]
    z = vector_array[:, 2]
    rolls = np.zeros(len(x))
    pitches = np.arctan(z, np.sqrt(x**2 + y**2))
    yaws = np.arctan2(y, x)
    return np.vstack([rolls, pitches, yaws]).T


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_folder", required=True)
    parser.add_argument("--test_number", default=None, type=int)
    parser.add_argument("--n_measurements", default=None, type=int)
    parser.add_argument("--rel_yaw_period", default=None, type=float)
    parser.add_argument("--rel_yaw_amplitude", default=None, type=float)
    parser.add_argument("--h_disp_period", default=None, type=float)
    parser.add_argument("--h_disp_amplitude", default=None, type=float)
    parser.add_argument("--plot_axis_and_labels", default=0, type=int)
    args, _ = parser.parse_known_args()
    return args


def main():
    # Setup args
    args = get_args()
    env_folder = args.env_folder
    test_number = args.test_number
    desired_n_measurements = args.n_measurements
    desired_rel_yaw_period = args.rel_yaw_period
    desired_rel_yaw_amplitude = args.rel_yaw_amplitude
    desired_h_disp_period = args.h_disp_period
    desired_h_disp_amplitude = args.h_disp_amplitude
    plot_axis_and_path = bool(args.plot_axis_and_labels)
    if test_number is None:
        tests = os.listdir(os.path.join(env_folder, "network_tests"))
        tests = set([name.split(".")[0] for name in tests])
        tests = list(tests)
        tests.sort()
    else:
        tests = [f"test_{test_number:04d}"]
    # Start plotting
    axis_data = np.loadtxt(os.path.join(env_folder, "axis.txt"))
    aps = axis_data[:, :3]
    ds = [0]
    for i in range(1, len(aps)):
        ds.append(ds[-1] + np.linalg.norm(aps[i, :] - aps[i - 1, :]))
    avs = axis_data[:, 3:6]
    for test in tests:
        base_path_to_files = os.path.join(env_folder, "network_tests", test)
        path_to_json = base_path_to_files + ".json"
        path_to_test_data = base_path_to_files + ".txt"
        with open(path_to_json, "r") as f:
            info = json.load(f)
        test_data = np.loadtxt(path_to_test_data)
        # Check if the files meets the requirements
        n_measurements = info["n_measurements"]
        rel_yaw_period = info["rel_yaw_period"]
        rel_yaw_amplitude = info["rel_yaw_amplitude_deg"]
        h_disp_period = info["h_disp_period"]
        h_disp_amplitude = info["h_disp_amplitude"]
        if (
            (
                not desired_n_measurements is None
                and desired_n_measurements != n_measurements
            )
            or (
                not desired_rel_yaw_period is None
                and desired_rel_yaw_period != rel_yaw_period
            )
            or (
                not desired_rel_yaw_amplitude is None
                and desired_rel_yaw_amplitude != rel_yaw_amplitude
            )
            or (
                not desired_h_disp_period is None
                and desired_h_disp_period != h_disp_period
            )
            or (
                not desired_h_disp_amplitude is None
                and desired_h_disp_amplitude != h_disp_amplitude
            )
        ):
            continue
        ps = test_data[:, :3]
        rpys = test_data[:, 3:6]
        rel_yaws = test_data[:, 6]
        h_disps = test_data[:, 7]
        vs = rpy_to_vector(rpys)
        measured_yaws = test_data[:, 8 : (8 + n_measurements)]
        assert len(aps) == len(avs) == len(ps) == len(rpys) == len(measured_yaws)
        # Plot the axis and path followed
        if plot_axis_and_path:
            plotter = pv.Plotter()
            aps_polydata = pv.PolyData(aps)
            aps_polydata["vectors"] = avs
            avs_arrows = aps_polydata.glyph(orient="vectors")
            plotter.add_mesh(aps_polydata, color="r")
            plotter.add_mesh(avs_arrows, color="r")
            ps_polydata = pv.PolyData(ps)
            ps_polydata["vectors"] = vs
            vs_arrows = ps_polydata.glyph(orient="vectors")
            plotter.add_mesh(ps_polydata, color="b")
            plotter.add_mesh(vs_arrows, color="b")
            plotter.show()
        # Plot the errors
        fig = plt.figure()
        mean_measured_yaws = np.mean(measured_yaws, axis=1)
        standard_dev_measured_yaws = np.std(measured_yaws, axis=1)
        plt.plot(
            ds,
            np.rad2deg(mean_measured_yaws),
            color="b",
            label="Predicted relative yaw",
        )
        plt.plot(ds, np.rad2deg(rel_yaws), color="r", label="Relative yaw ground truth")
        plt.plot(ds, h_disps, color="g", label="Relative horizontal displacement")
        plt.fill_between(
            ds,
            np.rad2deg(mean_measured_yaws) - np.rad2deg(standard_dev_measured_yaws),
            np.rad2deg(mean_measured_yaws) + np.rad2deg(standard_dev_measured_yaws),
            alpha=0.5,
            color="b",
        )
        plt.ylim(-60, 80)
        plt.xlabel("Meters", fontsize=15)
        plt.ylabel("Yaw (deg)", fontsize=15)
        plt.tick_params("both", labelsize=14)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
