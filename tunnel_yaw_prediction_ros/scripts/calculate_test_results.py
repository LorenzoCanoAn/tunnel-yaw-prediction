import os
import json
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_envs_folder", required=True, type=str)
    return parser.parse_args()


def calculate_distances_to_axis(positions, aps, avs):
    distances = np.zeros(len(positions))
    n = 0
    for position in positions:
        ap_to_ax_dist = np.linalg.norm(aps - position.reshape(1, 3), axis=1)
        idx = np.argmin(ap_to_ax_dist)
        ap = aps[idx]
        av = avs[idx]
        ap_to_p = position - ap
        av2d = av[:2]
        av2d /= np.linalg.norm(av2d)
        perp_to_av2d = np.array([av2d[1], -av2d[0]])
        ap_to_p2d = ap_to_p[:2]
        ap_to_p2d_proj_in_av2d = np.dot(perp_to_av2d, ap_to_p2d)
        distances[n] = np.linalg.norm(ap_to_p2d_proj_in_av2d)
        n += 1
    return distances


def get_available_max_vels(path_to_folder_with_test_data):
    max_vels = set()
    for file_name in os.listdir(path_to_folder_with_test_data):
        if ".json" in file_name:
            path_to_info_file = os.path.join(path_to_folder_with_test_data, file_name)
            with open(path_to_info_file, "r") as f:
                info = json.load(f)
                max_vel = info["max_vel"]
                max_vels.add(max_vel)
    max_vels = list(max_vels)
    max_vels.sort()
    return max_vels


def get_test_names_with_desired_max_vel(path_to_folder_with_test_data, desired_max_vel):
    test_names = []
    for file_name in os.listdir(path_to_folder_with_test_data):
        if ".json" in file_name:
            path_to_file = os.path.join(path_to_folder_with_test_data, file_name)
            with open(path_to_file, "r") as f:
                info = json.load(f)
                if info["max_vel"] == desired_max_vel:
                    test_name = file_name.replace(".json", "")
                    test_names.append(test_name)
    return test_names


def main():
    args = get_args()
    envs_folder = args.path_to_envs_folder
    env_names = os.listdir(envs_folder)
    env_names.sort()
    for env_name in env_names:
        env_name_string = env_name.replace("_", "\\_")
        print(
            f"\\multirow{'{'}3{'}'}{'{'}4em{'}'}{'{'}\\texttt{'{'}{env_name_string}{'}'}{'}'}"
        )
        path_to_env = os.path.join(envs_folder, env_name)
        path_to_axis_data = os.path.join(path_to_env, "axis.txt")
        axis_data = np.loadtxt(path_to_axis_data)
        aps = axis_data[:, :3]
        avs = axis_data[:, 3:6]
        path_to_env_test_data_folder = os.path.join(
            path_to_env, "complete_system_tests"
        )
        for max_vel in get_available_max_vels(path_to_env_test_data_folder):
            test_data = np.zeros((0, 6))
            for test_name in get_test_names_with_desired_max_vel(
                path_to_env_test_data_folder, max_vel
            ):
                path_to_test_data = os.path.join(
                    path_to_env_test_data_folder, test_name + ".txt"
                )
                test_data = np.vstack((test_data, np.loadtxt(path_to_test_data)))
            test_positions = test_data[:, :3]
            test_distances_to_axis = calculate_distances_to_axis(
                test_positions, aps, avs
            )
            max_distance = np.max(test_distances_to_axis)
            avg_distance = np.average(test_distances_to_axis)
            median_distance = np.median(test_distances_to_axis)
            print(f"&{int(max_vel)}&{max_distance:.02f}&{avg_distance:.02f}\\\\")
        print("\\hline")


if __name__ == "__main__":
    main()
