import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import json
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs_folder", required=True, type=str)
    return parser.parse_args()


def get_envs_to_plot(path_to_envs_folder):
    env_names = os.listdir(path_to_envs_folder)
    env_names_with_results = []
    for env_name in env_names:
        path_to_env = os.path.join(path_to_envs_folder, env_name)
        path_to_tests_folder = os.path.join(path_to_env, "complete_system_tests")
        if len(os.listdir(path_to_tests_folder)) > 0:
            env_names_with_results.append(env_name)
    for n_env_name, env_name in enumerate(env_names_with_results):
        print(f"{n_env_name}: {env_name}")
    what_envs_to_plot = input(
        "Write numbers of environmts to plot, or 'all' to plot every available env: \n\t"
    )
    if what_envs_to_plot == "all":
        idx_of_what_envs_to_plot = list(range(len(env_names_with_results)))
    else:
        idx_of_what_envs_to_plot = [int(i) for i in what_envs_to_plot.split(",")]
    return [env_names_with_results[i] for i in idx_of_what_envs_to_plot]


def select_desired_max_vels(path_to_folder_with_test_data):
    max_vels = set()
    for file_name in os.listdir(path_to_folder_with_test_data):
        if ".json" in file_name:
            path_to_info_file = os.path.join(path_to_folder_with_test_data, file_name)
            with open(path_to_info_file, "r") as f:
                info = json.load(f)
                max_vel = info["max_vel"]
                max_vels.add(max_vel)
    max_vels = list(max_vels)
    for n_max_vel, max_vel in enumerate(max_vels):
        print(f"{n_max_vel}: {max_vel}")
    what_vels_to_plot = "all"  # input("Write numbers of environmts to plot, or 'all' to plot every available env: \n\t")
    if what_vels_to_plot == "all":
        idx_of_what_max_vels = list(range(len(max_vels)))
    else:
        idx_of_what_max_vels = [int(i) for i in what_vels_to_plot.split(",")]
    return [max_vels[i] for i in idx_of_what_max_vels]


def get_files_with_desired_max_vel(path_to_folder_with_test_data, desired_max_vel):
    file_names = []
    for file_name in os.listdir(path_to_folder_with_test_data):
        if ".json" in file_name:
            path_to_file = os.path.join(path_to_folder_with_test_data, file_name)
            with open(path_to_file, "r") as f:
                info = json.load(f)
                if info["max_vel"] == desired_max_vel:
                    data_file_name = file_name.replace(".json", ".txt")
                    file_names.append(data_file_name)
    return file_names


def generate_points_of_the_walls(aps, avs, mesh: trimesh.Trimesh):
    points_of_the_walls = np.zeros((len(aps) * 2, 3))
    n = 0
    for ap, av in zip(aps, avs):
        vx, vy, vz = av
        yaw = np.arctan2(vy, vx)
        yaw1 = yaw + np.deg2rad(90)
        yaw2 = yaw - np.deg2rad(90)
        vx1 = np.cos(yaw1)
        vy1 = np.sin(yaw1)
        vx2 = np.cos(yaw2)
        vy2 = np.sin(yaw2)
        ints_1, _, _ = mesh.ray.intersects_location(
            np.reshape(ap, (1, 3)), np.array(((vx1, vy1, 0),))
        )
        ints_2, _, _ = mesh.ray.intersects_location(
            np.reshape(ap, (1, 3)), np.array(((vx2, vy2, 0),))
        )
        idx_of_closest_int_1 = np.argmin(np.linalg.norm(ints_1 - ap, axis=1))
        idx_of_closest_int_2 = np.argmin(np.linalg.norm(ints_2 - ap, axis=1))
        int_1 = ints_1[idx_of_closest_int_1, :]
        int_2 = ints_2[idx_of_closest_int_2, :]
        points_of_the_walls[n, :] = int_1
        n += 1
        points_of_the_walls[n, :] = int_2
        n += 1
    return points_of_the_walls


def get_env_geometric_data(path_to_env_folder):
    path_to_axis_data = os.path.join(path_to_env_folder, "axis.txt")
    path_to_mesh = os.path.join(path_to_env_folder, "mesh.obj")
    path_to_points_of_the_walls = os.path.join(
        path_to_env_folder, "tunnel_wall_points.txt"
    )
    axis_data = np.loadtxt(path_to_axis_data)
    aps = axis_data[:, 0:3]
    avs = axis_data[:, 3:6]
    if True:  # not os.path.isfile(path_to_points_of_the_walls_1):
        mesh = trimesh.load_mesh(path_to_mesh)
        points_of_the_walls = generate_points_of_the_walls(aps, avs, mesh)
        np.savetxt(path_to_points_of_the_walls, points_of_the_walls)
    else:
        points_of_the_walls_1 = np.loadtxt(path_to_points_of_the_walls_1)
    return aps, avs, points_of_the_walls


def get_axis_lims_from_points(points):
    max_x = np.max(points[:, 0])
    min_x = np.min(points[:, 0])
    max_y = np.max(points[:, 1])
    min_y = np.min(points[:, 1])
    x_range = max_x - min_x
    y_range = max_y - min_y
    if x_range > y_range:
        f_max_x = max_x
        f_min_x = min_x
        f_max_y = (max_y + min_y) / 2 + x_range / 2
        f_min_y = (max_y + min_y) / 2 - x_range / 2
    else:
        f_max_y = max_y
        f_min_y = min_y
        f_max_x = (max_x + min_x) / 2 + y_range / 2
        f_min_x = (max_x + min_x) / 2 - y_range / 2
    return f_min_x, f_max_x, f_min_y, f_max_y


def get_fig_size_from_points(points, env_name):
    max_x = np.max(points[:, 0])
    min_x = np.min(points[:, 0])
    max_y = np.max(points[:, 1])
    min_y = np.min(points[:, 1])
    x_range = max_x - min_x
    y_range = max_y - min_y
    if x_range > y_range:
        fig_size = (10, 10 * y_range / x_range)
    else:
        fig_size = (10 * x_range / y_range, 10)
    if "cv" in env_name:
        fig_size = (fig_size[1], fig_size[0])
    return fig_size


def plot_multicolor_line(list_of_i_points, list_of_values, ax, fig, env_name):
    segments = np.zeros((0, 2, 2))
    for i_points in list_of_i_points:
        x = i_points[:, 0]
        y = i_points[:, 1]
        if "cv" in env_name:
            x_temp = x
            x = y
            y = x_temp
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        print(np.hstack([points[:-1], points[1:]]).shape)
        segments = np.vstack([segments, np.hstack([points[:-1], points[1:]])])
    values = np.zeros(0)
    for val in list_of_values:
        values = np.concatenate((values, val[:-1]))
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, values.max())
    lc = LineCollection(segments, cmap="viridis", norm=norm)
    # Set the values used for colormapping
    lc.set_array(values)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    cbar = fig.colorbar(line, ax=ax)
    cbar.ax.tick_params(labelsize=30)


def plot_square(ax, fig, x, y, width):
    w = width / 2
    p1 = (x - w, y - w)
    p2 = (x + w, y - w)
    p3 = (x + w, y + w)
    p4 = (x - w, y + w)
    array = np.array((p1, p2, p3, p4, p1))
    plt.plot(array[:, 0], array[:, 1], color="k", linewidth=4)


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


def main():
    args = get_args()
    envs_folder = args.envs_folder
    for env_name in get_envs_to_plot(envs_folder):
        if "cv" in env_name:
            a = 1
            b = 0
        else:
            a = 0
            b = 1
        path_to_env_folder = os.path.join(envs_folder, env_name)
        path_to_test_data = os.path.join(path_to_env_folder, "complete_system_tests")
        aps, avs, points_of_the_walls = get_env_geometric_data(path_to_env_folder)
        max_vels_to_plot = select_desired_max_vels(path_to_test_data)
        # Begin plotting
        for max_vel in max_vels_to_plot:
            figure = plt.figure(
                figsize=get_fig_size_from_points(points_of_the_walls, env_name)
            )
            ax = figure.add_subplot(111)
            plt.sca(ax)
            plt.plot(aps[:, a], aps[:, b], color="r", linewidth=2)
            plt.plot(
                points_of_the_walls[::2, a],
                points_of_the_walls[::2, b],
                color="k",
                linewidth=5,
            )
            plt.plot(
                points_of_the_walls[1::2, a],
                points_of_the_walls[1::2, b],
                color="k",
                linewidth=5,
            )
            print(f"Plotting tests with max_vel: {max_vel}")
            files_to_plot = get_files_with_desired_max_vel(path_to_test_data, max_vel)
            test_data = []
            distances_to_axis = []
            for n_file, file_name in enumerate(files_to_plot):
                if n_file == 0:
                    path_to_info = os.path.join(
                        path_to_test_data, file_name.replace(".txt", ".json")
                    )
                    with open(path_to_info, "r") as f:
                        info = json.load(f)
                    if "obstacle_poses" in info:
                        for obstacle_pose in info["obstacle_poses"]:
                            x, y, z = obstacle_pose
                            if "cv" in env_name:
                                x_temp = x
                                x = y
                                y = x_temp
                            plot_square(ax, figure, x, y, 1)
                else:
                    break
                path_to_file = os.path.join(path_to_test_data, file_name)
                _test_data = np.loadtxt(path_to_file)
                test_data.append(_test_data)
                distances_to_axis.append(
                    calculate_distances_to_axis(test_data[-1][:, :3], aps, avs)
                )
            plot_multicolor_line(test_data, distances_to_axis, ax, figure, env_name)
            # plt.plot(test_data[:, 0], test_data[:, 1], color="b")
            # min_x, max_x, min_y, max_y = get_axis_lims_from_points(points_of_the_walls)
            # ax.set_xlim(min_x, max_x)
            # ax.set_ylim(min_y, max_y)
            plt.yticks(fontsize=30)
            plt.xticks(fontsize=30)
            plt.draw()
            max_vel_str = str(max_vel)
            max_vel_str = max_vel_str.replace(".", "_")
            filename = f"trc_{env_name}_{max_vel_str}.pdf"
            plt.savefig(
                f"/home/lorenzo/Documents/my_papers/ROBOT2023/images/{filename}"
            )
            plt.close()


if __name__ == "__main__":
    main()
