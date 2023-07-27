#!/home/lorenzo/miniconda3/envs/train_nn/bin/python
import numpy as np
import os
import rospy
from gazebo_msgs.srv import (
    SpawnModel,
    SpawnModelRequest,
    SpawnModelResponse,
    SetModelState,
    SetModelStateRequest,
    SetModelStateResponse,
    DeleteModel,
    DeleteModelRequest,
    DeleteModelResponse,
)
import geometry_msgs
import sensor_msgs
from cv_bridge import CvBridge
import time
import math
import json
import datetime


def get_transformation_matrix(x, y, z, roll, pitch, yaw):
    c_roll = np.cos(roll)
    s_roll = np.sin(roll)
    c_pitch = np.cos(pitch)
    s_pitch = np.sin(pitch)
    c_yaw = np.cos(yaw)
    s_yaw = np.sin(yaw)
    R_x = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])
    R_y = np.array([[c_pitch, 0, s_pitch], [0, 1, 0], [-s_pitch, 0, c_pitch]])
    R_z = np.array([[c_yaw, -s_yaw, 0], [s_yaw, c_yaw, 0], [0, 0, 1]])
    rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
    translation_vector = np.array([x, y, z])
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    return transformation_matrix


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return qx, qy, qz, qw


def xyzrpy_to_T(x, y, z, roll, pitch, yaw):
    c_roll = np.cos(roll)
    s_roll = np.sin(roll)
    c_pitch = np.cos(pitch)
    s_pitch = np.sin(pitch)
    c_yaw = np.cos(yaw)
    s_yaw = np.sin(yaw)
    R_x = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])
    R_y = np.array([[c_pitch, 0, s_pitch], [0, 1, 0], [-s_pitch, 0, c_pitch]])
    R_z = np.array([[c_yaw, -s_yaw, 0], [s_yaw, c_yaw, 0], [0, 0, 1]])
    rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
    translation_vector = np.array([x, y, z])
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    return transformation_matrix


def T_to_xyzrpy(transform):
    # Extract translation components
    x = transform[0, 3]
    y = transform[1, 3]
    z = transform[2, 3]
    # Extract rotation components
    roll = math.atan2(transform[2, 1], transform[2, 2])
    pitch = math.atan2(
        -transform[2, 0], math.sqrt(transform[2, 1] ** 2 + transform[2, 2] ** 2)
    )
    yaw = math.atan2(transform[1, 0], transform[0, 0])
    # Convert rotation angles to degrees
    return x, y, z, roll, pitch, yaw


def robot_pose_from_axis_and_displacement(
    ap, av, h_disp, v_disp, rel_roll, rel_pitch, rel_yaw
):
    """Given an axis point and its corresponding direction, as well as
    the robot's relative horizontal displacement and orientation,
    an array of xyzrpy of the robot in global coordinates"""
    apx, apy, apz = np.reshape(ap, -1)
    avx, avy, avz = np.reshape(av, -1)
    a_roll = 0
    a_pitch = -np.arctan2(avz, np.sqrt(avx**2 + avy**2))
    a_yaw = np.arctan2(avy, avx)
    axis_T = xyzrpy_to_T(apx, apy, apz, a_roll, a_pitch, a_yaw)
    # Create T matrix from displacements
    x_rel = 0
    y_rel = h_disp
    z_rel = v_disp
    rel_pitch = 0
    rel_T = xyzrpy_to_T(x_rel, y_rel, z_rel, rel_roll, rel_pitch, rel_yaw)
    # Obtain the final from the two transformation matrices
    final_T = np.matmul(axis_T, rel_T)
    return np.array(T_to_xyzrpy(final_T))


def label_from_pose_and_aps(
    robot_xyzrpy: np.ndarray, aps: np.ndarray, detection_radius
):
    robot_xyzrpy = np.reshape(robot_xyzrpy, -1)
    x, y, z, roll, pitch, yaw = robot_xyzrpy
    robot_T = xyzrpy_to_T(x, y, z, roll, pitch, yaw)
    inv_robot_T = np.linalg.inv(robot_T)
    robot_xyz = np.reshape(np.array((x, y, z)), (1, 3))
    d_of_aps_to_circle = np.abs(
        np.linalg.norm(aps - robot_xyz, axis=1) - detection_radius
    )
    # Iterate until an aps close to the circle is in the adequate direction
    while True:
        idx_of_candidate = np.argmin(d_of_aps_to_circle)
        candidate_aps = np.reshape(aps[idx_of_candidate, :], (1, 3))
        candidate_aps_unitary = np.hstack([candidate_aps, np.ones((1, 1))]).T
        # Project the candidate aps into the robots reference frame
        apx, apy, apz = np.matmul(inv_robot_T, candidate_aps_unitary)[0:3, 0]
        yaw_of_ap_in_robot_frame = np.arctan2(apy, apx)
        # If the aps is in front of the robot, that is the label, else, try with the next one
        if np.abs(yaw_of_ap_in_robot_frame) < np.pi / 2:
            return yaw_of_ap_in_robot_frame
        else:
            d_of_aps_to_circle = np.delete(d_of_aps_to_circle, idx_of_candidate, axis=0)
            aps = np.delete(aps, idx_of_candidate, axis=0)


class AxisPointManager:
    def __init__(self, axis_points, axis_vectors, voxel_size=5):
        self.axis_points = axis_points
        self.axis_vectors = axis_vectors
        self.voxel_size = voxel_size
        self.grid = dict()
        max_x = max(axis_points[:, 0])
        min_x = min(axis_points[:, 0])
        max_y = max(axis_points[:, 1])
        min_y = min(axis_points[:, 1])
        max_z = max(axis_points[:, 2])
        min_z = min(axis_points[:, 2])
        max_i = int(np.ceil(max_x / voxel_size)) + 3
        min_i = int(np.floor(min_x / voxel_size)) - 3
        max_j = int(np.ceil(max_y / voxel_size)) + 3
        min_j = int(np.floor(min_y / voxel_size)) - 3
        max_k = int(np.ceil(max_z / voxel_size)) + 3
        min_k = int(np.floor(min_z / voxel_size)) - 3
        for i in range(min_i, max_i):
            for j in range(min_j, max_j):
                for k in range(min_k, max_k):
                    self.grid[(i, j, k)] = np.zeros([0, 6])
        ijks = np.floor(self.axis_points / voxel_size).astype(int)
        for ijk, ap, av in zip(ijks, self.axis_points, self.axis_vectors):
            i, j, k = ijk
            ap = np.reshape(ap, (-1, 3))
            av = np.reshape(av, (-1, 3))
            self.grid[(i, j, k)] = np.concatenate(
                [self.grid[(i, j, k)], np.concatenate((ap, av), axis=1)], axis=0
            )

    def get_relevant_points(self, xyz):
        _i, _j, _k = np.floor(xyz / self.voxel_size).astype(int)
        relevant_points = np.zeros((0, 6))
        for i in (_i - 1, _i, _i + 1):
            for j in (_j - 1, _j, _j + 1):
                for k in (_k - 1, _k, _k + 1):
                    relevant_points = np.concatenate(
                        [relevant_points, self.grid[(i, j, k)]], axis=0
                    )
        return relevant_points


class DirToAxisPoseAndLabelGenerator:
    def __init__(
        self,
        axis_points,
        axis_vectors,
        max_hor_disp,
        max_vert_disp,
        min_vert_disp,
        fta_dist,
        max_inclination_rad,
        rel_yaw_range_rad,
        label_distance,
    ):
        self.aps = axis_points
        self.avs = axis_vectors
        self.max_hor_disp = max_hor_disp
        self.fta_dist = fta_dist
        self.max_vert_disp = max_vert_disp
        self.min_vert_disp = min_vert_disp
        self.max_inc_rad = max_inclination_rad
        self.rel_yaw_range_rad = rel_yaw_range_rad
        self.label_distance = label_distance
        # This type of dataset cannot use points that are too close tho the ends
        # of the tunnel, so it has to cut them off
        avg_distance_between_aps = np.mean(
            np.linalg.norm(self.aps[:-1, :] - self.aps[1:, :], axis=1), axis=0
        )
        self.n_aps_to_ignore = int(self.label_distance / avg_distance_between_aps) + 1

    def gen_one_sample(self):
        base_point_idx = np.random.randint(
            self.n_aps_to_ignore, len(self.aps) - self.n_aps_to_ignore
        )
        ap = np.reshape(self.aps[base_point_idx, :], (1, 3))
        av = np.reshape(self.avs[base_point_idx, :], (1, 3))
        h_disp = self.max_hor_disp * np.random.uniform(-1, 1)
        v_disp = np.random.uniform(
            self.min_vert_disp + self.fta_dist, self.max_vert_disp + self.fta_dist
        )
        rel_roll = np.random.uniform(-self.max_inc_rad, self.max_inc_rad)
        rel_pitch = np.random.uniform(-self.max_inc_rad, self.max_inc_rad)
        rel_yaw = np.random.uniform(-self.rel_yaw_range_rad, self.rel_yaw_range_rad)
        robot_xyzrpy = robot_pose_from_axis_and_displacement(
            ap, av, h_disp, v_disp, rel_roll, rel_pitch, rel_yaw
        )
        label = label_from_pose_and_aps(robot_xyzrpy, self.aps, self.label_distance)
        return robot_xyzrpy, label

    def gen_n_samples(self, n_samples):
        robot_xyzrpys = np.zeros((n_samples, 6))
        labels = np.zeros((n_samples, 1))
        for i in range(n_samples):
            robot_xyzrpy, label = self.gen_one_sample()
            robot_xyzrpys[i, :] = robot_xyzrpy
            labels[i, :] = label
        return robot_xyzrpys, labels


class ImageStorage:
    def __init__(self, image_topic):
        self._sub = rospy.Subscriber(
            image_topic, sensor_msgs.msg.Image, callback=self.callback
        )
        self._switch = True
        self._brdg = CvBridge()

    def callback(self, msg):
        self.image = np.frombuffer(msg.data, dtype=np.float32).reshape(
            msg.height,
            msg.width,
        )
        self._switch = False

    def block(self):
        self._switch = True
        while self._switch:
            time.sleep(0.2)


class DatasetRecorderNode:
    private_param_keys = (
        "data_folder",
        "number_of_samples_per_env",
        "max_rel_yaw_deg",
        "label_distance",
        "max_horizontal_displacement",
        "min_vertical_displacement",
        "max_vertical_displacement",
        "max_inclination_deg",
        "robot_name",
        "image_topic",
    )
    external_param_keys = ("conversor/max_coord_val", "conversor/img_size")

    def __init__(self):
        self.counter = 0
        rospy.init_node("dataset_collector")
        date = datetime.datetime.now()
        self.name = f"{date.year}-{date.month:02d}-{date.day:02d}_{date.hour:02d}:{date.minute:02d}:{date.second:02d}"
        self.get_private_parameters()
        self.get_external_parameters()
        self.set_sub_pub()

    def find_matching_param_key(self, desired_param_key):
        all_param_keys = rospy.get_param_names()
        matching_params = []
        for param_key in all_param_keys:
            if desired_param_key in param_key:
                matching_params.append(param_key)
        if len(matching_params) == 1:
            return matching_params[0]
        elif len(matching_params) == 0:
            raise Exception(
                f"Sub-param name '{desired_param_key}' does not match any params in server"
            )
        else:
            raise Exception(
                f"Sub-param name '{desired_param_key}' matches more than one params in server"
            )

    def get_private_parameters(self):
        self.private_params = dict()
        for private_parameter_key in self.private_param_keys:
            self.private_params[private_parameter_key] = rospy.get_param(
                "~" + private_parameter_key
            )

    def get_external_parameters(self):
        self.external_params = dict()
        for external_parameter_key in self.external_param_keys:
            self.external_params[external_parameter_key] = rospy.get_param(
                self.find_matching_param_key(external_parameter_key)
            )

    def set_sub_pub(self):
        self.image_storage = ImageStorage(self.private_params["image_topic"])
        self.set_pose_srv_proxy = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState, persistent=True
        )
        self.spawn_model_srv_proxy = rospy.ServiceProxy(
            "/gazebo/spawn_sdf_model", SpawnModel, persistent=True
        )
        self.delete_model_srv_proxy = rospy.ServiceProxy(
            "/gazebo/delete_model", DeleteModel
        )

    def send_position(self, x, y, z, roll, pitch, yaw):
        position = geometry_msgs.msg.Point(x, y, z)
        qx, qy, qz, qw = get_quaternion_from_euler(roll, pitch, yaw)
        orientation = geometry_msgs.msg.Quaternion(qx, qy, qz, qw)
        pose = geometry_msgs.msg.Pose(position, orientation)
        twist = geometry_msgs.msg.Twist(
            geometry_msgs.msg.Vector3(0, 0, 0), geometry_msgs.msg.Vector3(0, 0, 0)
        )
        request = SetModelStateRequest()
        request.model_state.model_name = "/"
        request.model_state.pose = pose
        request.model_state.twist = twist
        request.model_state.reference_frame = ""
        response = self.set_pose_srv_proxy(request)

    def record_dataset(self):
        data_folder = self.private_params["data_folder"]
        n_samples_per_env = self.private_params["number_of_samples_per_env"]
        save_folder = self.set_save_folder(data_folder)
        envs_folder = os.path.join(data_folder, "environments")
        env_folders = os.listdir(envs_folder)
        env_folders.sort()
        self.save_json_file(save_folder)
        n_datapoint = 0
        for env_folder in env_folders:
            print(env_folder)
            path_to_env_folder = os.path.join(envs_folder, env_folder)
            (
                model_file_path,
                axis_points,
                axis_vectors,
                fta_dist,
            ) = self.get_environment_data(path_to_env_folder)
            print(model_file_path)
            self.change_environment(model_file_path)
            # Generate the poses to obtain the datapoints
            pose_and_label_generator = DirToAxisPoseAndLabelGenerator(
                axis_points,
                axis_vectors,
                self.private_params["max_horizontal_displacement"],
                self.private_params["max_vertical_displacement"],
                self.private_params["min_vertical_displacement"],
                fta_dist,
                np.deg2rad(self.private_params["max_inclination_deg"]),
                np.deg2rad(self.private_params["max_rel_yaw_deg"]),
                self.private_params["label_distance"],
            )
            (
                robot_xyzrpys,
                labels,
            ) = pose_and_label_generator.gen_n_samples(n_samples_per_env)
            # Record the datapoints for the environment
            for i, (robot_xyzrpy, label) in enumerate(zip(robot_xyzrpys, labels)):
                print(f"{n_datapoint:07d}", end="\r", flush=True)
                x, y, z, roll, pitch, yaw = robot_xyzrpy
                self.send_position(x, y, z, roll, pitch, yaw)
                time.sleep(0.1)
                self.image_storage.block()
                image = self.image_storage.image
                np.savez(
                    os.path.join(save_folder, f"{n_datapoint:08d}.npz"),
                    image=image,
                    label=label,
                )
                n_datapoint += 1

    def get_environment_data(self, abs_env_folder):
        axis_file = os.path.join(abs_env_folder, "axis.txt")
        model_file = os.path.join(abs_env_folder, "model.sdf")
        fta_dist_file = os.path.join(abs_env_folder, "fta_dist.txt")
        axis_data = np.loadtxt(axis_file)
        axis_points = axis_data[:, :3]
        axis_vectors = axis_data[:, 3:6]
        fta_dist = np.loadtxt(fta_dist_file).item(0)
        return model_file, axis_points, axis_vectors, fta_dist

    def set_save_folder(self, data_folder):
        # Check if the data folder exists, if not, create it
        datasets_folder = os.path.join(data_folder, "training_data")
        dataset_folder = os.path.join(datasets_folder, self.name)
        os.makedirs(dataset_folder, exist_ok=True)
        return dataset_folder

    def save_json_file(self, dataset_folder):
        # Create the info file of the dataset
        dataset_params = {}
        dataset_params["name"] = self.name
        dataset_params["dataset_type"] = "dir_to_axis"
        for private_param_key in self.private_param_keys:
            dataset_params[private_param_key] = self.private_params[private_param_key]
        for external_param_key in self.external_param_keys:
            dataset_params[external_param_key] = self.external_params[
                external_param_key
            ]
        path_to_json = os.path.join(dataset_folder, "info.json")
        with open(path_to_json, "w") as f:
            json.dump(dataset_params, f)

    def change_environment(self, file_to_model):
        with open(file_to_model, "r") as f:
            model_text = f.read()
        delete_request = DeleteModelRequest()
        delete_request.model_name = "cave"
        self.delete_model_srv_proxy.call(delete_request)
        time.sleep(2)
        spawn_request = SpawnModelRequest()
        spawn_request.model_name = "cave"
        spawn_request.model_xml = model_text
        spawn_request.reference_frame = ""
        self.spawn_model_srv_proxy.call(spawn_request)


def main():
    dataset_recorder = DatasetRecorderNode()
    dataset_recorder.record_dataset()


if __name__ == "__main__":
    main()
