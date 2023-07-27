import argparse
import rospy
import gazebo_msgs.srv as gazebo_msgs_srv
import gazebo_msgs.msg as gazebo_msgs_msg
import std_msgs.msg as std_msg
import geometry_msgs.msg as geometry_msg
import numpy as np
import time
import os
from subt_proc_gen.geometry import Vector3D, get_two_perpendicular_vectors
import math
import json


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, type=str)
    parser.add_argument("--multiple_envs_in_folder", required=True, type=int)
    parser.add_argument("--yaw_prediction_topic", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--n_measurements", type=int, const=3, default=3, nargs="?")
    parser.add_argument(
        "--rel_yaw_amplitude_deg",
        type=float,
        # const=50,
        default=50,
        nargs="*",
    )
    parser.add_argument(
        "--rel_yaw_period",
        type=float,
        const=3,
        default=3,
        nargs="?",
    )
    parser.add_argument(
        "--h_disp_amplitude",
        type=float,
        # const=1,
        default=1,
        nargs="*",
    )
    parser.add_argument(
        "--h_disp_period",
        type=float,
        const=3,
        default=3,
        nargs="?",
    )
    args = parser.parse_known_args()[0]
    return args


def load_axis_data(file_path):
    data = np.loadtxt(file_path)
    aps = data[:, :3]
    avs = data[:, 3:6]
    return aps, avs


class YawStorage:
    def __init__(self):
        self._switch = True

    def callback(self, msg: std_msg.Float32):
        self.data = msg.data
        self._switch = False

    def block(self):
        self._switch = True
        while self._switch:
            time.sleep(0.2)


class YawPredictionTestingNode:
    def __init__(
        self,
        aps,
        avs,
        fta_dist,
        n_measurements,
        model_name,
        yaw_prediction_topic,
        environment_folder,
        rel_yaw_amplitude,
        rel_yaw_period,
        h_disp_amplitude,
        h_disp_period,
    ):
        self.aps = aps
        self.avs = avs
        self.fta_dist = fta_dist
        self.n_measurements = n_measurements
        self.model_name = model_name
        self.environment_folder = environment_folder
        self.rel_yaw_amplitude = rel_yaw_amplitude
        self.rel_yaw_period = rel_yaw_period
        self.h_disp_amplitude = h_disp_amplitude
        self.h_disp_period = h_disp_period
        # Obtain the distance between the axis points
        self.intra_point_dist = np.reshape(
            np.linalg.norm(self.aps[0, :] - self.aps[1, :]), -1
        ).item(0)
        rospy.init_node("yaw_prediction_testing_node")
        rospy.wait_for_service("/gazebo/set_model_state")
        self.gzb_pose_setter = rospy.ServiceProxy(
            "/gazebo/set_model_state", gazebo_msgs_srv.SetModelState, persistent=True
        )
        self.yaw_storage = YawStorage()
        rospy.Subscriber(
            yaw_prediction_topic, std_msg.Float32, callback=self.yaw_storage.callback
        )
        self.spawn_model_srv_proxy = rospy.ServiceProxy(
            "/gazebo/spawn_sdf_model", gazebo_msgs_srv.SpawnModel, persistent=True
        )
        self.delete_model_srv_proxy = rospy.ServiceProxy(
            "/gazebo/delete_model", gazebo_msgs_srv.DeleteModel
        )

    def h_disp(self, x):
        """Return the horizontal displacent at a dist x form the start of the tunnel"""
        return self.h_disp_amplitude * np.cos(x / self.h_disp_period)

    def rel_yaw(self, x):
        """Return the relative yaw at dist x from the start of the tunnel"""
        return self.rel_yaw_amplitude * np.cos(x / self.rel_yaw_period)

    def generate_testing_poses(self):
        """Returns an array of shape [N, 6], Each row represnts x,y,z,roll,pitch,yaw"""
        n_aps = len(self.aps)
        testing_poses = np.zeros((n_aps, 6))
        yaws_rel = np.zeros((n_aps, 1))
        h_disps = np.zeros((n_aps, 1))
        for n, (ap, av) in enumerate(zip(self.aps, self.avs)):
            # Dist from start of tunnel
            dist = n * self.intra_point_dist
            # Get displacements
            h_disp = self.h_disp(dist)
            yaw_rel = self.rel_yaw(dist)
            # Calculate T from origin to ap
            robot_pose = robot_pose_from_axis_and_displacement(
                ap, av, h_disp, self.fta_dist, 0, 0, yaw_rel
            )
            testing_poses[n, :] = robot_pose
            yaws_rel[n, :] = yaw_rel
            h_disps[n, :] = h_disp
        testing_poses = np.hstack([testing_poses, yaws_rel, h_disps])
        return testing_poses

    def change_environment(self, file_to_model):
        with open(file_to_model, "r") as f:
            model_text = f.read()
        delete_request = gazebo_msgs_srv.DeleteModelRequest()
        delete_request.model_name = "cave"
        self.delete_model_srv_proxy.call(delete_request)
        time.sleep(1)
        spawn_request = gazebo_msgs_srv.SpawnModelRequest()
        spawn_request.model_name = "cave"
        spawn_request.model_xml = model_text
        spawn_request.reference_frame = ""
        self.spawn_model_srv_proxy.call(spawn_request)

    def run(self):
        self.change_environment(os.path.join(self.environment_folder, "model.sdf"))
        testing_poses = self.generate_testing_poses()
        testing_data = []
        for x, y, z, roll, pitch, yaw, _, _ in testing_poses:
            qx, qy, qz, qw = get_quaternion_from_euler(roll, pitch, yaw)
            position = geometry_msg.Point(x, y, z)
            orientation = geometry_msg.Quaternion(qx, qy, qz, qw)
            pose = geometry_msg.Pose(position, orientation)
            twist = geometry_msg.Twist()
            model_state = gazebo_msgs_msg.ModelState("/", pose, twist, "")
            pose_request = gazebo_msgs_srv.SetModelStateRequest(model_state)
            self.gzb_pose_setter.call(pose_request)
            predicted_yaws = []
            for _ in range(self.n_measurements):
                self.yaw_storage.block()
                predicted_yaws.append(self.yaw_storage.data)
            testing_data.append(predicted_yaws)
        return testing_poses, np.array(testing_data)

    def save_data(self, tests_data_folder, test_number, testing_poses, predicted_yaws):
        general_array = np.hstack([testing_poses, predicted_yaws])
        info = {
            "n_measurements": self.n_measurements,
            "rel_yaw_amplitude_deg": int(np.round(np.rad2deg(self.rel_yaw_amplitude))),
            "rel_yaw_period": self.rel_yaw_period,
            "h_disp_amplitude": self.h_disp_amplitude,
            "h_disp_period": self.h_disp_period,
            "model_name": self.model_name,
        }
        np.savetxt(
            os.path.join(tests_data_folder, f"test_{test_number:04d}.txt"),
            general_array,
        )
        with open(
            os.path.join(tests_data_folder, f"test_{test_number:04d}.json"), "w+"
        ) as f:
            json.dump(info, f)


def main():
    args = get_args()
    folder = args.folder
    yaw_prediction_topic = args.yaw_prediction_topic
    n_measurements = args.n_measurements
    rel_yaw_amplitudes = np.deg2rad(args.rel_yaw_amplitude_deg)
    model_name = args.model_name
    rel_yaw_period = args.rel_yaw_period
    h_disp_amplitudes = args.h_disp_amplitude
    h_disp_period = args.h_disp_period
    multiple_envs_in_folder = args.multiple_envs_in_folder
    # Handle the environment folders
    if not multiple_envs_in_folder:
        environment_folders = [folder]
    else:
        environment_folders = [
            os.path.join(folder, env_folder) for env_folder in os.listdir(folder)
        ]
    # Handle the different amplitudes
    if len(rel_yaw_amplitudes) == len(h_disp_amplitudes):
        amplitudes = []
        for rel_yaw_amp, h_disp_amp in zip(rel_yaw_amplitudes, h_disp_amplitudes):
            amplitudes.append((rel_yaw_amp, h_disp_amp))
    elif len(rel_yaw_amplitudes) == 1 and len(h_disp_amplitudes) > 1:
        amplitudes = []
        for h_disp_amp in h_disp_amplitudes:
            amplitudes.append((rel_yaw_amplitudes[0], h_disp_amp))
    elif len(rel_yaw_amplitudes) > 1 and len(h_disp_amplitudes) == 1:
        amplitudes = []
        for rel_yaw_amp in rel_yaw_amplitudes:
            amplitudes.append((rel_yaw_amp, h_disp_amplitudes[0]))
    else:
        raise Exception(
            "If there is a different number of each amplitude, there has to be only one of them"
        )
    environment_folders.sort()
    for environment_folder in environment_folders:
        print(environment_folder)
        axis_data_file_path = os.path.join(environment_folder, "axis.txt")
        fta_file_path = os.path.join(environment_folder, "fta_dist.txt")
        aps, avs = load_axis_data(axis_data_file_path)
        fta_dist = np.reshape(np.loadtxt(fta_file_path), -1).item(0)
        test_results_folder = os.path.join(environment_folder, "network_tests")
        if not os.path.isdir(test_results_folder):
            os.mkdir(test_results_folder)
        for rel_yaw_amplitude, h_disp_amplitude in amplitudes:
            print(
                f"yaw_amp: {rel_yaw_amplitude:05f}, h_disp_amp: {h_disp_amplitude:05f}"
            )
            n_tests = int(len(os.listdir(test_results_folder)) / 2)
            test_number = n_tests + 1
            yaw_prediction_testing_node = YawPredictionTestingNode(
                aps,
                avs,
                fta_dist,
                n_measurements,
                model_name,
                yaw_prediction_topic,
                environment_folder,
                rel_yaw_amplitude,
                rel_yaw_period,
                h_disp_amplitude,
                h_disp_period,
            )
            test_poses, test_data = yaw_prediction_testing_node.run()
            yaw_prediction_testing_node.save_data(
                test_results_folder, test_number, test_poses, test_data
            )


if __name__ == "__main__":
    main()
