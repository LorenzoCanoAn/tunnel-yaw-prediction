import rospy
import os
import argparse
import json
import numpy as np
import std_msgs.msg as std_msgs
from gazebo_msgs.srv import (
    SetModelState,
    SetModelStateRequest,
    SetModelStateResponse,
    GetModelState,
    GetModelStateRequest,
    GetModelStateResponse,
    SpawnModel,
    SpawnModelRequest,
    SpawnModelResponse,
    DeleteModel,
    DeleteModelRequest,
    DeleteModelResponse,
)
from std_msgs.msg import Bool
import geometry_msgs
import math
import threading
from time import time_ns, sleep


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z  # in radians


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_folder", required=True)
    parser.add_argument("--n_trials_per_env", required=True, type=int)
    parser.add_argument("--max_vel", required=True, type=float)
    parser.add_argument("--max_ang_vel", required=True, type=float)
    return parser.parse_args()


class EnvFolderHandler:
    def __init__(self, path_to_env):
        self._path_to_env = path_to_env
        self._path_to_testing_data = os.path.join(path_to_env, "complete_system_tests")
        if not os.path.isdir(self._path_to_testing_data):
            os.mkdir(self._path_to_testing_data)
        self.path_to_model = os.path.join(self._path_to_env, "model.sdf")
        self.path_to_axis_data = os.path.join(self._path_to_env, "axis.txt")
        self.path_to_fta_dist = os.path.join(self._path_to_env, "fta_dist.txt")

    def save_new_test(self, test_data: np.ndarray, test_info: dict):
        n_previous_tests = int(len(os.listdir(self._path_to_testing_data)) / 2)
        test_name = f"test_{n_previous_tests+1:04d}"
        path_to_data_file = os.path.join(self._path_to_testing_data, f"{test_name}.txt")
        path_to_info_file = os.path.join(
            self._path_to_testing_data, f"{test_name}.json"
        )
        np.savetxt(path_to_data_file, test_data)
        with open(path_to_info_file, "w") as f:
            json.dump(test_info, f)


class PoseRecorder:
    def __init__(
        self, get_model_pose_proxy: rospy.ServiceProxy, get_model_request, frequency=10
    ):
        self.get_model_state_proxy = get_model_pose_proxy
        self.get_model_request = get_model_request
        self.period = 1 / frequency
        self.current_pose_n = 0
        self.recorded_poses = np.zeros([100, 6])
        self.switch = False
        self.thread = threading.Thread(target=self.recording_thread_target)
        self.thread.start()

    def __del__(self):
        self.switch = False
        self.thread.join()

    def add_recorded_pose(self, new_pose):
        if self.current_pose_n == len(self.recorded_poses):
            self.recorded_poses = np.vstack([self.recorded_poses, np.zeros([100, 6])])
        self.recorded_poses[self.current_pose_n, :] = new_pose
        self.current_pose_n += 1

    @property
    def last_recorded_pose(self):
        return self.recorded_poses[self.current_pose_n - 1, :]

    def get_recorded_poses(self):
        return self.recorded_poses[: self.current_pose_n, :]

    def reset_recorded_poses(self):
        self.current_pose_n = 0

    def stop_recording(self):
        self.switch = False

    def start_recording(self):
        self.switch = True

    def recording_thread_target(self):
        while not rospy.is_shutdown():
            start_time = time_ns()
            if self.switch:
                pose = self.get_model_state_proxy.call(self.get_model_request)
                assert isinstance(pose, GetModelStateResponse)
                x = pose.pose.position.x
                y = pose.pose.position.y
                z = pose.pose.position.z
                qx = pose.pose.orientation.x
                qy = pose.pose.orientation.y
                qz = pose.pose.orientation.z
                qw = pose.pose.orientation.w
                r, p, yw = euler_from_quaternion(qx, qy, qz, qw)
                self.add_recorded_pose(np.array((x, y, z, r, p, yw)))
            end_time = time_ns()
            elapsed_time = end_time - start_time
            time_to_sleep = self.period - elapsed_time
            if time_to_sleep > 0:
                sleep(time_to_sleep)


class DirToAxisTestingNode:
    ####################################################################################################################################
    # 	Setup functions
    ####################################################################################################################################

    def __init__(
        self,
        env_folder,
        n_trials_per_env,
        max_vel,
        max_ang_vel,
    ):
        self.path_to_env_folder = env_folder
        self.n_trials_per_env = n_trials_per_env
        self.max_vel = max_vel
        self.max_ang_vel = max_ang_vel
        self.init_ros_node()

    def init_ros_node(self):
        rospy.init_node("system_test_node")
        self.set_pose_proxy = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState, persistent=True
        )
        self.get_pose_proxy = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState, persistent=True
        )
        self.spawn_model_proxy = rospy.ServiceProxy(
            "/gazebo/spawn_sdf_model", SpawnModel, persistent=True
        )
        self.delete_model_proxy = rospy.ServiceProxy(
            "/gazebo/delete_model", DeleteModel, persistent=True
        )
        self.change_max_vel_pub = rospy.Publisher(
            "/tunnel_traversal/new_max_vel",
            std_msgs.Float32,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.change_max_ang_vel_pub = rospy.Publisher(
            "/tunnel_traversal/new_max_ang_vel",
            std_msgs.Float32,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.stop_robot_pub = rospy.Publisher("/obstacle_detected", Bool, queue_size=1)
        self.get_pose_msg = GetModelStateRequest("/", "")
        self.pose_recorder = PoseRecorder(self.get_pose_proxy, self.get_pose_msg)

    ####################################################################################################################################
    # 	Gazebo Functions
    ####################################################################################################################################
    def change_environment(self, file_to_model):
        with open(file_to_model, "r") as f:
            model_text = f.read()
        delete_request = DeleteModelRequest()
        delete_request.model_name = "cave"
        self.delete_model_proxy.call(delete_request)
        sleep(2)
        spawn_request = SpawnModelRequest()
        spawn_request.model_name = "cave"
        spawn_request.model_xml = model_text
        spawn_request.reference_frame = ""
        response: DeleteModelResponse = self.spawn_model_proxy.call(spawn_request)

    def spawn_sdf_model(self, file_to_model, name, x, y, z):
        with open(file_to_model, "r") as f:
            model_text = f.read()
        delete_request = DeleteModelRequest()
        delete_request.model_name = name
        self.delete_model_proxy.call(delete_request)
        sleep(2)
        spawn_request = SpawnModelRequest()
        spawn_request.model_name = name
        spawn_request.model_xml = model_text
        spawn_request.reference_frame = ""
        spawn_request.initial_pose.position.x = x
        spawn_request.initial_pose.position.y = y
        spawn_request.initial_pose.position.z = z
        response: DeleteModelResponse = self.spawn_model_proxy.call(spawn_request)

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
        response = self.set_pose_proxy(request)

    def run(self):
        sleep(1)
        rospy.loginfo("Changing vels")
        self.change_max_vel_pub.publish(std_msgs.Float32(self.max_vel))
        self.change_max_ang_vel_pub.publish(std_msgs.Float32(self.max_ang_vel))
        sleep(1)
        print(f"Environment: {self.path_to_env_folder}")
        folder_handler = EnvFolderHandler(self.path_to_env_folder)
        axis_data = np.loadtxt(folder_handler.path_to_axis_data)
        aps = axis_data[:, 0:3]
        avs = axis_data[:, 3:6]
        fta_dist = np.loadtxt(folder_handler.path_to_fta_dist)
        rx, ry, rz, rr, rp, ryw = np.reshape(
            robot_pose_from_axis_and_displacement(
                aps[0, :], avs[0, :], 0, fta_dist + 0.2, 0, 0, 0
            ),
            -1,
        )
        info = {
            "max_vel": self.max_vel,
            "max_ang_vel": self.max_ang_vel,
        }
        # Get the poses of the obstacles
        obstacle_poses = [
            (10.443672180175781, -1.5932906866073608, -0.53411865234375),
            (10.4772, -0.788059, -0.534119),
            (20.844310760498047, 0.8689512014389038, -0.5341190099716187),
            (20.897977828979492, 1.8596277236938477, -0.5341190099716187),
            (25.653221130371094, 7.827459335327148, -0.5341190099716187),
            (28.1561, 5.7178, -0.534119),
            (31.9947, 10.6694, -0.534119),
            (28.87476348876953, 6.761902809143066, -0.5341190099716187),
            (23.9272, 3.84301, -0.534119),
        ]
        for n_obstacle, obstacle_pose in enumerate(obstacle_poses):
            x, y, z = obstacle_pose
            model_name = f"obstacle_{n_obstacle:03d}"
            self.spawn_sdf_model(
                "/home/lorenzo/model_editor_models/st_unit_box/model.sdf",
                model_name,
                x,
                y,
                z,
            )
        self.change_environment(folder_handler.path_to_model)
        print(obstacle_poses)
        for n_test in range(self.n_trials_per_env):
            print(f"Test: {n_test}")
            self.stop_robot_pub.publish(Bool(True))
            self.send_position(rx, ry, rz, rr, rp, ryw)
            sleep(1)
            self.send_position(rx, ry, rz, rr, rp, ryw)
            sleep(1)
            # The robot is now stopped and at the beguinning of the tunnel
            self.pose_recorder.start_recording()
            sleep(0.5)
            self.stop_robot_pub.publish(Bool(False))
            # The robot should now start moving
            prev_pose = None
            while True:
                current_pose = self.pose_recorder.last_recorded_pose
                robot_xyz = current_pose[:3]
                dist_to_end_of_tunnel = np.linalg.norm(robot_xyz - aps[-1, :])
                if dist_to_end_of_tunnel < 2:
                    print("Test Successful")
                    success = True
                    break
                # If roll or pitch are too high, there has been a failure
                if current_pose[3] > np.deg2rad(40) or current_pose[4] > np.deg2rad(40):
                    print("Test unsuccessful")
                    success = False
                    break
                if not prev_pose is None:
                    if np.linalg.norm(prev_pose - current_pose) < 0.01:
                        print("Test unsuccessful")
                        success = False
                        break
                    prev_pose = current_pose
                    sleep(0.1)
                else:
                    prev_pose = current_pose
                    sleep(0.2)
            info["success"] = success
            info["obstacle_poses"] = obstacle_poses
            self.pose_recorder.stop_recording()
            self.stop_robot_pub.publish(Bool(True))
            sleep(5)
            recorded_poses = self.pose_recorder.get_recorded_poses()
            folder_handler.save_new_test(recorded_poses, info)
            self.pose_recorder.reset_recorded_poses()


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


def main():
    args = get_args()
    node = DirToAxisTestingNode(
        args.env_folder,
        args.n_trials_per_env,
        args.max_vel,
        args.max_ang_vel,
    )
    main_thread = threading.Thread(target=node.run)
    main_thread.start()
    rospy.spin()
    main_thread.join()


if __name__ == "__main__":
    main()
