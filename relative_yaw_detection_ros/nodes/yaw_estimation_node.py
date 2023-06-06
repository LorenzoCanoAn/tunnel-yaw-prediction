#!/bin/python3
import pathlib
import torch
from cv_bridge import CvBridge
import sensor_msgs.msg as sensor_msg
import std_msgs.msg as std_msg
import rospy
import importlib
import numpy as np


class NetworkNode:
    def __init__(self):
        rospy.init_node(
            "gallery_network",
        )
        self.init_network()
        self._cv_bridge = CvBridge()
        self.image_subscriber = rospy.Subscriber(
            "/cenital_image",
            sensor_msg.Image,
            self.image_callback,
            queue_size=1,
        )
        self.detection_publisher = rospy.Publisher(
            "/estimated_relative_yaw", std_msg.Float32, queue_size=1
        )

    def init_network(self):
        file_path = rospy.get_param(
            "~nn_path",
            default="/home/lorenzo/models/yaw_estimation/YawEstimator-_bs128_ne128_lr0_001346.torch",
        )
        print(file_path)
        file_name = pathlib.Path(file_path).name
        nn_type = file_name.split("-")[0]
        module = importlib.import_module("yaw_estimation.models")
        self.model = getattr(module, nn_type)()
        self.model.load_state_dict(
            torch.load(file_path, map_location=torch.device("cpu"))
        )
        self.model.eval()

    def image_callback(self, msg: sensor_msg.Image):
        depth_image = np.reshape(
            np.frombuffer(msg.data, dtype=np.float32), (msg.height, msg.width)
        )
        depth_image_tensor = torch.tensor(depth_image).float().to(torch.device("cpu"))
        depth_image_tensor /= torch.max(depth_image_tensor)
        depth_image_tensor = torch.reshape(depth_image_tensor, [1, 1, 30, -1])
        data = self.model(depth_image_tensor)
        data = data.cpu().detach().numpy()
        yaw = data.item(0)
        output_message = std_msg.Float32(yaw)
        print(np.rad2deg(yaw))
        self.detection_publisher.publish(output_message)


def main():
    network_node = NetworkNode()
    rospy.loginfo("Gallery network beguinning to spin")
    rospy.spin()


if __name__ == "__main__":
    main()
