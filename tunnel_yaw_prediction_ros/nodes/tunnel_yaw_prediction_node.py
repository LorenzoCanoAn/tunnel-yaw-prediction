#!/home/lorenzo/miniconda3/envs/train_nn/bin/python3
import pathlib
import torch
from cv_bridge import CvBridge
import sensor_msgs.msg as sensor_msg
import std_msgs.msg as std_msg
import rospy
import importlib
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser("yaw_estimation_node")
    parser.add_argument("--path_to_model", required=True, type=str)
    parser.add_argument(
        "--input_topic", required=False, type=str, default="/cenital_image"
    )
    parser.add_argument(
        "--output_topic", required=False, type=str, default="~estimated_yaw"
    )
    parser.add_argument(
        "--model_module",
        required=False,
        type=str,
        default="tunnel_yaw_prediction.models",
    )
    parser.add_argument("--model_type", required=False, type=str, default=None)
    args = parser.parse_args()
    return args


def load_model(model_path, module, model_type):
    file_name = pathlib.Path(model_path).name
    if model_type == None:
        model_type = file_name.split("-")[0]
    module = importlib.import_module(module)
    model = getattr(module, model_type)()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


class NetworkNode:
    def __init__(self, model, input_topic, output_topic):
        self.model = model
        self.input_topic = input_topic
        self.output_topic = output_topic
        rospy.init_node(
            "tunnel_yaw_prediction_node",
        )
        self.init_network()
        self._cv_bridge = CvBridge()
        self.image_subscriber = rospy.Subscriber(
            input_topic,
            sensor_msg.Image,
            self.image_callback,
            queue_size=1,
        )
        self.detection_publisher = rospy.Publisher(
            output_topic, std_msg.Float32, queue_size=1
        )

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
