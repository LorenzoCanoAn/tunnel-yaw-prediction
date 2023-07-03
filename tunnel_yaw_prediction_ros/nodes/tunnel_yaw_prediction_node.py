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
    parser = argparse.ArgumentParser("tunnel_yaw_prediction_node")
    parser.add_argument("--path_to_model", required=True, type=str)
    parser.add_argument(
        "--input_topic",
        type=str,
        default="/cenital_image",
        const="/cenital_image",
        nargs="?",
    )
    parser.add_argument(
        "--output_topic",
        type=str,
        default="~estimated_yaw",
        const="~estimated_yaw",
        nargs="?",
    )
    parser.add_argument(
        "--model_module",
        type=str,
        default="tunnel_yaw_prediction.models",
        const="tunnel_yaw_prediction.models",
        nargs="?",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        const=None,
        nargs="?",
    )
    parser.add_argument(
        "--publish_also_in_deg",
        type=int,
        default=1,
        const=1,
        nargs="?",
    )
    args, trash = parser.parse_known_args()
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
    def __init__(self, model, input_topic, output_topic, publish_also_in_deg):
        self.model = model
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.publish_also_in_deg = publish_also_in_deg
        rospy.init_node(
            "tunnel_yaw_prediction_node",
        )
        self._cv_bridge = CvBridge()
        self.image_subscriber = rospy.Subscriber(
            input_topic,
            sensor_msg.Image,
            self.image_callback,
            queue_size=1,
        )
        self.predicted_yaw_rad_publisher = rospy.Publisher(
            output_topic, std_msg.Float32, queue_size=1
        )
        if publish_also_in_deg:
            self.predicted_yaw_deg_publisher = rospy.Publisher(
                output_topic + "_deg", std_msg.Float32, queue_size=1
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
        self.predicted_yaw_rad_publisher.publish(output_message)
        if self.publish_also_in_deg:
            output_message = std_msg.Float32(np.rad2deg(yaw))
            self.predicted_yaw_deg_publisher.publish(output_message)

    def run(self):
        rospy.loginfo("Gallery network beguinning to spin")
        rospy.spin()


def main():
    args = get_args()
    path_to_model = args.path_to_model
    input_topic = args.input_topic
    output_topic = args.output_topic
    model_module = args.model_module
    model_type = args.model_type
    publish_also_in_deg = bool(args.publish_also_in_deg)
    model = load_model(path_to_model, model_module, model_type)
    network_node = NetworkNode(model, input_topic, output_topic, publish_also_in_deg)
    network_node.run()


if __name__ == "__main__":
    main()
