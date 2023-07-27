#!/home/lorenzo/miniconda3/envs/train_nn/bin/python3
import pathlib
import torch
from cv_bridge import CvBridge
import sensor_msgs.msg as sensor_msg
import std_msgs.msg as std_msg
import rospy
import importlib
import numpy as np
import os
import json


def load_model_by_string(model_path, module, model_type):
    file_name = pathlib.Path(model_path).name
    if model_type == None:
        model_type = file_name.split("-")[0]
    module = importlib.import_module(module)
    model = getattr(module, model_type)()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


class NetworkNode:
    def __init__(self):
        rospy.init_node(
            "tunnel_yaw_prediction_node",
        )
        self._cv_bridge = CvBridge()
        self.setup_params()
        self.setup_model()
        self.setup_sub_and_pub()

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

    def get_param_from_pram_sub_key(self, param_sub_key):
        return rospy.get_param(self.find_matching_param_key(param_sub_key))

    def setup_params(self):
        model_folder = rospy.get_param("~model_folder")
        model_name = rospy.get_param("~model_name")
        self.input_topic = rospy.get_param("~input_topic", default="/cenital_image")
        self.output_topic = rospy.get_param("~output_topic", default="~estimated_yaw")
        self.publish_also_in_deg = rospy.get_param("~publish_also_in_deg", default=True)
        self.path_to_model = os.path.join(model_folder, model_name + ".torch")
        self.path_to_model_info = os.path.join(model_folder, model_name + ".json")

    def setup_model(self):
        rospy.loginfo(f"loading {self.path_to_model}")
        with open(self.path_to_model_info, "r") as f:
            self.model_info = json.load(f)
        self.model = load_model_by_string(
            self.path_to_model,
            self.model_info["module_to_import_network"],
            self.model_info["network_class_name"],
        )
        training_img_size = self.model_info["dataset_paramters"]["conversor/img_size"]
        training_max_coord_val = self.model_info["dataset_paramters"][
            "conversor/max_coord_val"
        ]
        recieved_img_size = self.get_param_from_pram_sub_key("conversor/img_size")
        recieved_max_coord_val = self.get_param_from_pram_sub_key(
            "conversor/max_coord_val"
        )
        if training_img_size == recieved_img_size:
            self.img_size = training_img_size
        else:
            raise Exception(
                f"The recieved img size is {recieved_img_size}, but the training img_size is {training_img_size}"
            )
        if training_max_coord_val == recieved_max_coord_val:
            self.max_coord_val = training_max_coord_val
        else:
            raise Exception(
                f"The recieved max_coord_val {recieved_max_coord_val}, but the training max_coord_val is {training_max_coord_val}"
            )

    def setup_sub_and_pub(self):
        self.image_subscriber = rospy.Subscriber(
            self.input_topic,
            sensor_msg.Image,
            self.image_callback,
            queue_size=1,
        )
        self.predicted_yaw_rad_publisher = rospy.Publisher(
            self.output_topic, std_msg.Float32, queue_size=1
        )
        if self.publish_also_in_deg:
            self.predicted_yaw_deg_publisher = rospy.Publisher(
                self.output_topic + "_deg", std_msg.Float32, queue_size=1
            )

    def image_callback(self, msg: sensor_msg.Image):
        depth_image = np.reshape(
            np.frombuffer(msg.data, dtype=np.float32), (msg.height, msg.width)
        )
        depth_image_tensor = torch.tensor(depth_image).float().to(torch.device("cpu"))
        depth_image_tensor /= torch.max(depth_image_tensor)
        depth_image_tensor = torch.reshape(
            depth_image_tensor, [1, 1, self.img_size, self.img_size]
        )
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
    network_node = NetworkNode()
    network_node.run()


if __name__ == "__main__":
    main()
