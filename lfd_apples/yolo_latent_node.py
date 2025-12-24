#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

# ROS2 imports
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

# Computer Vision Imports
from cv_bridge import CvBridge
from ultralytics import YOLO

# Custom Imports
from lfd_apples.lfd_vision import extract_pooled_latent_vector


class YoloLatentVector(Node):

    def __init__(self):
        super().__init__('yolo_latent_vector')

        # Topic subscriptions
        self.palm_camera_sub = self.create_subscription(
            Image, 
            'gripper/rgb_palm_camera/image_raw', 
            self.palm_camera_callback,
            1)      

        # Topic Publishers
        self.latent_vector_pub = self.create_publisher(
            Float32MultiArray,
            'lfd/latent_image',
            10)  

        # Computer Vision
        self.bridge = CvBridge()

        # Get the absolute path to the 'resources' folder in your package
        package_share_dir = get_package_share_directory('lfd_apples')
        pt_path = os.path.join(package_share_dir, 'resources', 'best_segmentation.pt')        
        self.yolo_model = YOLO(pt_path)
        self.yolo_latent_layer = 15       

        self.expected_latent_dim = 64

        self.get_logger().info("YOLO latent vector node initialized.")


    def palm_camera_callback(self, msg:Image):

        # --- Process Image ---
        raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Get latent space features
        pooled_vector, feat_map = extract_pooled_latent_vector(
            raw_image,
            self.yolo_model,
            layer_index=self.yolo_latent_layer
        )

        if len(pooled_vector) != self.expected_latent_dim:
            self.get_logger().warn(
                f"Latent vector size mismatch. Expected {self.expected_latent_dim}, got {len(pooled_vector)}"
            )
            return
        
        # --- Publish Latent Vector ---
        latent_vector_msg = Float32MultiArray()
        latent_vector_msg.data = pooled_vector.tolist()
        self.latent_vector_pub.publish(latent_vector_msg)

        # self.get_logger().info("Latent vector published.")   

    


def main(args=None):
    rclpy.init(args=args)
    yolo_latent_vector = YoloLatentVector()
    rclpy.spin(yolo_latent_vector)
    yolo_latent_vector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()