#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
import numpy as np

# ROS2 imports
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int16MultiArray

# Computer Vision Imports
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

# Signal Processing Imports
from hampel import hampel

# Custom Imports
from lfd_apples.lfd_vision import extract_pooled_latent_vector, bounding_box_centers


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
        
        self.bbox_center_pub = self.create_publisher(
            Int16MultiArray,
            'lfd/bbox_center',
            10)  
        
        self.palm_camera_pub = self.create_publisher(
            Image, 
            'gripper/rgb_palm_camera/image_raw_with_artifacts', 
            10)      


        # Flags:
        self.publish_latent_vector = False
        self.publish_bbox_center = False


        # Computer Vision
        self.bridge = CvBridge()
        self.bbox_cxs = []
        self.bbox_cys = []
        self.filter_window = 10
        

        # Get the absolute path to the 'resources' folder in your package
        package_share_dir = get_package_share_directory('lfd_apples')
        pt_path = os.path.join(package_share_dir, 'resources', 'best_segmentation.pt')        
        self.yolo_model = YOLO(pt_path)
        self.yolo_model.to("cuda")  # Use GPU if available
        print(self.yolo_model.device)
        self.yolo_latent_layer = 15       

        self.expected_latent_dim = 64

        self.get_logger().info("YOLO latent vector node initialized.")


    def palm_camera_callback(self, msg:Image, publish_latent_vector=True, publish_bbox_center=True):

        # --- Process Image ---
        raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img_w = msg.width
        img_h = msg.height            

        # Draw Crosshair at image center
        cv2.line(raw_image, (img_w//2 - 10, img_h//2), (img_w//2 + 10, img_h//2), color=(0, 255, 0), thickness=1)
        cv2.line(raw_image, (img_w//2, img_h//2-10), (img_w//2, img_h//2+10), color=(0, 255, 0), thickness=1)

        if publish_latent_vector:
            # --- Extract Latent Features ---        
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


        if publish_bbox_center:
            # --- Publish Bounding Box Center ---
            bbox_center = bounding_box_centers(
                raw_image,
                self.yolo_model)                                     
                    
            if bbox_center:

                bbox_center[0] = int(bbox_center[0] - img_w/2)
                bbox_center[1] = int(bbox_center[1] - img_h/2)                

                # Book keeping
                self.bbox_cxs.append(bbox_center[0])
                self.bbox_cys.append(bbox_center[1])

                if len(self.bbox_cxs) > self.filter_window:                    
                    # Remove oldest center
                    self.bbox_cxs.pop(0)
                    self.bbox_cys.pop(0)

                    # Hampel Filter
                    filtered_cx = int(hampel(np.array(self.bbox_cxs), window_size=5, n_sigma=1.0).filtered_data[-1].item())
                    filtered_cy  = - int(hampel(np.array(self.bbox_cys), window_size=5, n_sigma=1.0).filtered_data[-1].item())

                    # # Moving Average Filter                   
                    # filtered_cx = int(np.mean(self.bbox_cxs))
                    # filtered_cy = - int(np.mean(self.bbox_cys))     # Invert Y axis for robot frame                 

                    bbox_center = [filtered_cx, filtered_cy]

                else:
                    filtered_cx = int(bbox_center[0])
                    filtered_cy = - int(bbox_center[1])             # Invert Y axis for robot frame      
                
                # Draw a circle at the bounding box center                
                img_cx = int(filtered_cx + img_w/2)
                img_cy = int(- filtered_cy + img_h/2)              # Invert Y axis for image frame

                cv2.circle(raw_image, (img_cx, img_cy), radius=5, color=(0, 0, 255), thickness=-1)  # red dot
                cv2.putText(raw_image, f"Center: ({filtered_cx},{filtered_cy})", (img_cx + 10, img_cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)                

            else:            
                self.get_logger().warn(f"No bounding boxes dectected")
                bbox_center = [-1, -1]            

            bbox_center_msg = Int16MultiArray()
            bbox_center_msg.data = bbox_center
            # # bbox_center_msg.data = centers_array.flatten().tolist()
            self.bbox_center_pub.publish(bbox_center_msg)
        
        img_msg = Image()
        img_msg.data = raw_image.flatten().tolist()
        img_msg.height = img_h
        img_msg.width = img_w
        img_msg.encoding = "bgr8"
        img_msg.step = img_w * 3  # 3 bytes per pixel for B
        self.palm_camera_pub.publish(img_msg)

        # --- Uncomment for debugging purposes ---
        # cv2.imshow("Palm Camera", raw_image)
        # cv2.waitKey(1)  #

        # self.get_logger().info("Latent vector published.")   

    


def main(args=None):
    rclpy.init(args=args)
    yolo_latent_vector = YoloLatentVector()
    rclpy.spin(yolo_latent_vector)
    yolo_latent_vector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()