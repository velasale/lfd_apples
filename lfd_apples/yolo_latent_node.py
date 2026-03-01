#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
import numpy as np
import math

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



def transform_point_image_to_frame(px, py, tx, ty, theta):
    c = math.cos(theta)
    s = math.sin(theta)

    T_FI = np.array([
        [ c,  s, -(c*tx + s*ty)],
        [-s,  c,  (s*tx - c*ty)],
        [ 0,  0,  1]
    ])

    p_I = np.array([px, py, 1.0])

    p_F = T_FI @ p_I
    return int(p_F[0]), int(p_F[1])


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


    def palm_camera_callback(self, msg:Image, publish_latent_vector=True, publish_bbox_center=False):

        # --- Process Image ---
        raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img_w = msg.width
        img_h = msg.height            

        # --- Draw Crosshair at image center
        # Transform from Image cFrame to TCP cframe
        cx, cy = img_w // 2, img_h // 2
        length = 70
        angle = math.radians(45)
        dx = int(length * math.cos(angle))
        dy = int(length * math.sin(angle))
        label_offset_x = 10
        label_offset_y = 5

        # X axis (45°)
        x1 = (cx - dx, cy - dy)
        x2 = (cx + dx, cy + dy)
        
        cv2.circle(raw_image, (cx, cy), radius=25, color=(0, 255, 0), thickness=1)

        cv2.line(raw_image, x1, x2, color=(0, 255, 0), thickness=1)
        # X-axis label (placed past positive direction)
        cv2.putText(raw_image, "X axis",
                    (x2[0] + label_offset_x, x2[1] - label_offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)

        # Y axis (-45°)
        y1 = (cx + dx, cy - dy)
        y2 = (cx -+ dx, cy + dy)

        cv2.line(raw_image, y1, y2, color=(0, 255, 0), thickness=1)

        # Y-axis label
        cv2.putText(raw_image, "Y axis",
                    (y2[0] + label_offset_x, y2[1] + label_offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)


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
            
            bbox_center_at_tcp = [-1,-1]

            if bbox_center:                                             

                # Book keeping
                self.bbox_cxs.append(bbox_center[0])
                self.bbox_cys.append(bbox_center[1])

                if len(self.bbox_cxs) > self.filter_window:                    
                    # Remove oldest center
                    self.bbox_cxs.pop(0)
                    self.bbox_cys.pop(0)

                    # Hampel Filter
                    filtered_cx = int(hampel(np.array(self.bbox_cxs), window_size=5, n_sigma=1.0).filtered_data[-1].item())
                    # Invert y from camera sensor to real world
                    filtered_cy  = - int(hampel(np.array(self.bbox_cys), window_size=5, n_sigma=1.0).filtered_data[-1].item())

                    # # Moving Average Filter                   
                    # filtered_cx = int(np.mean(self.bbox_cxs))
                    # filtered_cy = - int(np.mean(self.bbox_cys))     # Invert Y axis for robot frame                 

                    bbox_center = [filtered_cx, filtered_cy]

                # Bounding Box center at Image frame:                
                # self.get_logger().info(f"BBox center at Image frame: {bbox_center}\n")               
                cx_img_frame = int(bbox_center[0])
                cy_img_frame = - int(bbox_center[1])             # Invert Y axis back for img frame 
                cv2.circle(raw_image, (cx_img_frame, cy_img_frame), radius=5, color=(0, 0, 255), thickness=-1)
                cv2.putText(raw_image, f"Bbox center at img frame: ({cx_img_frame},{cy_img_frame})", (20,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # Bounding Box center at TCP frame:
                cx_tcp_frame, cy_tcp_frame = transform_point_image_to_frame(cx_img_frame, cy_img_frame, img_w/2, img_h/2, theta=angle)               
                # self.get_logger().info(f"Bbox center @ tcp frame: {cx_tcp_frame}, {cy_tcp_frame}\n")       
                cv2.putText(raw_image, f"({cx_tcp_frame},{cy_tcp_frame})", (cx_img_frame + 10, cy_img_frame),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)               

                bbox_center_at_tcp = [cx_tcp_frame, cy_tcp_frame]                

            else:            
                self.get_logger().warn(f"No bounding boxes dectected")
                bbox_center = [-1, -1]            

            bbox_center_msg = Int16MultiArray()
            bbox_center_msg.data = bbox_center_at_tcp
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