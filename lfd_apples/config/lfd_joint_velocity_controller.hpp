// Copyright (c) 2023 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <vector>
#include <std_msgs/msg/float64_multi_array.hpp>


#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {

/**
 * The joint velocity example controller
 */
class LfdJointVelocityController : public controller_interface::ControllerInterface {
 public:
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  std::string arm_id_;
  std::string robot_description_;
  bool is_gazebo{false};
  const int num_joints = 7;
  rclcpp::Duration elapsed_time_ = rclcpp::Duration(0, 0);

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr velocity_sub_;
  // Velocity limits for S-ramp
  double max_velocity_ = 0.5;      // rad/s, configurable via ROS params
  double max_acceleration_ = 0.5;  // rad/s^2, configurable via ROS params
  double max_jerk_ = 2.0;          // rad/s^3, optional, for smoother ramp

  // Internal ramp state
  std::vector<double> current_velocities_;
  std::vector<double> target_velocities_;
  std::vector<double> commanded_velocities_;


  bool command_received_ = false;

};

}  // namespace franka_example_controllers
