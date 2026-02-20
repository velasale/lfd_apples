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

#include <franka_example_controllers/default_robot_behavior_utils.hpp>
#include <franka_example_controllers/lfd_joint_velocity_controller.hpp>
#include <franka_example_controllers/robot_utils.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <string>

#include <Eigen/Eigen>

namespace franka_example_controllers {

// Command Interface configuration
controller_interface::InterfaceConfiguration
LfdJointVelocityController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

// State Interface configuration
controller_interface::InterfaceConfiguration
LfdJointVelocityController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

// Main update loop
controller_interface::return_type LfdJointVelocityController::update(
    const rclcpp::Time&,
    const rclcpp::Duration& period)
{
  double dt = period.seconds();

  for (int i = 0; i < num_joints; ++i) {

    // 1. Acceleration-limited delta
    double error = target_velocities_[i] - current_velocities_[i];
    double max_delta = max_acceleration_ * dt;

    if (std::abs(error) > max_delta) {
      current_velocities_[i] += std::copysign(max_delta, error);
    } else {
      current_velocities_[i] = target_velocities_[i];
    }

    // 2. Velocity limit
    if (current_velocities_[i] > max_velocity_) current_velocities_[i] = max_velocity_;
    if (current_velocities_[i] < -max_velocity_) current_velocities_[i] = -max_velocity_;


    command_interfaces_[i].set_value(current_velocities_[i]);
  }

  return controller_interface::return_type::OK;
}



// ON-Init: Initialize controller
CallbackReturn LfdJointVelocityController::on_init() {
  try {
    auto_declare<bool>("gazebo", false);
    auto_declare<std::string>("robot_description", "");
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

// Configure controller
CallbackReturn LfdJointVelocityController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  is_gazebo = get_node()->get_parameter("gazebo").as_bool();

  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
  parameters_client->wait_for_service();

  auto future = parameters_client->get_parameters({"robot_description"});
  auto result = future.get();
  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
    if (robot_description_.empty()) {
      RCLCPP_ERROR(get_node()->get_logger(), "robot_description parameter is empty.");
      return CallbackReturn::ERROR;
    }
  } else {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
  }

  arm_id_ = robot_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());

  current_velocities_.assign(num_joints, 0.0);
  commanded_velocities_.assign(num_joints, 0.0);
  target_velocities_.assign(num_joints, 0.0);
  get_node()->get_parameter_or("max_velocity", max_velocity_, max_velocity_);
  get_node()->get_parameter_or("max_acceleration", max_acceleration_, max_acceleration_);
  get_node()->get_parameter_or("max_jerk", max_jerk_, max_jerk_);

  // Subscribe to velocity command topic
  velocity_sub_ =
    get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/fr3_joint_velocity_controller/lfd_fr3_joint_velocity_cmd",
      10,
      [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg)
      {
        if (msg->data.size() != static_cast<size_t>(num_joints)) {
          RCLCPP_WARN(get_node()->get_logger(),
                      "Velocity command size mismatch: expected %d, got %zu",
                      num_joints, msg->data.size());
          return;
        }
        target_velocities_ = msg->data;
        commanded_velocities_ = msg->data;
        command_received_ = true;
      });


  if (!is_gazebo) {
    auto client = get_node()->create_client<franka_msgs::srv::SetFullCollisionBehavior>(
        "service_server/set_full_collision_behavior");
    auto request = DefaultRobotBehavior::getDefaultCollisionBehaviorRequest();

    auto future_result = client->async_send_request(request);
    future_result.wait_for(robot_utils::time_out);

    auto success = future_result.get();
    if (!success) {
      RCLCPP_FATAL(get_node()->get_logger(), "Failed to set default collision behavior.");
      return CallbackReturn::ERROR;
    } else {
      RCLCPP_INFO(get_node()->get_logger(), "Default collision behavior set.");
    }
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn LfdJointVelocityController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  elapsed_time_ = rclcpp::Duration(0, 0);
  return CallbackReturn::SUCCESS;
}

}  // namespace franka_example_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::LfdJointVelocityController,
                       controller_interface::ControllerInterface)
