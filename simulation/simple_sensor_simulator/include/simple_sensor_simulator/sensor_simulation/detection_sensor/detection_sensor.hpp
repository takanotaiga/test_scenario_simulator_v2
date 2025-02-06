// Copyright 2015 TIER IV, Inc. All rights reserved.
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

#ifndef SIMPLE_SENSOR_SIMULATOR__SENSOR_SIMULATION__DETECTION_SENSOR__DETECTION_SENSOR_HPP_
#define SIMPLE_SENSOR_SIMULATOR__SENSOR_SIMULATION__DETECTION_SENSOR__DETECTION_SENSOR_HPP_

#include <simulation_api_schema.pb.h>

#include <geometry/plane.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

namespace simple_sensor_simulator
{
struct EntityNoiseStatus
{
  std::string entity_name;
  double last_published_time;
  double last_distance_noise{};
  double last_yaw_noise{};
  bool is_last_ground_truth_masked{};
  bool is_last_yaw_flipped{};
  
  EntityNoiseStatus() = default;
  EntityNoiseStatus(const std::string& name, double time = 0.0): entity_name(name), last_published_time(time) {}
};

class DetectionSensorBase
{
protected:
  double previous_simulation_time_;

  simulation_api_schema::DetectionSensorConfiguration configuration_;

  explicit DetectionSensorBase(
    const double current_simulation_time,
    const simulation_api_schema::DetectionSensorConfiguration & configuration)
  : previous_simulation_time_(current_simulation_time), configuration_(configuration)
  {
  }

  auto isEgoEntityStatusToWhichThisSensorIsAttached(
    const traffic_simulator_msgs::EntityStatus &) const -> bool;

  auto findEgoEntityStatusToWhichThisSensorIsAttached(
    const std::vector<traffic_simulator_msgs::EntityStatus> &) const
    -> std::vector<traffic_simulator_msgs::EntityStatus>::const_iterator;

  auto isOnOrAboveEgoPlane(
    const geometry_msgs::Pose & npc_pose, const geometry_msgs::Pose & ego_pose) -> bool;

public:
  virtual ~DetectionSensorBase() = default;

  virtual void update(
    const double current_simulation_time, const std::vector<traffic_simulator_msgs::EntityStatus> &,
    const rclcpp::Time & current_ros_time,
    const std::vector<std::string> & lidar_detected_entities) = 0;

private:
  std::optional<math::geometry::Plane> ego_plane_opt_{std::nullopt};
  std::optional<geometry_msgs::msg::Pose> ego_plane_pose_opt_{std::nullopt};
};

template <typename T, typename U = autoware_perception_msgs::msg::TrackedObjects>
class DetectionSensor : public DetectionSensorBase
{
  const typename rclcpp::Publisher<T>::SharedPtr detected_objects_publisher;

  const typename rclcpp::Publisher<U>::SharedPtr ground_truth_objects_publisher;

  /*---------------------------------------------------*/
  std::default_random_engine random_engine_;

  std::queue<std::pair<std::vector<traffic_simulator_msgs::EntityStatus>, double>>
    unpublished_detected_entities, unpublished_ground_truth_entities;
  /* a structure to save noise info*/

  std::unordered_map<std::string, EntityNoiseStatus> entity_noise_status_map;
  /*----temp utility functions----*/
  double calculateEllipseDistance(double x, double y, double normalized_x_radius)
  {
    return std::hypot(x / normalized_x_radius, y);
  }

  double findValue(const std::vector<double>& radius_values, const std::vector<double>& target_values, double dist) {
      for (size_t i = 0; i < radius_values.size(); i++) {
          if (dist < radius_values[i]) {
              return target_values[i];
          }
      }
      return 0.0;
  }

  double generate_AR1_noise(
    double prev_noise, double mean, double std, double phi, std::default_random_engine & gen)
  {
    double noise_std = std * std::sqrt(1 - phi * phi);
    std::normal_distribution<double> dist(0, noise_std);
    double noise = dist(gen);

    return mean + phi * (prev_noise - mean) + noise;
  }

  /*----temp config for perception noise----*/
  // old statistics from ellipse-based modeling
  const std::vector<double> ellipse_y_radius_values = {10, 20, 40, 60, 80, 120, 150, 180, 1000};
  const double ellipse_normalized_x_radius_for_unmask_rate = 0.6;
  const double ellipse_normalized_x_radius_for_distance_mean = 1.8;
  const double ellipse_normalized_x_radius_for_distance_std = 1.8;
  const double ellipse_normalized_x_radius_for_yaw_mean = 0.6;
  const double ellipse_normalized_x_radius_for_yaw_std = 1.6;
  const std::vector<double> unmask_rate_values = {0.92, 0.77, 0.74, 0.66, 0.57, 0.28, 0.09, 0.03, 0.00};
  const std::vector<double> distance_mean_values = {0.25, 0.27, 0.44, 0.67, 1.00,
                                                    3.00, 4.09, 3.40, 0.00};
  const std::vector<double> distance_std_values = {0.35, 0.54, 0.83, 1.14, 1.60,
                                                   3.56, 4.31, 3.61, 0.00};
  const std::vector<double> yaw_mean_values = {0.01, 0.01, 0.00, 0.03, 0.04,
                                               0.00, 0.01, 0.00, 0.00};
  // yaw noise (handly tuned)
  const std::vector<double> yaw_std_values = {0.05, 0.1, 0.15, 0.15, 0.2, 0.2, 0.3, 0.4, 0.5};
  // yaw flip (handly tuned)
  static constexpr auto yaw_flip_velocity_threshold = 0.1;
  const double yaw_flip_rate_for_stop_objects = 0.3;

  // We use AR(1) model for distance and yaw noise, and a keep rate for true positive noise and yaw flip noise.
  // To model noises' autocorrelation coefficients, we define the tau (handly tuned) as the following.
  static constexpr double correlation_for_delta_t = 0.5; // autocorrelation coefficient=0.5 for an interval of delta_t.

  const double delta_t_for_distance = 0.5; //sec
  const double delta_t_for_yaw = 0.3; //sec
  const double delta_t_for_yaw_flip = 0.1; //sec
  const double delta_t_for_unmask_rate = 0.3; //sec

  const double tau_for_distance = -delta_t_for_distance / std::log(correlation_for_delta_t);
  const double tau_for_yaw = -delta_t_for_yaw / std::log(correlation_for_delta_t);
  const double tau_for_unmask_rate = -delta_t_for_unmask_rate / std::log(correlation_for_delta_t);
  const double tau_for_yaw_flip = -delta_t_for_yaw_flip / std::log(correlation_for_delta_t);
  // phi for AR(1)-nased noise or keep_rate for bool noise can be calculated by `exp(-time_interval / tau)`.
  /*---------------------------------------------------*/

public:
  explicit DetectionSensor(
    const double current_simulation_time,
    const simulation_api_schema::DetectionSensorConfiguration & configuration,
    const typename rclcpp::Publisher<T>::SharedPtr & publisher,
    const typename rclcpp::Publisher<U>::SharedPtr & ground_truth_publisher = nullptr)
  : DetectionSensorBase(current_simulation_time, configuration),
    detected_objects_publisher(publisher),
    ground_truth_objects_publisher(ground_truth_publisher),
    random_engine_(configuration.random_seed())
  {
  }

  ~DetectionSensor() override = default;

  auto update(
    const double, const std::vector<traffic_simulator_msgs::EntityStatus> &, const rclcpp::Time &,
    const std::vector<std::string> & lidar_detected_entities) -> void override;
};
}  // namespace simple_sensor_simulator

#endif  // SIMPLE_SENSOR_SIMULATOR__SENSOR_SIMULATION__DETECTION_SENSOR__DETECTION_SENSOR_HPP_
