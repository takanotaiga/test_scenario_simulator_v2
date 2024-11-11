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

#ifndef TRAFFIC_SIMULATOR__UTILS__DISTANCE_HPP_
#define TRAFFIC_SIMULATOR__UTILS__DISTANCE_HPP_

#include <geometry/spline/catmull_rom_spline_interface.hpp>
#include <traffic_simulator/data_type/entity_status.hpp>
#include <traffic_simulator/data_type/lanelet_pose.hpp>
#include <traffic_simulator/lanelet_wrapper/distance.hpp>
#include <traffic_simulator_msgs/msg/bounding_box.hpp>
#include <traffic_simulator_msgs/msg/waypoints_array.hpp>

namespace traffic_simulator
{
inline namespace distance
{
using Pose = geometry_msgs::msg::Pose;
using BoundingBox = traffic_simulator_msgs::msg::BoundingBox;
using Spline = math::geometry::CatmullRomSpline;
using Waypoints = traffic_simulator_msgs::msg::WaypointsArray;
using SplineInterface = math::geometry::CatmullRomSplineInterface;

// Lateral
auto lateralDistance(
  const CanonicalizedLaneletPose & from, const CanonicalizedLaneletPose & to,
  const bool allow_lane_change) -> std::optional<double>;

auto lateralDistance(
  const CanonicalizedLaneletPose & from, const CanonicalizedLaneletPose & to,
  const double matching_distance, const bool allow_lane_change) -> std::optional<double>;

// Lateral (unit: lanes)
auto countLaneChanges(
  const CanonicalizedLaneletPose & from, const CanonicalizedLaneletPose & to,
  bool allow_lane_change, const std::shared_ptr<hdmap_utils::HdMapUtils> & hdmap_utils_ptr)
  -> std::optional<std::pair<int, int>>;

// Longitudinal
auto longitudinalDistance(
  const CanonicalizedLaneletPose & from, const CanonicalizedLaneletPose & to,
  const bool include_adjacent_lanelet, const bool include_opposite_direction,
  const bool allow_lane_change) -> std::optional<double>;

// BoundingBox
auto boundingBoxDistance(
  const Pose & from, const BoundingBox & from_bounding_box, const Pose & to,
  const BoundingBox & to_bounding_box) -> std::optional<double>;

auto boundingBoxLaneLateralDistance(
  const CanonicalizedLaneletPose & from, const BoundingBox & from_bounding_box,
  const CanonicalizedLaneletPose & to, const BoundingBox & to_bounding_box,
  const bool allow_lane_change) -> std::optional<double>;

auto boundingBoxLaneLongitudinalDistance(
  const CanonicalizedLaneletPose & from, const BoundingBox & from_bounding_box,
  const CanonicalizedLaneletPose & to, const BoundingBox & to_bounding_box,
  const bool include_adjacent_lanelet, const bool include_opposite_direction,
  const bool allow_lane_change) -> std::optional<double>;

auto splineDistanceToBoundingBox(
  const SplineInterface & spline, const CanonicalizedLaneletPose & pose,
  const BoundingBox & bounding_box, double width_extension_right = 0.0,
  double width_extension_left = 0.0, double length_extension_front = 0.0,
  double length_extension_rear = 0.0) -> std::optional<double>;

// Bounds
auto distanceToLaneBound(
  const Pose & map_pose, const BoundingBox & bounding_box, lanelet::Id lanelet_id) -> double;

auto distanceToLaneBound(
  const Pose & map_pose, const BoundingBox & bounding_box, const lanelet::Ids & lanelet_ids)
  -> double;

auto distanceToLeftLaneBound(
  const Pose & map_pose, const BoundingBox & bounding_box, const lanelet::Id lanelet_id) -> double;

auto distanceToLeftLaneBound(
  const Pose & map_pose, const BoundingBox & bounding_box, const lanelet::Ids & lanelet_ids)
  -> double;

auto distanceToRightLaneBound(
  const Pose & map_pose, const BoundingBox & bounding_box, const lanelet::Id lanelet_id) -> double;

auto distanceToRightLaneBound(
  const Pose & map_pose, const BoundingBox & bounding_box, const lanelet::Ids & lanelet_ids)
  -> double;

// Other objects
template <typename... Ts>
auto distanceToStopLine(Ts &&... xs)
{
  return lanelet_wrapper::distance::distanceToStopLine(std::forward<decltype(xs)>(xs)...);
}

template <typename... Ts>
auto distanceToTrafficLightStopLine(Ts &&... xs)
{
  return lanelet_wrapper::distance::distanceToTrafficLightStopLine(
    std::forward<decltype(xs)>(xs)...);
}

template <typename... Ts>
auto distanceToCrosswalk(Ts &&... xs)
{
  return lanelet_wrapper::distance::distanceToCrosswalk(std::forward<decltype(xs)>(xs)...);
}

auto distanceToYieldStop(
  const CanonicalizedLaneletPose & reference_pose, const lanelet::Ids & following_lanelets,
  const std::vector<CanonicalizedLaneletPose> & other_poses) -> std::optional<double>;

/*
   Here it is required to pass the CanonicalizedEntityStatus vector, instead of just 
   the CanonicalizedLaneletPose vector, since it is necessary to know the BoundingBox of each Entity
*/
auto distanceToNearestConflictingPose(
  const lanelet::Ids & following_lanelets, const SplineInterface & spline,
  const std::vector<CanonicalizedEntityStatus> & other_statuses) -> std::optional<double>;
}  // namespace distance
}  // namespace traffic_simulator
#endif  // TRAFFIC_SIMULATOR__UTILS__DISTANCE_HPP_
