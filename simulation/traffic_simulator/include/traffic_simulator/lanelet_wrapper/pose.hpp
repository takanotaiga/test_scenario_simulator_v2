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

#ifndef TRAFFIC_SIMULATOR__LANELET_WRAPPER_POSE_HPP_
#define TRAFFIC_SIMULATOR__LANELET_WRAPPER_POSE_HPP_

#include <lanelet2_matching/LaneletMatching.h>

#include <traffic_simulator/lanelet_wrapper/lanelet_wrapper.hpp>

namespace traffic_simulator
{
namespace lanelet_wrapper
{
namespace pose
{
/// @note This value was determined experimentally by @hakuturu583 and not theoretically.
/// @sa https://github.com/tier4/scenario_simulator_v2/commit/4c8e9f496b061b00bec799159d59c33f2ba46b3a
constexpr static double DEFAULT_MATCH_TO_LANE_REDUCTION_RATIO = 0.8;

auto toMapPose(const LaneletPose & lanelet_pose, const bool fill_pitch = true) -> PoseStamped;

auto toLaneletPose(
  const Pose & map_pose, const lanelet::Id lanelet_id, const double matching_distance = 1.0)
  -> std::optional<LaneletPose>;

auto toLaneletPose(
  const Pose & map_pose, const lanelet::Ids & lanelet_ids, const double matching_distance = 1.0)
  -> std::optional<LaneletPose>;

auto toLaneletPose(
  const Pose & map_pose, const bool include_crosswalk, const double matching_distance = 1.0)
  -> std::optional<LaneletPose>;

auto toLaneletPose(
  const Pose & map_pose, const BoundingBox & bounding_box, const bool include_crosswalk,
  const double matching_distance = 1.0,
  const RoutingGraphType type = RoutingConfiguration().routing_graph_type)
  -> std::optional<LaneletPose>;

auto toLaneletPoses(
  const Pose & map_pose, const lanelet::Id lanelet_id, const double matching_distance = 5.0,
  const bool include_opposite_direction = true,
  const RoutingGraphType type = RoutingConfiguration().routing_graph_type)
  -> std::vector<LaneletPose>;

auto alternativeLaneletPoses(const LaneletPose & reference_lanelet_pose)
  -> std::vector<LaneletPose>;

auto alongLaneletPose(
  const LaneletPose & from_pose, const lanelet::Ids & route_lanelets, const double distance)
  -> LaneletPose;

auto alongLaneletPose(
  const LaneletPose & from_pose, const double distance,
  const RoutingGraphType type = RoutingConfiguration().routing_graph_type) -> LaneletPose;

auto canonicalizeLaneletPose(const LaneletPose & lanelet_pose)
  -> std::tuple<std::optional<LaneletPose>, std::optional<lanelet::Id>>;

auto canonicalizeLaneletPose(const LaneletPose & lanelet_pose, const lanelet::Ids & route_lanelets)
  -> std::tuple<std::optional<LaneletPose>, std::optional<lanelet::Id>>;

// used only by this namespace
auto matchToLane(
  const Pose & map_pose, const BoundingBox & bounding_box, const bool include_crosswalk,
  const double matching_distance = 1.0,
  const double reduction_ratio = DEFAULT_MATCH_TO_LANE_REDUCTION_RATIO,
  const RoutingGraphType type = RoutingConfiguration().routing_graph_type)
  -> std::optional<lanelet::Id>;

auto leftLaneletIds(
  const lanelet::Id lanelet_id, const RoutingGraphType type, const bool include_opposite_direction)
  -> lanelet::Ids;

auto rightLaneletIds(
  const lanelet::Id lanelet_id, const RoutingGraphType type, const bool include_opposite_direction)
  -> lanelet::Ids;
}  // namespace pose
}  // namespace lanelet_wrapper
}  // namespace traffic_simulator
#endif  // TRAFFIC_SIMULATOR__LANELET_WRAPPER_POSE_HPP_
