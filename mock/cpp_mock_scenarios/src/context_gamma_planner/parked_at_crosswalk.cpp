// Copyright 2022 TIER IV, Inc. All rights reserved.
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

#include <quaternion_operation/quaternion_operation.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cpp_mock_scenarios/catalogs.hpp>
#include <cpp_mock_scenarios/cpp_scenario_node.hpp>
#include <rclcpp/rclcpp.hpp>
#include <traffic_simulator/api/api.hpp>

// headers in STL
#include <memory>
#include <string>
#include <vector>

class ParkedAtCrosswalkScenario : public cpp_mock_scenarios::CppScenarioNode
{
public:
  explicit ParkedAtCrosswalkScenario(const rclcpp::NodeOptions & option)
  : cpp_mock_scenarios::CppScenarioNode(
      "parked_at_crosswalk",
      ament_index_cpp::get_package_share_directory("kashiwanoha_map") + "/map", "lanelet2_map.osm",
      __FILE__, false, option)
  {
    start();
  }

private:
  void onUpdate() override
  {
    const auto t = api_.getCurrentTime();
    // LCOV_EXCL_START

    if (api_.getEntityStatus("bob").getMapPose().position.y > 73744.0) {
      stop(cpp_mock_scenarios::Result::SUCCESS);
    }
    // LCOV_EXCL_STOP
    if (t >= 60) {
      stop(cpp_mock_scenarios::Result::FAILURE);
    }
  }

  void onInitialize() override
  {
    api_.spawn(
      "ego",
      traffic_simulator::helper::constructCanonicalizedLaneletPose(
        34981, 0, 0, api_.getHdmapUtils()),
      getVehicleParameters());
    api_.setEntityStatus(
      "ego",
      traffic_simulator::helper::constructCanonicalizedLaneletPose(
        34981, 0, 0, api_.getHdmapUtils()),
      traffic_simulator::helper::constructActionStatus(5.7));
    api_.requestSpeedChange("ego", 0, true);
    api_.requestAssignRoute(
      "ego", std::vector<traffic_simulator::CanonicalizedLaneletPose>{
               traffic_simulator::helper::constructCanonicalizedLaneletPose(
                 34675, 0.0, 0.0, api_.getHdmapUtils()),
               traffic_simulator::helper::constructCanonicalizedLaneletPose(
                 34690, 0.0, 0.0, api_.getHdmapUtils()),
             });

    api_.spawn(
      "bob",
      traffic_simulator::helper::constructCanonicalizedLaneletPose(
        34378, 0.0, 0, api_.getHdmapUtils()),
      getPedestrianParameters(), traffic_simulator::PedestrianBehavior::contextGamma());
    api_.requestSpeedChange(
      "bob", 0.5, traffic_simulator::speed_change::Transition::LINEAR,
      traffic_simulator::speed_change::Constraint(
        traffic_simulator::speed_change::Constraint::Type::LONGITUDINAL_ACCELERATION, 1.0),
      true);
    const geometry_msgs::msg::Pose goal_pose = traffic_simulator::pose::toMapPose(
      traffic_simulator::helper::constructCanonicalizedLaneletPose(
        34378, 7.5, 0, 0, 0, 0, api_.getHdmapUtils()));
    api_.requestAcquirePosition("bob", goal_pose);
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto component = std::make_shared<ParkedAtCrosswalkScenario>(options);
  rclcpp::spin(component);
  rclcpp::shutdown();
  return 0;
}
