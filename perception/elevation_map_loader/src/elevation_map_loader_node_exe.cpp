// Copyright 2021 Tier IV, Inc. All rights reserved.
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

#include "elevation_map_loader/elevation_map_loader_node.hpp"

#include <chrono>

#include <grid_map_core/GridMap.hpp>
#include <grid_map_cv/InpaintFilter.hpp>
#include <grid_map_pcl/GridMapPclLoader.hpp>
#include <grid_map_pcl/helpers.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <rclcpp/logger.hpp>

#include <grid_map_msgs/msg/grid_map.hpp>

#include <boost/geometry/algorithms/convex_hull.hpp>
#include <boost/geometry/algorithms/intersects.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include <lanelet2_core/geometry/Polygon.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.h>

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define EIGEN_MPL2_ONLY
#include <Eigen/Core>
#include <Eigen/Geometry>

int main(int argc, char ** argv)
{
  using namespace std::chrono_literals;

  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;

  auto node = std::make_shared<ElevationMapLoaderNode>(node_options);
  rclcpp::sleep_for(20s);
  rclcpp::spin_some(node);
  // rclcpp::spin(node);
  rclcpp::sleep_for(120s);

  rclcpp::shutdown();

  return 0;
}
