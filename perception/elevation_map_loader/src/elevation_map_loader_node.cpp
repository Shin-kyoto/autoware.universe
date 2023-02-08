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

#include <grid_map_core/GridMap.hpp>
#include <grid_map_cv/InpaintFilter.hpp>
#include <grid_map_pcl/GridMapPclLoader.hpp>
#include <grid_map_pcl/helpers.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <grid_map_utils/polygon_iterator.hpp>
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
#include <rosbag2_storage_default_plugins/sqlite/sqlite_statement_wrapper.hpp>

ElevationMapLoaderNode::ElevationMapLoaderNode(const rclcpp::NodeOptions & options)
: Node("elevation_map_loader", options)
{
  layer_name_ = this->declare_parameter("map_layer_name", std::string("elevation"));
  std::string param_file_path = this->declare_parameter("param_file_path", "path_default");
  map_frame_ = this->declare_parameter("map_frame", "map");
  use_inpaint_ = this->declare_parameter("use_inpaint", true);
  use_morphology_ = this->declare_parameter("use_morphology", false);
  use_incremental_generation_ = this->declare_parameter("use_incremental_generation", true);
  inpaint_radius_ = this->declare_parameter("inpaint_radius", 0.3);
  use_elevation_map_cloud_publisher_ =
    this->declare_parameter("use_elevation_map_cloud_publisher", false);
  elevation_map_directory_ = this->declare_parameter("elevation_map_directory", "path_default");
  elevation_map_directory_original_ =
    this->declare_parameter("elevation_map_directory_original", "path_default");
  const bool use_lane_filter = this->declare_parameter("use_lane_filter", false);
  data_manager_.use_lane_filter_ = use_lane_filter;

  lane_filter_.use_lane_filter_ = use_lane_filter;
  lane_filter_.lane_margin_ = this->declare_parameter("lane_margin", 0.5);
  lane_filter_.lane_height_diff_thresh_ = this->declare_parameter("lane_height_diff_thresh", 1.0);
  lane_filter_.voxel_size_x_ = declare_parameter("lane_filter_voxel_size_x", 0.04);
  lane_filter_.voxel_size_y_ = declare_parameter("lane_filter_voxel_size_y", 0.04);
  lane_filter_.voxel_size_z_ = declare_parameter("lane_filter_voxel_size_z", 0.04);

  auto grid_map_logger = rclcpp::get_logger("grid_map_logger");
  grid_map_logger.set_level(rclcpp::Logger::Level::Error);
  grid_map_pcl_loader_ = pcl::make_shared<grid_map::GridMapPclLoader>(grid_map_logger);
  grid_map_pcl_loader_->loadParameters(param_file_path);

  rclcpp::QoS durable_qos{1};
  durable_qos.transient_local();
  pub_elevation_map_ =
    this->create_publisher<grid_map_msgs::msg::GridMap>("output/elevation_map", durable_qos);

  if (use_elevation_map_cloud_publisher_) {
    pub_elevation_map_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "output/elevation_map_cloud", durable_qos);
  }

  using std::placeholders::_1;
  sub_map_hash_ = create_subscription<tier4_external_api_msgs::msg::MapHash>(
    "/api/autoware/get/map/info/hash", durable_qos,
    std::bind(&ElevationMapLoaderNode::onMapHash, this, _1));
  sub_vector_map_ = this->create_subscription<autoware_auto_mapping_msgs::msg::HADMapBin>(
    "input/vector_map", durable_qos, std::bind(&ElevationMapLoaderNode::onVectorMap, this, _1));
  // bool enable_differential_load = declare_parameter<bool>("enable_differential_load");
  bool enable_differential_load = true;
  if (enable_differential_load) {
    {
      const auto period_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0));
      group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
      pcd_loader_client_ = create_client<autoware_map_msgs::srv::GetDifferentialPointCloudMap>(
        "pcd_loader_service", rmw_qos_profile_services_default, group_);
      while (!pcd_loader_client_->wait_for_service(std::chrono::seconds(1)) && rclcpp::ok()) {
        RCLCPP_INFO(
          this->get_logger(),
          "Waiting for pcd map loader service. Check if the enable_differential_load in "
          "pointcloud_map_loader is set `true`.");
      }
      timer_ = rclcpp::create_timer(
        this, get_clock(), period_ns, std::bind(&ElevationMapLoaderNode::timer_callback, this));
    }

    if (data_manager_.isInitialized()) {
      publish();
    }
  } else {
    sub_pointcloud_map_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "input/pointcloud_map", durable_qos,
      std::bind(&ElevationMapLoaderNode::onPointcloudMap, this, _1));
  }
}

void ElevationMapLoaderNode::publish()
{
  struct stat info;
  if (stat(data_manager_.elevation_map_path_->c_str(), &info) != 0) {
    RCLCPP_INFO(this->get_logger(), "Create elevation map from pointcloud map ");
    create_elevation_map();
    RCLCPP_INFO(
      this->get_logger(), "Finish creating elevation map from pointcloud map. Start inpaint");
    if (use_inpaint_) {
      inpaintElevationMap(inpaint_radius_);
    }
    saveElevationMap();
  } else if (info.st_mode & S_IFDIR) {
    RCLCPP_INFO(
      this->get_logger(), "Load elevation map from: %s",
      data_manager_.elevation_map_path_->c_str());

    // Check if bag can be loaded
    bool is_bag_loaded = false;
    try {
      is_bag_loaded = grid_map::GridMapRosConverter::loadFromBag(
        *data_manager_.elevation_map_path_, "elevation_map", elevation_map_);
    } catch (rosbag2_storage_plugins::SqliteException & e) {
      is_bag_loaded = false;
    }
    if (!is_bag_loaded) {
      // Delete directory including elevation map if bag is broken
      RCLCPP_ERROR(
        this->get_logger(), "Try to loading bag, but bag is broken. Remove %s",
        data_manager_.elevation_map_path_->c_str());
      std::filesystem::remove_all(data_manager_.elevation_map_path_->c_str());
      // Create elevation map from pointcloud map if bag is broken
      RCLCPP_INFO(this->get_logger(), "Create elevation map from pointcloud map ");
      create_elevation_map();
      if (use_inpaint_) {
        inpaintElevationMap(inpaint_radius_);
      }
      saveElevationMap();
    }
  }

  elevation_map_.setFrameId(map_frame_);
  auto msg = grid_map::GridMapRosConverter::toMessage(elevation_map_);
  pub_elevation_map_->publish(std::move(msg));

  if (use_elevation_map_cloud_publisher_) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr elevation_map_cloud_ptr =
      createPointcloudFromElevationMap();
    sensor_msgs::msg::PointCloud2 elevation_map_cloud_msg;
    pcl::toROSMsg(*elevation_map_cloud_ptr, elevation_map_cloud_msg);
    pub_elevation_map_cloud_->publish(elevation_map_cloud_msg);
  }
  finish_pub_elevation_map_ = true;
}

void ElevationMapLoaderNode::timer_callback()
{
  {
    if (use_incremental_generation_) {
      if (!finish_pub_elevation_map_) {
        std::shared_ptr<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> map_pcl_vector =
          std::make_shared<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>>();
        ;
        ElevationMapLoaderNode::receive_map_vector(map_pcl_vector);
        RCLCPP_INFO(this->get_logger(), "receive service with pointcloud_map");
        data_manager_.map_pcl_vector_ptr_ =
          pcl::make_shared<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>>(*map_pcl_vector);
      }
    } else {
      ElevationMapLoaderNode::receive_map();
      RCLCPP_INFO(this->get_logger(), "receive service with pointcloud_map");
    }
    // RCLCPP_INFO(
    //   this->get_logger(), "data_manager_.isInitialized(): %d", data_manager_.isInitialized());
  }
  if (data_manager_.isInitialized() && !finish_pub_elevation_map_) {
    publish();
  }
}

void ElevationMapLoaderNode::onMapHash(
  const tier4_external_api_msgs::msg::MapHash::ConstSharedPtr map_hash)
{
  RCLCPP_INFO(this->get_logger(), "subscribe map_hash");
  const auto elevation_map_hash = map_hash->pcd;
  data_manager_.elevation_map_path_ = std::make_unique<std::filesystem::path>(
    std::filesystem::path(elevation_map_directory_) / elevation_map_hash);
  if (data_manager_.isInitialized()) {
    publish();
  }
}

void ElevationMapLoaderNode::onPointcloudMap(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr pointcloud_map)
{
  RCLCPP_INFO(this->get_logger(), "subscribe pointcloud_map");
  pcl::PointCloud<pcl::PointXYZ> map_pcl;
  pcl::fromROSMsg<pcl::PointXYZ>(*pointcloud_map, map_pcl);
  data_manager_.map_pcl_ptr_ = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>(map_pcl);
  if (data_manager_.isInitialized()) {
    publish();
  }
}

void ElevationMapLoaderNode::onVectorMap(
  const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr vector_map)
{
  RCLCPP_INFO(this->get_logger(), "subscribe vector_map");
  data_manager_.lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
  lanelet::utils::conversion::fromBinMsg(*vector_map, data_manager_.lanelet_map_ptr_);
  const lanelet::ConstLanelets all_lanelets =
    lanelet::utils::query::laneletLayer(data_manager_.lanelet_map_ptr_);
  lane_filter_.road_lanelets_ = lanelet::utils::query::roadLanelets(all_lanelets);
  if (data_manager_.isInitialized()) {
    publish();
  }
}

void ElevationMapLoaderNode::receive_map()
{
  pcl::PointCloud<pcl::PointXYZ> map_pcl;
  sensor_msgs::msg::PointCloud2 pointcloud_map;
  // create a loading request with mode = 1
  auto request = std::make_shared<autoware_map_msgs::srv::GetDifferentialPointCloudMap::Request>();
  // 地図全体を要求する
  request->area.type = autoware_map_msgs::msg::AreaInfo::ALL_AREA;  // 1;
  std::vector<std::string> cached_ids{};
  if (!pcd_loader_client_->service_is_ready()) {
    RCLCPP_INFO(
      this->get_logger(),
      "Waiting for pcd map loader service. Check if the enable_differential_load in "
      "pointcloud_map_loader is set `true`.");
    ;
  }
  bool is_all_received = false;
  while (!is_all_received) {
    request->cached_ids = cached_ids;  //毎回書き換える
    // send a request to map_loader
    RCLCPP_INFO(this->get_logger(), "send a request to map_loader");
    auto result{pcd_loader_client_->async_send_request(
      request,
      [](rclcpp::Client<autoware_map_msgs::srv::GetDifferentialPointCloudMap>::SharedFuture) {})};
    std::future_status status = result.wait_for(std::chrono::seconds(0));
    while (status != std::future_status::ready) {
      RCLCPP_INFO(this->get_logger(), "waiting response");
      if (!rclcpp::ok()) {
        return;
      }
      status = result.wait_for(std::chrono::seconds(1));
    }

    RCLCPP_INFO(this->get_logger(), "concat maps");
    if (result.get()->new_pointcloud_with_ids.empty()) {  // new_pointcloud_with_idsが空
      RCLCPP_INFO(this->get_logger(), "finish receiving");
      is_all_received = true;
    } else {
      // concat maps
      // 点群地図と，serviceとして渡されてきた差分地図をconcat
      for (const auto & new_pointcloud_with_id : result.get()->new_pointcloud_with_ids) {
        if (pointcloud_map.width == 0) {
          pointcloud_map = new_pointcloud_with_id.pointcloud;
        } else {
          pointcloud_map.width += new_pointcloud_with_id.pointcloud.width;
          pointcloud_map.row_step += new_pointcloud_with_id.pointcloud.row_step;
          RCLCPP_INFO(this->get_logger(), "start insert");
          pointcloud_map.data.insert(
            pointcloud_map.data.end(), new_pointcloud_with_id.pointcloud.data.begin(),
            new_pointcloud_with_id.pointcloud.data.end());
          RCLCPP_INFO(this->get_logger(), "end insert");
        }
        RCLCPP_INFO(this->get_logger(), "start push back");
        cached_ids.push_back(new_pointcloud_with_id.cell_id);
        RCLCPP_INFO(this->get_logger(), "end push back");
      }
    }
  }
  RCLCPP_INFO(this->get_logger(), "start pcl::fromROSMsg");
  pcl::fromROSMsg<pcl::PointXYZ>(pointcloud_map, map_pcl);
  data_manager_.map_pcl_ptr_ = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>(map_pcl);
  RCLCPP_INFO(this->get_logger(), "end pcl::fromROSMsg");
}

void ElevationMapLoaderNode::receive_map_vector(
  const std::shared_ptr<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> pointcloud_map_vector)
{
  // create a loading request with mode = 1
  auto request = std::make_shared<autoware_map_msgs::srv::GetDifferentialPointCloudMap::Request>();
  // 地図全体を要求する
  request->area.type = autoware_map_msgs::msg::AreaInfo::ALL_AREA;  // 1;
  std::vector<std::string> cached_ids{};
  bool is_all_received = false;
  while (!is_all_received) {
    request->cached_ids = cached_ids;  //毎回書き換える
    // send a request to map_loader
    RCLCPP_INFO(this->get_logger(), "send a request to map_loader");
    auto result{pcd_loader_client_->async_send_request(
      request,
      [](rclcpp::Client<autoware_map_msgs::srv::GetDifferentialPointCloudMap>::SharedFuture) {})};
    std::future_status status = result.wait_for(std::chrono::seconds(0));
    while (status != std::future_status::ready) {
      RCLCPP_INFO(this->get_logger(), "waiting response");
      if (!rclcpp::ok()) {
        return;
      }
      status = result.wait_for(std::chrono::seconds(1));
    }

    if (result.get()->new_pointcloud_with_ids.empty()) {  // new_pointcloud_with_idsが空
      is_all_received = true;
    } else {
      // vectorにまとめる
      for (const auto & new_pointcloud_with_id : result.get()->new_pointcloud_with_ids) {
        pcl::PointCloud<pcl::PointXYZ> map_pcl;
        RCLCPP_INFO(this->get_logger(), "start pcl::fromROSMsg");
        pcl::fromROSMsg<pcl::PointXYZ>(new_pointcloud_with_id.pointcloud, map_pcl);
        RCLCPP_INFO(this->get_logger(), "end pcl::fromROSMsg");
        const pcl::PointCloud<pcl::PointXYZ>::Ptr map_pcl_ptr =
          pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>(map_pcl);
        RCLCPP_INFO(
          this->get_logger(), "end pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>(map_pcl);");
        pointcloud_map_vector->push_back(map_pcl_ptr);
        RCLCPP_INFO(
          this->get_logger(), "start cached_ids.push_back(new_pointcloud_with_id.cell_id);");
        cached_ids.push_back(new_pointcloud_with_id.cell_id);
      }
    }
  }
}

void ElevationMapLoaderNode::create_elevation_map()
{
  if (use_incremental_generation_) {
    {
      // length取得
      double max_bound_x;
      double max_bound_y;
      double min_bound_x;
      double min_bound_y;
      std::tie(max_bound_x, max_bound_y, min_bound_x, min_bound_y) = get_bound();

      std::vector<grid_map::GridMap> grid_map_vector;
      RCLCPP_INFO(
        this->get_logger(), "*data_manager_.map_pcl_vector_ptr_.size() %d",
        static_cast<int>(data_manager_.map_pcl_vector_ptr_->size()));
      for (const auto & map_pcl : *data_manager_.map_pcl_vector_ptr_) {
        // RCLCPP_INFO(this->get_logger(), "pointcloud to grid_map");
        grid_map_vector.push_back(createElevationMap_incremental(map_pcl));
      }
      RCLCPP_INFO(this->get_logger(), "get length and center position");
      grid_map::Length length =
        grid_map::Length(max_bound_x - min_bound_x, max_bound_y - min_bound_y);
      double resolution = 0.3;  // elevation_map_.getResolution();  // node paramから取るべき
      grid_map::Position position =
        grid_map::Position((max_bound_x + min_bound_x) / 2.0, (max_bound_y + min_bound_y) / 2.0);

      elevation_map_.clearAll();
      RCLCPP_INFO(this->get_logger(), "set elevation_map's geometry");
      elevation_map_.setGeometry(length, resolution, position);
      RCLCPP_INFO(this->get_logger(), "Grid map dimensions: %f", elevation_map_.getLength()(0));
      RCLCPP_INFO(this->get_logger(), "Grid map dimensions: %f", elevation_map_.getLength()(1));
      RCLCPP_INFO(this->get_logger(), "Grid map size: %d", elevation_map_.getSize()(0));
      RCLCPP_INFO(this->get_logger(), "Grid map size: %d", elevation_map_.getSize()(1));
      RCLCPP_INFO(this->get_logger(), "Grid map resolution: %f", elevation_map_.getResolution());
      RCLCPP_INFO(this->get_logger(), "Initialized map geometry");
      RCLCPP_INFO(this->get_logger(), "add layer to elevation map");
      double value = 0.0;
      elevation_map_.add(layer_name_, value);
      RCLCPP_INFO(this->get_logger(), "finish adding layer to elevation map");

      for (const auto & grid_map : grid_map_vector) {
        grid_map::Matrix gridMapData = grid_map.get("elevation_");
        unsigned int linearGridMapSize = grid_map.getSize().prod();
        for (unsigned int linearIndex = 0; linearIndex < linearGridMapSize; ++linearIndex) {
          // RCLCPP_INFO(this->get_logger(), "move each grid_map to all grid_map");
          const grid_map::Index index(
            grid_map::getIndexFromLinearIndex(linearIndex, grid_map.getSize()));
          // 各grid_mapのセルのデータを，elevation_map全体における，セルに格納していく
          // RCLCPP_INFO(
          //   this->get_logger(), "start move cell value: index(0) %d, index(1) %d", index(0),
          //   index(1));
          grid_map::Position position;
          grid_map.getPosition(index, position);
          grid_map::Index index_all;
          elevation_map_.getIndex(position, index_all);
          elevation_map_.get(layer_name_)(index_all(0), index_all(1)) =
            (gridMapData)(index(0), index(1));
        }
      }
    }
    RCLCPP_INFO(this->get_logger(), "finish incremental generation");
  } else {
    createElevationMap();
  }
}

std::tuple<double, double, double, double> ElevationMapLoaderNode::get_bound()
{
  bool bound_flag = false;  // TODO(Shin-kyoto): bound_flagは暫定対応．boundがdoubleかどうかも確認
  double all_max_bound_x = 0.0;
  double all_max_bound_y = 0.0;
  double all_min_bound_x = 0.0;
  double all_min_bound_y = 0.0;
  for (const auto & map_pcl : *data_manager_.map_pcl_vector_ptr_) {
    // lengthを取っていく
    pcl::PointXYZ minBound;
    pcl::PointXYZ maxBound;
    pcl::getMinMax3D(*map_pcl, minBound, maxBound);
    if (!bound_flag) {
      all_max_bound_x = maxBound.x;
      all_max_bound_y = maxBound.y;
      all_min_bound_x = minBound.x;
      all_min_bound_y = minBound.y;
      bound_flag = true;
    } else {
      if (all_max_bound_x < maxBound.x) {
        all_max_bound_x = maxBound.x;
      }
      if (all_max_bound_y < maxBound.y) {
        all_max_bound_y = maxBound.y;
      }
      if (all_min_bound_x > minBound.x) {
        all_min_bound_x = minBound.x;
      }
      if (all_min_bound_y > minBound.y) {
        all_min_bound_y = minBound.y;
      }
    }
  }
  return std::forward_as_tuple(all_max_bound_x, all_max_bound_y, all_min_bound_x, all_min_bound_y);
}

void ElevationMapLoaderNode::createElevationMap()
{
  if (lane_filter_.use_lane_filter_) {
    const auto convex_hull = getConvexHull(data_manager_.map_pcl_ptr_);
    lanelet::ConstLanelets intersected_lanelets =
      getIntersectedLanelets(convex_hull, lane_filter_.road_lanelets_);
    pcl::PointCloud<pcl::PointXYZ>::Ptr lane_filtered_map_pcl_ptr =
      getLaneFilteredPointCloud(intersected_lanelets, data_manager_.map_pcl_ptr_);
    grid_map_pcl_loader_->setInputCloud(lane_filtered_map_pcl_ptr);
  } else {
    grid_map_pcl_loader_->setInputCloud(data_manager_.map_pcl_ptr_);
  }
  createElevationMapFromPointcloud();
  elevation_map_ = grid_map_pcl_loader_->getGridMap();
}

grid_map::GridMap ElevationMapLoaderNode::createElevationMap_incremental(
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_pcl)
{
  if (lane_filter_.use_lane_filter_) {
    const auto convex_hull = getConvexHull(map_pcl);
    lanelet::ConstLanelets intersected_lanelets =
      getIntersectedLanelets(convex_hull, lane_filter_.road_lanelets_);
    pcl::PointCloud<pcl::PointXYZ>::Ptr lane_filtered_map_pcl_ptr =
      getLaneFilteredPointCloud(intersected_lanelets, map_pcl);
    grid_map_pcl_loader_->setInputCloud(lane_filtered_map_pcl_ptr);
  } else {
    grid_map_pcl_loader_->setInputCloud(map_pcl);
  }
  createElevationMapFromPointcloud();
  return grid_map_pcl_loader_->getGridMap();
}

void ElevationMapLoaderNode::createElevationMapFromPointcloud()
{
  const auto start = std::chrono::high_resolution_clock::now();
  grid_map_pcl_loader_->preProcessInputCloud();
  grid_map_pcl_loader_->initializeGridMapGeometryFromInputCloud();
  grid_map_pcl_loader_->addLayerFromInputCloud("elevation_");
  grid_map::grid_map_pcl::printTimeElapsedToRosInfoStream(
    start, "Finish creating elevation map. Total time: ", this->get_logger());
}

void ElevationMapLoaderNode::inpaintElevationMap(const float radius)
{
  // Convert elevation layer to OpenCV image to fill in holes.
  // Get the inpaint mask (nonzero pixels indicate where values need to be filled in).
  elevation_map_.add("inpaint_mask", 0.0);

  // elevation_map_.setBasicLayers(std::vector<std::string>());
  // for (grid_map::GridMapIterator iterator(elevation_map_); !iterator.isPastEnd(); ++iterator) {
  //   if (!elevation_map_.isValid(*iterator, layer_name_)) {
  //     elevation_map_.at("inpaint_mask", *iterator) = 1.0;
  //   }
  // }

  // grid_map::Polygon lanelet_polygon;
  // lanelet_polygon.setFrameId(map_frame_);
  // for (const auto & lanelet : lane_filter_.road_lanelets_) {
  //   for (const auto & point : lanelet.polygon2d().basicPolygon()) {
  //     lanelet_polygon.addVertex(grid_map::Position(point.x(), point.y()));
  //   }
  // }

  // for (grid_map_utils::PolygonIterator iterator(elevation_map_, lanelet_polygon);
  //      !iterator.isPastEnd(); ++iterator) {
  double upper_limit = elevation_map_.getSize()(1) / 2;
  for (grid_map::GridMapIterator iterator(elevation_map_); !iterator.isPastEnd(); ++iterator) {
    if ((*iterator)(1) < upper_limit) {
      continue;
    }
    if (!elevation_map_.isValid(*iterator, layer_name_)) {
      elevation_map_.at("inpaint_mask", *iterator) = 1.0;
      // grid_map::Position position;
      // elevation_map_.getPosition(*iterator, position);
      // if (checkPointWithinLanelets(
      //       pcl::PointXYZ(position.x(), position.y(), 0.0), lane_filter_.road_lanelets_)) {
      //   RCLCPP_INFO(this->get_logger(), "iterator 0: %d", (*iterator)(0));
      //   RCLCPP_INFO(this->get_logger(), "iterator 1: %d", (*iterator)(1));
      //   elevation_map_.at("inpaint_mask", *iterator) = 1.0;
      // }
    }
  }

  cv::Mat original_image;
  cv::Mat mask;
  cv::Mat filled_image;
  const float min_value = elevation_map_.get(layer_name_).minCoeffOfFinites();
  const float max_value = elevation_map_.get(layer_name_).maxCoeffOfFinites();

  grid_map::GridMapCvConverter::toImage<unsigned char, 3>(
    elevation_map_, layer_name_, CV_8UC3, min_value, max_value, original_image);
  grid_map::GridMapCvConverter::toImage<unsigned char, 1>(
    elevation_map_, "inpaint_mask", CV_8UC1, mask);
  cv::imwrite("original_image.jpg", original_image);
  cv::imwrite("mask.jpg", mask);

  const auto start_inpaint = std::chrono::high_resolution_clock::now();
  if (use_morphology_) {
    const float radius_in_pixels = 2 * radius / elevation_map_.getResolution();
    cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(2 * radius_in_pixels + 1, 2 * radius_in_pixels + 1),
      cv::Point(radius_in_pixels, radius_in_pixels));
    RCLCPP_INFO_STREAM(this->get_logger(), "radius_in_pixels: %f" << radius_in_pixels);
    RCLCPP_INFO_STREAM(this->get_logger(), "start cv::morphologyEx");
    cv::morphologyEx(original_image, filled_image, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
    RCLCPP_INFO_STREAM(this->get_logger(), "finish cv::morphologyEx");
  } else {
    const float radius_in_pixels = radius / elevation_map_.getResolution();
    RCLCPP_INFO_STREAM(this->get_logger(), "radius_in_pixels: %f" << radius_in_pixels);
    RCLCPP_INFO_STREAM(this->get_logger(), "start cv::inpaint");
    cv::inpaint(original_image, mask, filled_image, radius_in_pixels, cv::INPAINT_NS);
    RCLCPP_INFO_STREAM(this->get_logger(), "finish cv::inpaint");
  }
  cv::imwrite("filled_image.jpg", filled_image);

  {
    const auto stop_inpaint = std::chrono::high_resolution_clock::now();
    const auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop_inpaint - start_inpaint).count() /
      1000.0;
    RCLCPP_INFO_STREAM(this->get_logger(), "inpaint: " << duration << " sec");
  }

  grid_map::GridMapCvConverter::addLayerFromImage<unsigned char, 3>(
    filled_image, layer_name_, elevation_map_, min_value, max_value);
  elevation_map_.erase("inpaint_mask");

  // compare maps
  // load map
  grid_map::GridMap elevation_map_original;
  grid_map::GridMapRosConverter::loadFromBag(
    elevation_map_directory_original_, "elevation_map", elevation_map_original);

  // for (grid_map_utils::PolygonIterator iterator(elevation_map_, lanelet_polygon);
  //      !iterator.isPastEnd(); ++iterator) {
  for (grid_map::GridMapIterator iterator(elevation_map_); !iterator.isPastEnd(); ++iterator) {
    grid_map::Position position;
    elevation_map_.getPosition(*iterator, position);
    if (checkPointWithinLanelets(
          pcl::PointXYZ(position.x(), position.y(), 0.0), lane_filter_.road_lanelets_)) {
      // RCLCPP_INFO(this->get_logger(), "iterator 0: %d", (*iterator)(0));
      // RCLCPP_INFO(this->get_logger(), "iterator 1: %d", (*iterator)(1));
      if (
        fabs(
          elevation_map_.at("elevation", *iterator) -
          elevation_map_original.at("elevation", *iterator)) > 0.1) {
        RCLCPP_INFO(this->get_logger(), "not equal");
        RCLCPP_INFO(this->get_logger(), "iterator 0: %d", (*iterator)(0));
        RCLCPP_INFO(this->get_logger(), "iterator 1: %d", (*iterator)(1));
      }
    }
  }

  for (grid_map::GridMapIterator iterator(elevation_map_); !iterator.isPastEnd(); ++iterator) {
    elevation_map_.at("elevation", *iterator);
    // if (!elevation_map_.isValid(*iterator, layer_name_)) {
    //   elevation_map_.at("inpaint_mask", *iterator) = 1.0;
    // }
  }
}

tier4_autoware_utils::LinearRing2d ElevationMapLoaderNode::getConvexHull(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & input_cloud)
{
  // downsample pointcloud to reduce convex hull calculation cost
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  downsampled_cloud->points.reserve(input_cloud->points.size());
  pcl::VoxelGrid<pcl::PointXYZ> filter;
  filter.setInputCloud(input_cloud);
  filter.setLeafSize(0.5, 0.5, 100.0);
  filter.filter(*downsampled_cloud);

  tier4_autoware_utils::MultiPoint2d candidate_points;
  for (const auto & p : downsampled_cloud->points) {
    candidate_points.emplace_back(p.x, p.y);
  }

  tier4_autoware_utils::LinearRing2d convex_hull;
  boost::geometry::convex_hull(candidate_points, convex_hull);

  return convex_hull;
}

lanelet::ConstLanelets ElevationMapLoaderNode::getIntersectedLanelets(
  const tier4_autoware_utils::LinearRing2d & convex_hull,
  const lanelet::ConstLanelets & road_lanelets)
{
  lanelet::ConstLanelets intersected_lanelets;
  for (const auto & road_lanelet : road_lanelets) {
    if (boost::geometry::intersects(convex_hull, road_lanelet.polygon2d().basicPolygon())) {
      intersected_lanelets.push_back(road_lanelet);
    }
  }
  return intersected_lanelets;
}

bool ElevationMapLoaderNode::checkPointWithinLanelets(
  const pcl::PointXYZ & point, const lanelet::ConstLanelets & intersected_lanelets)
{
  tier4_autoware_utils::Point2d point2d(point.x, point.y);
  for (const auto & lanelet : intersected_lanelets) {
    if (lane_filter_.lane_margin_ > 0) {
      if (
        boost::geometry::distance(point2d, lanelet.polygon2d().basicPolygon()) >
        lane_filter_.lane_margin_) {
        continue;
      }
    } else {
      if (!boost::geometry::within(point2d, lanelet.polygon2d().basicPolygon())) {
        continue;
      }
    }

    if (lane_filter_.lane_height_diff_thresh_ > 0) {
      float distance = calculateDistancePointFromPlane(point, lanelet);
      if (distance < lane_filter_.lane_height_diff_thresh_) {
        return true;
      }
    } else {
      return true;
    }
  }
  return false;
}

float ElevationMapLoaderNode::calculateDistancePointFromPlane(
  const pcl::PointXYZ & point, const lanelet::ConstLanelet & lanelet)
{
  const Eigen::Vector3d point_3d(point.x, point.y, point.z);
  const Eigen::Vector2d point_2d(point.x, point.y);

  const float distance_3d = boost::geometry::distance(point_3d, lanelet.centerline3d());
  const float distance_2d = boost::geometry::distance(point_2d, lanelet.centerline2d());
  const float distance = std::sqrt(distance_3d * distance_3d - distance_2d * distance_2d);

  return distance;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ElevationMapLoaderNode::getLaneFilteredPointCloud(
  const lanelet::ConstLanelets & intersected_lanelets,
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud)
{
  pcl::PointCloud<pcl::PointXYZ> filtered_cloud;
  filtered_cloud.header = cloud->header;

  pcl::PointCloud<pcl::PointXYZ>::Ptr centralized_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  centralized_cloud->reserve(cloud->size());

  // The coordinates of the point cloud are too large, resulting in calculation errors,
  // so offset them to the center.
  // https://github.com/PointCloudLibrary/pcl/issues/4895
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cloud, centroid);
  for (const auto & p : cloud->points) {
    centralized_cloud->points.push_back(
      pcl::PointXYZ(p.x - centroid[0], p.y - centroid[1], p.z - centroid[2]));
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
  voxel_grid.setLeafSize(lane_filter_.voxel_size_x_, lane_filter_.voxel_size_y_, 100000.0);
  voxel_grid.setInputCloud(centralized_cloud);
  voxel_grid.setSaveLeafLayout(true);
  voxel_grid.filter(*downsampled_cloud);

  std::unordered_map<size_t, pcl::PointCloud<pcl::PointXYZ>> downsampled2original_map;
  for (const auto & p : centralized_cloud->points) {
    if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)) {
      continue;
    }
    const size_t index = voxel_grid.getCentroidIndex(p);
    downsampled2original_map[index].points.push_back(p);
  }

  for (auto & point : downsampled_cloud->points) {
    if (checkPointWithinLanelets(
          pcl::PointXYZ(point.x + centroid[0], point.y + centroid[1], point.z + centroid[2]),
          intersected_lanelets)) {
      const size_t index = voxel_grid.getCentroidIndex(point);
      for (auto & original_point : downsampled2original_map[index].points) {
        original_point.x += centroid[0];
        original_point.y += centroid[1];
        original_point.z += centroid[2];
        filtered_cloud.points.push_back(original_point);
      }
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_ptr;
  filtered_cloud_ptr = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>(filtered_cloud);
  return filtered_cloud_ptr;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ElevationMapLoaderNode::createPointcloudFromElevationMap()
{
  pcl::PointCloud<pcl::PointXYZ> output_cloud;
  output_cloud.header.frame_id = elevation_map_.getFrameId();

  for (grid_map::GridMapIterator iterator(elevation_map_); !iterator.isPastEnd(); ++iterator) {
    float z = elevation_map_.at(layer_name_, *iterator);
    if (!std::isnan(z)) {
      grid_map::Position position;
      elevation_map_.getPosition(grid_map::Index(*iterator), position);
      output_cloud.push_back(pcl::PointXYZ(position.x(), position.y(), z));
    }
  }
  output_cloud.width = output_cloud.points.size();
  output_cloud.height = 1;
  output_cloud.points.resize(output_cloud.width * output_cloud.height);
  output_cloud.is_dense = false;

  pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud_ptr;
  output_cloud_ptr = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>(output_cloud);
  return output_cloud_ptr;
}

void ElevationMapLoaderNode::saveElevationMap()
{
  const bool saving_successful = grid_map::GridMapRosConverter::saveToBag(
    elevation_map_, *data_manager_.elevation_map_path_, "elevation_map");
  RCLCPP_INFO_STREAM(
    this->get_logger(), "Saving elevation map successful: " << std::boolalpha << saving_successful);
}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ElevationMapLoaderNode)
