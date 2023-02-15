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
  param_file_path_ = this->declare_parameter("param_file_path", "path_default");
  map_frame_ = this->declare_parameter("map_frame", "map");
  use_inpaint_ = this->declare_parameter("use_inpaint", true);
  use_incremental_generation_ = this->declare_parameter("use_incremental_generation", true);
  data_manager_.use_incremental_generation_ = use_incremental_generation_;
  bool use_differential_load = this->declare_parameter<bool>("use_differential_load", true);
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

  rclcpp::QoS durable_qos{1};
  durable_qos.transient_local();
  pub_elevation_map_ =
    this->create_publisher<grid_map_msgs::msg::GridMap>("output/elevation_map", durable_qos);

  if (use_elevation_map_cloud_publisher_) {
    pub_elevation_map_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "output/elevation_map_cloud", durable_qos);
  }

  using std::placeholders::_1;
  sub_vector_map_ = this->create_subscription<autoware_auto_mapping_msgs::msg::HADMapBin>(
    "input/vector_map", durable_qos, std::bind(&ElevationMapLoaderNode::onVectorMap, this, _1));
  sub_map_hash_ = create_subscription<tier4_external_api_msgs::msg::MapHash>(
    "/api/autoware/get/map/info/hash", durable_qos,
    std::bind(&ElevationMapLoaderNode::onMapHash, this, _1));
  if (use_differential_load) {
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
        this, get_clock(), period_ns, std::bind(&ElevationMapLoaderNode::timerCallback, this));
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
    createElevationMap();
    RCLCPP_INFO(
      this->get_logger(), "Finish creating elevation map from pointcloud map. Start inpaint");
    if (use_inpaint_) {
      inpaintElevationMap(inpaint_radius_);
    }
    if (true) {
      RCLCPP_INFO(this->get_logger(), "compare maps");
      compareElevationMapWithOtherGridMap();
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
      createElevationMap();
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

void ElevationMapLoaderNode::timerCallback()
{
  {
    if (use_incremental_generation_) {
      if (!finish_pub_elevation_map_) {
        std::shared_ptr<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> map_pcl_vector =
          std::make_shared<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>>();
        ;
        ElevationMapLoaderNode::receiveMapVector(map_pcl_vector);
        RCLCPP_INFO(this->get_logger(), "receive service with pointcloud_map");
        data_manager_.map_pcl_vector_ptr_ =
          pcl::make_shared<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>>(*map_pcl_vector);
      }
    } else {
      ElevationMapLoaderNode::receiveMap();
      RCLCPP_INFO(this->get_logger(), "receive service with pointcloud_map");
    }
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
  {
    pcl::PointCloud<pcl::PointXYZ> map_pcl;
    pcl::fromROSMsg<pcl::PointXYZ>(*pointcloud_map, map_pcl);
    data_manager_.map_pcl_ptr_ = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>(map_pcl);
  }
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

void ElevationMapLoaderNode::receiveMap()
{
  pcl::PointCloud<pcl::PointXYZ> map_pcl;
  sensor_msgs::msg::PointCloud2 pointcloud_map;
  // create a loading request with mode = 1
  auto request = std::make_shared<autoware_map_msgs::srv::GetDifferentialPointCloudMap::Request>();
  // request all area
  request->area.type = autoware_map_msgs::msg::AreaInfo::ALL_AREA;
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
    // update cached_ids
    request->cached_ids = cached_ids;
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
    if (result.get()->new_pointcloud_with_ids.empty()) {
      RCLCPP_INFO(this->get_logger(), "finish receiving");
      is_all_received = true;
    } else {
      // concat maps
      for (const auto & new_pointcloud_with_id : result.get()->new_pointcloud_with_ids) {
        if (pointcloud_map.width == 0) {
          pointcloud_map = new_pointcloud_with_id.pointcloud;
        } else {
          pointcloud_map.width += new_pointcloud_with_id.pointcloud.width;
          pointcloud_map.row_step += new_pointcloud_with_id.pointcloud.row_step;
          pointcloud_map.data.insert(
            pointcloud_map.data.end(), new_pointcloud_with_id.pointcloud.data.begin(),
            new_pointcloud_with_id.pointcloud.data.end());
        }
        cached_ids.push_back(new_pointcloud_with_id.cell_id);
      }
    }
  }
  pcl::fromROSMsg<pcl::PointXYZ>(pointcloud_map, map_pcl);
  data_manager_.map_pcl_ptr_ = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>(map_pcl);
}

void ElevationMapLoaderNode::receiveMapVector(
  const std::shared_ptr<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> pointcloud_map_vector)
{
  // create a loading request with mode = 1
  auto request = std::make_shared<autoware_map_msgs::srv::GetDifferentialPointCloudMap::Request>();
  // request all area
  request->area.type = autoware_map_msgs::msg::AreaInfo::ALL_AREA;
  std::vector<std::string> cached_ids{};
  bool is_all_received = false;
  while (!is_all_received) {
    // update cached_ids
    request->cached_ids = cached_ids;
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

    if (result.get()->new_pointcloud_with_ids.empty()) {
      is_all_received = true;
    } else {
      // push each partial map to pointcloud_map_vector
      for (const auto & new_pointcloud_with_id : result.get()->new_pointcloud_with_ids) {
        pcl::PointCloud<pcl::PointXYZ> map_pcl;
        pcl::fromROSMsg<pcl::PointXYZ>(new_pointcloud_with_id.pointcloud, map_pcl);
        const pcl::PointCloud<pcl::PointXYZ>::Ptr map_pcl_ptr =
          pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>(map_pcl);
        pointcloud_map_vector->push_back(map_pcl_ptr);
        cached_ids.push_back(new_pointcloud_with_id.cell_id);
      }
    }
  }
}

void ElevationMapLoaderNode::createElevationMap()
{
  if (use_incremental_generation_) {
    {
      // get bound
      double max_bound_x;
      double max_bound_y;
      double min_bound_x;
      double min_bound_y;
      std::tie(max_bound_x, max_bound_y, min_bound_x, min_bound_y) = getBound();

      std::vector<grid_map::GridMap> grid_map_vector;
      for (const auto & map_pcl : *data_manager_.map_pcl_vector_ptr_) {
        grid_map_vector.push_back(createElevationMapFromPointcloud(map_pcl));
      }

      // create elevation map for all area
      RCLCPP_INFO(this->get_logger(), "create elevation map for all area");
      grid_map::Length length =
        grid_map::Length(max_bound_x - min_bound_x, max_bound_y - min_bound_y);
      double resolution = 0.3;  // elevation_map_.getResolution();  // node paramから取るべき
      grid_map::Position position =
        grid_map::Position((max_bound_x + min_bound_x) / 2.0, (max_bound_y + min_bound_y) / 2.0);
      elevation_map_.clearAll();
      elevation_map_.setGeometry(length, resolution, position);
      double value = 0.0;
      elevation_map_.add(layer_name_, value);

      // create elevation map for all area
      RCLCPP_INFO(
        this->get_logger(),
        "update cell value in elevation map for all area by that in grid_map for each pointcloud "
        "map");
      for (const auto & grid_map : grid_map_vector) {
        grid_map::Matrix gridMapData = grid_map.get("elevation");
        unsigned int linearGridMapSize = grid_map.getSize().prod();
        for (unsigned int linearIndex = 0; linearIndex < linearGridMapSize; ++linearIndex) {
          const grid_map::Index index(
            grid_map::getIndexFromLinearIndex(linearIndex, grid_map.getSize()));
          // update cell value in elevation map for all area
          //   by that in grid_map for each pointcloud map
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
    elevation_map_ = createElevationMapFromPointcloud(data_manager_.map_pcl_ptr_);
  }
}

std::tuple<double, double, double, double> ElevationMapLoaderNode::getBound()
{
  bool bound_flag = false;  // TODO(Shin-kyoto): bound_flagは暫定対応．
  double all_max_bound_x = 0.0;
  double all_max_bound_y = 0.0;
  double all_min_bound_x = 0.0;
  double all_min_bound_y = 0.0;
  for (const auto & map_pcl : *data_manager_.map_pcl_vector_ptr_) {
    // get bound
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

grid_map::GridMap ElevationMapLoaderNode::createElevationMapFromPointcloud(
  pcl::PointCloud<pcl::PointXYZ>::Ptr map_pcl)
{
  auto grid_map_logger = rclcpp::get_logger("grid_map_logger");
  grid_map_logger.set_level(rclcpp::Logger::Level::Error);
  pcl::shared_ptr<grid_map::GridMapPclLoader> grid_map_pcl_loader =
    pcl::make_shared<grid_map::GridMapPclLoader>(grid_map_logger);

  // set point cloud used for elevation map
  grid_map_pcl_loader->loadParameters(param_file_path_);
  if (lane_filter_.use_lane_filter_) {
    // filter point cloud by lanelets
    const auto convex_hull = getConvexHull(map_pcl);
    lanelet::ConstLanelets intersected_lanelets =
      getIntersectedLanelets(convex_hull, lane_filter_.road_lanelets_);
    pcl::PointCloud<pcl::PointXYZ>::Ptr lane_filtered_map_pcl_ptr =
      getLaneFilteredPointCloud(intersected_lanelets, map_pcl);
    grid_map_pcl_loader->setInputCloud(lane_filtered_map_pcl_ptr);
  } else {
    grid_map_pcl_loader->setInputCloud(map_pcl);
  }
  // create elevation map from point cloud
  {
    const auto start = std::chrono::high_resolution_clock::now();
    grid_map_pcl_loader->preProcessInputCloud();
    grid_map_pcl_loader->initializeGridMapGeometryFromInputCloud();
    grid_map_pcl_loader->addLayerFromInputCloud(layer_name_);
    grid_map::grid_map_pcl::printTimeElapsedToRosInfoStream(
      start, "Finish creating elevation map. Total time: ", this->get_logger());
  }
  return grid_map_pcl_loader->getGridMap();
}

void ElevationMapLoaderNode::inpaintElevationMap(const float radius)
{
  // Convert elevation layer to OpenCV image to fill in holes.
  // Get the inpaint mask (nonzero pixels indicate where values need to be filled in).
  elevation_map_.add("inpaint_mask", 0.0);

  elevation_map_.setBasicLayers(std::vector<std::string>());
  for (grid_map::GridMapIterator iterator(elevation_map_); !iterator.isPastEnd(); ++iterator) {
    if (!elevation_map_.isValid(*iterator, layer_name_)) {
      elevation_map_.at("inpaint_mask", *iterator) = 1.0;
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

  const float radius_in_pixels = radius / elevation_map_.getResolution();
  cv::inpaint(original_image, mask, filled_image, radius_in_pixels, cv::INPAINT_NS);

  grid_map::GridMapCvConverter::addLayerFromImage<unsigned char, 3>(
    filled_image, layer_name_, elevation_map_, min_value, max_value);
  cv::imwrite("original_image.jpg", original_image);
  cv::imwrite("mask.jpg", mask);
  cv::imwrite("filled_image.jpg", filled_image);
  elevation_map_.erase("inpaint_mask");
}

void ElevationMapLoaderNode::compareElevationMapWithOtherGridMap()
{
  // compare maps
  // load map
  grid_map::GridMap elevation_map_original;
  grid_map::GridMapRosConverter::loadFromBag(
    elevation_map_directory_original_, "elevation_map", elevation_map_original);

  RCLCPP_INFO(this->get_logger(), "compare 2 maps");
  for (const auto & lanelet : lane_filter_.road_lanelets_) {
    auto lane_polygon = lanelet.polygon2d().basicPolygon();
    grid_map::Polygon polygon;
    for (const auto & p : lane_polygon) {
      polygon.addVertex(grid_map::Position(p[0], p[1]));
    }
    std::ofstream ofs_diff_within_lanelet("diff_within_lanelet.csv", std::ios::app);
    for (grid_map::PolygonIterator iterator(elevation_map_, polygon); !iterator.isPastEnd(); ++iterator) {
      grid_map::Position position;
      elevation_map_.getPosition(*iterator, position);
      float diff = fabs(
        elevation_map_.at("elevation", *iterator) -
        elevation_map_original.at("elevation", *iterator));
      ofs_diff_within_lanelet << (*iterator)(0) << "," << (*iterator)(1) << "," << diff << std::endl;
      if (diff > 0.1) {
        RCLCPP_INFO(this->get_logger(), "not equal");
      }
    }
  }
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
