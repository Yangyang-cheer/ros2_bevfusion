/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <cv_bridge/cv_bridge.h>
#include <string.h>
#include <dlfcn.h>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>


#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "message_filters/subscriber.h" 
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "vision_msgs/msg/detection3_d.hpp"



#include "bevfusion/bevfusion.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"
#include "common/timer.hpp"
#include "common/visualize.hpp"



using MySyncPolicy = message_filters::sync_policies::ApproximateTime< 
                              sensor_msgs::msg::Image, 
                              sensor_msgs::msg::Image, 
                              sensor_msgs::msg::Image, 
                              sensor_msgs::msg::Image, 
                              sensor_msgs::msg::Image, 
                              sensor_msgs::msg::Image, 
                              sensor_msgs::msg::PointCloud2>;

// static std::vector<unsigned char*> load_images(const std::string& root) {
//   const char* file_names[] = {"0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
//                               "3-BACK.jpg",  "4-BACK_LEFT.jpg",   "5-BACK_RIGHT.jpg"};

//   std::vector<unsigned char*> images;
//   for (int i = 0; i < 6; ++i) {
//     char path[200];
//     sprintf(path, "%s/%s", root.c_str(), file_names[i]);

//     int width, height, channels;
//     images.push_back(stbi_load(path, &width, &height, &channels, 0));
//     // printf("Image info[%d]: %d x %d : %d\n", i, width, height, channels);
//   }
//   return images;
// }

static void free_images(std::vector<unsigned char*>& images) {
  for (size_t i = 0; i < images.size(); ++i) stbi_image_free(images[i]);

  images.clear();
}

static vision_msgs::msg::Detection3DArray convertBBoxesToDetection3DArray(
    const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes,
    const std::string& frame_id,
    const rclcpp::Time& stamp) {
  
  vision_msgs::msg::Detection3DArray detections;
  detections.header.stamp = stamp;
  detections.header.frame_id = frame_id;
  
  for (const auto& bbox : bboxes) {
    vision_msgs::msg::Detection3D detection;
    
    // 设置bounding box中心位置
    detection.bbox.center.position.x = bbox.position.x;
    detection.bbox.center.position.y = bbox.position.y;
    detection.bbox.center.position.z = bbox.position.z;
    
    // 设置旋转（四元数表示）
    // 绕Z轴旋转
    float half_yaw = bbox.z_rotation * 0.5f;
    detection.bbox.center.orientation.x = 0.0;
    detection.bbox.center.orientation.y = 0.0;
    detection.bbox.center.orientation.z = sin(half_yaw);
    detection.bbox.center.orientation.w = cos(half_yaw);
    
    // 设置bounding box尺寸
    detection.bbox.size.x = bbox.size.l;    // 长度（通常对应x轴）
    detection.bbox.size.y = bbox.size.w;     // 宽度（通常对应y轴）
    detection.bbox.size.z = bbox.size.h;    // 高度（通常对应z轴）
    
    // 设置检测结果（类别和置信度）
    // 可选：设置速度信息
    // detection.twist.linear.x = bbox.velocity.vx;
    // detection.twist.linear.y = bbox.velocity.vy;
    // detection.twist.linear.z = bbox.velocity.vz;
    
    detections.detections.push_back(detection);
  }
  
  return detections;
}

static void visualize(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes, const nv::Tensor& lidar_points,
                      const std::vector<unsigned char*> images, const nv::Tensor& lidar2image, const std::string& save_path,
                      cudaStream_t stream) {
  std::vector<nv::Prediction> predictions(bboxes.size());
  memcpy(predictions.data(), bboxes.data(), bboxes.size() * sizeof(nv::Prediction));

  int padding = 300;
  int lidar_size = 1024;
  int content_width = lidar_size + padding * 3;
  int content_height = 1080;
  nv::SceneArtistParameter scene_artist_param;
  scene_artist_param.width = content_width;
  scene_artist_param.height = content_height;
  scene_artist_param.stride = scene_artist_param.width * 3;

  nv::Tensor scene_device_image(std::vector<int>{scene_artist_param.height, scene_artist_param.width, 3}, nv::DataType::UInt8);
  scene_device_image.memset(0x00, stream);

  scene_artist_param.image_device = scene_device_image.ptr<unsigned char>();
  auto scene = nv::create_scene_artist(scene_artist_param);

  nv::BEVArtistParameter bev_artist_param;
  bev_artist_param.image_width = content_width;
  bev_artist_param.image_height = content_height;
  bev_artist_param.rotate_x = 70.0f;
  bev_artist_param.norm_size = lidar_size * 0.5f;
  bev_artist_param.cx = content_width * 0.5f;
  bev_artist_param.cy = content_height * 0.5f;
  bev_artist_param.image_stride = scene_artist_param.stride;

  auto points = lidar_points.to_device();
  auto bev_visualizer = nv::create_bev_artist(bev_artist_param);
  bev_visualizer->draw_lidar_points(points.ptr<nvtype::half>(), points.size(0));
  bev_visualizer->draw_prediction(predictions, false);
  bev_visualizer->draw_ego();
  bev_visualizer->apply(scene_device_image.ptr<unsigned char>(), stream);

  nv::ImageArtistParameter image_artist_param;
  image_artist_param.num_camera = images.size();
  image_artist_param.image_width = 1600;
  image_artist_param.image_height = 900;
  image_artist_param.image_stride = image_artist_param.image_width * 3;
  image_artist_param.viewport_nx4x4.resize(images.size() * 4 * 4);
  memcpy(image_artist_param.viewport_nx4x4.data(), lidar2image.ptr<float>(),
         sizeof(float) * image_artist_param.viewport_nx4x4.size());

  int gap = 0;
  int camera_width = 500;
  int camera_height = static_cast<float>(camera_width / (float)image_artist_param.image_width * image_artist_param.image_height);
  int offset_cameras[][3] = {
      {-camera_width / 2, -content_height / 2 + gap, 0},
      {content_width / 2 - camera_width - gap, -content_height / 2 + camera_height / 2, 0},
      {-content_width / 2 + gap, -content_height / 2 + camera_height / 2, 0},
      {-camera_width / 2, +content_height / 2 - camera_height - gap, 1},
      {-content_width / 2 + gap, +content_height / 2 - camera_height - camera_height / 2, 0},
      {content_width / 2 - camera_width - gap, +content_height / 2 - camera_height - camera_height / 2, 1}};

  auto visualizer = nv::create_image_artist(image_artist_param);
  for (size_t icamera = 0; icamera < images.size(); ++icamera) {
    int ox = offset_cameras[icamera][0] + content_width / 2;
    int oy = offset_cameras[icamera][1] + content_height / 2;
    bool xflip = static_cast<bool>(offset_cameras[icamera][2]);
    visualizer->draw_prediction(icamera, predictions, xflip);

    nv::Tensor device_image(std::vector<int>{900, 1600, 3}, nv::DataType::UInt8);
    device_image.copy_from_host(images[icamera], stream);

    if (xflip) {
      auto clone = device_image.clone(stream);
      scene->flipx(clone.ptr<unsigned char>(), clone.size(1), clone.size(1) * 3, clone.size(0), device_image.ptr<unsigned char>(),
                   device_image.size(1) * 3, stream);
      checkRuntime(cudaStreamSynchronize(stream));
    }
    visualizer->apply(device_image.ptr<unsigned char>(), stream);

    scene->resize_to(device_image.ptr<unsigned char>(), ox, oy, ox + camera_width, oy + camera_height, device_image.size(1),
                     device_image.size(1) * 3, device_image.size(0), 0.8f, stream);
    checkRuntime(cudaStreamSynchronize(stream));
  }

  printf("Save to %s\n", save_path.c_str());
  stbi_write_jpg(save_path.c_str(), scene_device_image.size(1), scene_device_image.size(0), 3,
                 scene_device_image.to_host(stream).ptr(), 100);
}

std::shared_ptr<bevfusion::Core> create_core(const std::string& model, const std::string& precision) {

  printf("Create by %s, %s\n", model.c_str(), precision.c_str());
  bevfusion::camera::NormalizationParameter normalization;
  normalization.image_width = 1600;
  normalization.image_height = 900;
  normalization.output_width = 704;
  normalization.output_height = 256;
  normalization.num_camera = 6;
  normalization.resize_lim = 0.48f;
  normalization.interpolation = bevfusion::camera::Interpolation::Bilinear;

  float mean[3] = {0.485, 0.456, 0.406};
  float std[3] = {0.229, 0.224, 0.225};
  normalization.method = bevfusion::camera::NormMethod::mean_std(mean, std, 1 / 255.0f, 0.0f);

  bevfusion::lidar::VoxelizationParameter voxelization;
  voxelization.min_range = nvtype::Float3(-54.0f, -54.0f, -5.0);
  voxelization.max_range = nvtype::Float3(+54.0f, +54.0f, +3.0);
  voxelization.voxel_size = nvtype::Float3(0.075f, 0.075f, 0.2f);
  voxelization.grid_size = voxelization.compute_grid_size(voxelization.max_range, voxelization.min_range, voxelization.voxel_size);
  voxelization.max_points_per_voxel = 10;
  voxelization.max_points = 300000;
  voxelization.max_voxels = 160000;
  voxelization.num_feature = 4;

  bevfusion::lidar::SCNParameter scn;
  scn.voxelization = voxelization;
  scn.model = nv::format("model/%s/lidar.backbone.xyz.onnx", model.c_str());
  scn.order = bevfusion::lidar::CoordinateOrder::XYZ;

  if (precision == "int8") {
    scn.precision = bevfusion::lidar::Precision::Int8;
  } else {
    scn.precision = bevfusion::lidar::Precision::Float16;
  }

  bevfusion::camera::GeometryParameter geometry;
  geometry.xbound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
  geometry.ybound = nvtype::Float3(-54.0f, 54.0f, 0.3f);
  geometry.zbound = nvtype::Float3(-10.0f, 10.0f, 20.0f);
  geometry.dbound = nvtype::Float3(1.0, 60.0f, 0.5f);
  geometry.image_width = 704;
  geometry.image_height = 256;
  geometry.feat_width = 88;
  geometry.feat_height = 32;
  geometry.num_camera = 6;
  geometry.geometry_dim = nvtype::Int3(360, 360, 80);

  bevfusion::head::transbbox::TransBBoxParameter transbbox;
  transbbox.out_size_factor = 8;
  transbbox.pc_range = {-54.0f, -54.0f};
  transbbox.post_center_range_start = {-61.2, -61.2, -10.0};
  transbbox.post_center_range_end = {61.2, 61.2, 10.0};
  transbbox.voxel_size = {0.075, 0.075};
  transbbox.model = nv::format("model/%s/build/head.bbox.plan", model.c_str());

  // if you got an inaccurate boundingbox result please turn on the layernormplugin plan.
  // transbbox.model = nv::format("model/%s/build/head.bbox.layernormplugin.plan", model.c_str());
  transbbox.confidence_threshold = 0.12f;
  transbbox.sorted_bboxes = true;

  bevfusion::CoreParameter param;
  param.camera_model = nv::format("model/%s/build/camera.backbone.plan", model.c_str());
  param.normalize = normalization;
  param.lidar_scn = scn;
  param.geometry = geometry;
  param.transfusion = nv::format("model/%s/build/fuser.plan", model.c_str());
  param.transbbox = transbbox;
  param.camera_vtransform = nv::format("model/%s/build/camera.vtransform.plan", model.c_str());
  return bevfusion::create_core(param);
}

class BevfusionNode : public rclcpp::Node {
  private:
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr pub_bbox_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_image1_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_image2_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_image3_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_image4_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_image5_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_image6_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_pointcloud_;
    const char* data      = "example-data";
    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> synchronizer_;
    nv::Tensor camera2lidar = nv::Tensor::load(nv::format("%s/camera2lidar.tensor", data), false);
  
    nv::Tensor camera_intrinsics = nv::Tensor::load(nv::format("%s/camera_intrinsics.tensor", data), false);

    nv::Tensor lidar2image = nv::Tensor::load(nv::format("%s/lidar2image.tensor", data), false);
   
    nv::Tensor img_aug_matrix = nv::Tensor::load(nv::format("%s/img_aug_matrix.tensor", data), false);
  
    std::shared_ptr<bevfusion::Core> core;
    const char* model= "resnet50int8";
    const char* precision = "int8";
   
    cudaStream_t stream;


   std::vector<unsigned char*> load_image_from_ros(std::vector<sensor_msgs::msg::Image::ConstSharedPtr>& ros_images) {
    std::vector<unsigned char*> images;
    int width;
    int height;
    int channels;
    for (const auto& ros_image : ros_images) {
      // cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(ros_image, sensor_msgs::image_encodings::BGR8);
      // cv::Mat bgr_image = cv_ptr->image;
      // cv::Mat rgb_image;
      // cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);
      // width = rgb_image.cols;
      // height = rgb_image.rows;
      // channels = rgb_image.channels();
      // size_t image_size = (width) * (height) * (channels);
      // unsigned char* image_data = (unsigned char*)malloc(image_size);
      // memcpy(image_data, rgb_image.data, image_size);

      ////////////////////////////////////////////////////////////
  //     unsigned char* image_data_stbi = stbi_load_from_memory(
  //     image_data,      // 图像数据缓冲区
  //     image_size,      // 数据长度
  //     &width,                       // 返回图像宽度
  //     &height,                      // 返回图像高度
  //     &channels,                    // 返回通道数
  //     0                            // 请求的通道数（0=保持原样）
  //     );
  //     free(image_data);
  //     images.push_back(image_data_stbi);

  //   }
  // return images;
  // }
  ////////////////////////////////////////////////////////
  //不确定stbi的这个转换是不是必要的，，，也不确定它能不能转rgb图片,如果不能转rgb图片就用下面的注释代码

    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(ros_image, "bgr8");
    cv::Mat bgr_image = cv_ptr->image;
    cv::Mat rgb_image;
    cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);
    
    // 2. 编码为JPEG（得到vector）
    std::vector<unsigned char> jpeg_buffer;
    cv::imencode(".jpg", rgb_image, jpeg_buffer);
    
    // 3. stbi解码JPEG
    unsigned char* image_data_stbi = stbi_load_from_memory(
        jpeg_buffer.data(),      // ✅ vector的数据指针
        jpeg_buffer.size(),      // ✅ vector的大小
        &width, &height, &channels, 3
    );
  images.push_back(image_data_stbi);
  }
  

  return images;
}

  void fusion_callback(const sensor_msgs::msg::Image::ConstSharedPtr& img1,
                       const sensor_msgs::msg::Image::ConstSharedPtr& img2,
                        const sensor_msgs::msg::Image::ConstSharedPtr& img3,
                        const sensor_msgs::msg::Image::ConstSharedPtr& img4,
                        const sensor_msgs::msg::Image::ConstSharedPtr& img5,
                        const sensor_msgs::msg::Image::ConstSharedPtr& img6,
                       const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud) {
     if (!cloud || cloud->data.empty()) {
        RCLCPP_DEBUG(this->get_logger(), "Point cloud is empty, skipping");
        return;
    }
    
    // 2. 检查所有图像是否为空
    for (const auto& img : {img1, img2, img3, img4, img5, img6}) {
        if (!img || img->data.empty()) {
            RCLCPP_DEBUG(this->get_logger(), "Image is empty, skipping");
            return;
        }
    }
    std::vector<sensor_msgs::msg::Image::ConstSharedPtr> image_list = {img1, img2, img3, img4, img5, img6};
    auto images = this->load_image_from_ros(image_list);
    nv::Tensor lidar_points_tensor = nv::Tensor::load_from_pointcloud2(*cloud, true);
    auto bboxes = core->forward((const unsigned char**)images.data(), lidar_points_tensor, lidar_points_tensor.size(0), stream);
    auto detections = convertBBoxesToDetection3DArray(bboxes, cloud ->header.frame_id, cloud->header.stamp);    
    pub_bbox_->publish(detections);
    free_images(images);
  }
  public:
    BevfusionNode () : Node("bev_fusion_node") {

      // pub_bbox_ = create_publisher<vision_msgs::msg::Detection3DArray>("output/bboxes_3d", 10);
      // sub_image1_.subscribe(this, "/fisheye/bright/image_raw");
      // sub_image2_.subscribe(this, "/fisheye/bleft/image_raw");
      // sub_image3_.subscribe(this, "/fisheye/left/image_raw");
      // sub_image4_.subscribe(this, "/fisheye/right/image_raw");
      // sub_image5_.subscribe(this, "/fisheye/right/image_raw");
      // sub_image6_.subscribe(this, "/fisheye/right/image_raw");
      //sub_pointcloud_.subscribe(this, "/lidar_points");
       pub_bbox_ = create_publisher<vision_msgs::msg::Detection3DArray>("output/bboxes_3d", 10);
      sub_image1_.subscribe(this, "image_0");
      sub_image2_.subscribe(this, "image_1");
      sub_image3_.subscribe(this, "image_2");
      sub_image4_.subscribe(this, "image_3");
      sub_image5_.subscribe(this, "image_4");
      sub_image6_.subscribe(this, "image_5");
      sub_pointcloud_.subscribe(this, "pointcloud");
     
      cudaStreamCreate(&stream);
      core = create_core(model, precision);
      core->update(camera2lidar.ptr<float>(), camera_intrinsics.ptr<float>(), lidar2image.ptr<float>(), img_aug_matrix.ptr<float>(), stream);
      core->print();
      core->set_timer(true);

    

      synchronizer_ = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(
          MySyncPolicy(10), sub_image1_, sub_image2_, sub_image3_, sub_image4_, sub_image5_, sub_image6_, sub_pointcloud_);




      auto callback_func = std::bind(&BevfusionNode::fusion_callback, this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3,
        std::placeholders::_4,
        std::placeholders::_5,
        std::placeholders::_6,
        std::placeholders::_7
      );
      synchronizer_->registerCallback(callback_func);
                  // message_filters 是命名空间，不是类
      
  //     synchronizer_->registerCallback(
  //       [this](const sensor_msgs::msg::Image::ConstSharedPtr& img1,
  //               const sensor_msgs::msg::Image::ConstSharedPtr& img2,
  //               const sensor_msgs::msg::Image::ConstSharedPtr& img3,
  //               const sensor_msgs::msg::Image::ConstSharedPtr& img4,
  //               const sensor_msgs::msg::Image::ConstSharedPtr& img5,
  //               const sensor_msgs::msg::Image::ConstSharedPtr& img6,
  //               const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud) {
  //                 this->fusion_callback(img1, img2, img3, img4, img5, img6, cloud);
  //               } 
  //     );
  }
  // 当同步器凑齐一组时间接近的消息时，就调用fusion_callback函数。这是一个简化的成员函数绑定写法
  // [this]:捕获this指针让lambda能访问类的成员变量和函数,否则lambda内部无法调用this->fusion_callback
  // this->fusion_callback(img1, img2, img3, img4, img5, img6, cloud);将参数原样转发给成员函数，img1, img2, img3, img4, cloud是成员函数真实定义的形式参数名
  // 这里使用lambda表达式进行绑定，是为了解决 "C++成员函数指针不能直接作为回调" 的核心问题
  // 它将“需要this指针的成员函数”和“调用时所需的实际参数”预先打包在一起
  // 把一个需要特定对象实例（this）的成员函数，转换成一个可以独立调用的函数对象。
  //     registerCallback 是 Synchronizer 类的一个方法，它允许你注册一个回调函数。通过绑定，
  // 你告诉同步器：“当你凑齐一组同步消息后，请调用我提供的函数（我的fusion_callback）来处理它们。”

};


int main(int argc, char** argv) {
  std::cout << "argc = " << argc << std::endl;
  rclcpp::init(argc, argv);
  auto node = std::make_shared<BevfusionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
}


 



  // const char* data      = "example-data";
  // const char* model     = "resnet50int8";
  // const char* precision = "int8";

  // if (argc > 1) data      = argv[1];
  // if (argc > 2) model     = argv[2];
  // if (argc > 3) precision = argv[3];
  // dlopen("libcustom_layernorm.so", RTLD_NOW);

  
  // if (core == nullptr) {
  //   printf("Core has been failed.\n");
  //   return -1;
  // }

  
  
 
  

  
  
//   // warmup
//   auto bboxes =
//       core->forward((const unsigned char**)images.data(), lidar_points.ptr<nvtype::half>(), lidar_points.size(0), stream);

//   // evaluate inference time
//   for (int i = 0; i < 5; ++i) {
//     core->forward((const unsigned char**)images.data(), lidar_points.ptr<nvtype::half>(), lidar_points.size(0), stream);
//   }

//   // visualize and save to jpg
//   visualize(bboxes, lidar_points, images, lidar2image, "build/cuda-bevfusion.jpg", stream);

//   // destroy memory
//   free_images(images);
//   checkRuntime(cudaStreamDestroy(stream));

//   printf("[Warning]: If you got an inaccurate boundingbox result please turn on the layernormplugin plan. (main.cpp:207)\n");
//   return 0;
// }