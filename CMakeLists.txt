# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.8)
project(ros2-bevfusion)

set(CMAKE_BUILD_PYTHON $ENV{USE_Python})
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(spconv_root ./lib/3DSparseConvolution/libspconv_cuda12)
set(spconv_include ./lib/3DSparseConvolution/libspconv_cuda12/include)
set(spconv_lib ./lib/3DSparseConvolution/libspconv_cuda12/lib/${arch})

# 1. 添加调试信息
message(STATUS "System architecture: ${arch}")
message(STATUS "CUDA version: $ENV{SPCONV_CUDA_VERSION}")

find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

# 2. 修复 OpenCV 版本要求（改为 4.5 或移除版本限制）
# find_package(OpenCV 4.5 REQUIRED)  # 如果你确定是 4.5
find_package(OpenCV REQUIRED)        # 或者不指定版本

find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(image_transport REQUIRED)  # 添加这个，因为可能也需要

# 3. 添加调试：打印找到的包信息
message(STATUS "OpenCV_FOUND: ${OpenCV_FOUND}")
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")

if("$ENV{SPCONV_CUDA_VERSION}" EQUAL "11.4")
  # For cuda 11.x
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB -Wno-unused-parameter")
  set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")
else()
  # For cuda 12.x
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB -Wno-unused-parameter")
  set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
  -gencode arch=compute_87,code=sm_87
)

include_directories(
  /opt/ros/humble/include
  ${rclcpp_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${vision_msgs_INCLUDE_DIRS}
  ${Protobuf_INCLUDE_DIRS}
  $ENV{CUDA_Inc}
  ${OpenCV_INCLUDE_DIRS}
  $ENV{TensorRT_Inc}
  $ENV{Python_Inc}
  ${cv_bridge_INCLUDE_DIRS}
  ${message_filters_INCLUDE_DIRS}
  ${image_transport_INCLUDE_DIRS}  # 添加
  ./dependencies/stb
  ./dependencies/dlpack/include
  ./dependencies/pybind11/include
  ./lib/cuOSD/src
  ${spconv_root}/include
  src/common
  src/
)

# 4. 修复链接目录（添加架构特定路径）
link_directories(
  /opt/ros/humble/lib/${arch}-linux-gnu  # ARM架构路径
  /usr/lib/${arch}-linux-gnu              # 系统库路径
  ${OpenCV_LIBRARY_DIRS}
  $ENV{CUDA_Lib}
  $ENV{TensorRT_Lib}
  ${spconv_root}/lib/${arch}
  build
  $ENV{Python_Lib}
)

file(GLOB_RECURSE CORE_FILES 
  src/bevfusion/*.cu 
  src/bevfusion/*.cpp
  src/onnx/*.cpp
  src/common/tensor.cu
  src/common/tensorrt.cpp
)

file(GLOB_RECURSE VISUALIZE_FILES 
  ./lib/cuOSD/src/*.cpp
  ./lib/cuOSD/src/*.cu
  src/common/visualize.cu
)

####################### libbevfusion_core.so ##########################
cuda_add_library(bevfusion_core SHARED 
  ${CORE_FILES}
)

# 5. bevfusion_core 不应该链接 ROS 库（它是底层库）
target_link_libraries(bevfusion_core
  cudart
  cublasLt
  nvinfer
  spconv
  ${Protobuf_LIBRARIES}
  nvinfer_plugin
)
#######################################################################

####################### bevfusion program #############################
cuda_add_executable(bevfusion 
  src/main.cpp
  ${VISUALIZE_FILES}
)

# 6. 修复链接顺序和方式
target_link_libraries(bevfusion
  bevfusion_core
  
  # ROS2 库 - 尝试不同的 cv_bridge 链接方式
  # 方式1：使用现代 CMake 目标（ROS2 Humble 通常这样）
  cv_bridge::cv_bridge
  
  # 方式2：如果方式1失败，取消注释下面这行
  # ${cv_bridge_LIBRARIES}
  
  ${OpenCV_LIBRARIES}           # OpenCV 必须在 cv_bridge 之后
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${vision_msgs_LIBRARIES}
  ${message_filters_LIBRARIES}
  ${image_transport_LIBRARIES}  # 添加
  
  # CUDA 和 TensorRT 库
  nvinfer
  nvinfer_plugin
  cudart
  cublasLt
  
  # 其他库
  spconv
  ${Protobuf_LIBRARIES}
  
  # 系统库
  pthread  # ROS2 需要
  dl       # 动态加载
)

#######################################################################

########################## custom_layernorm.so ################################
cuda_add_library(custom_layernorm SHARED
  src/plugins/custom_layernorm.cu
)

target_link_libraries(custom_layernorm
  nvinfer
  nvinfer_plugin
)
#######################################################################

install(TARGETS bevfusion bevfusion_core custom_layernorm
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
)

# 额外安装 spconv
# 注意：${spconv_lib} 变量可能未定义
# install(FILES ${spconv_lib}/libspconv.so
#   DESTINATION lib/${PROJECT_NAME}
# )

install(FILES ${spconv_lib}/libspconv.so
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()