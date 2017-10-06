cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(Astra_Deflection)
find_package(PCL 1.7 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pcd_write_test pcd_write.cpp)
target_link_libraries(pcd_write_test ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES})

add_executable(viewer pcl_viewer.cpp)
target_link_libraries(viewer ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES}  pcl_common pcl_io pcl_kdtree pcl_visualization)

