cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(Astra_Deflection)
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_path(OpenNI2_INCLUDEDIR
	  NAMES OpenNI.h
	  HINTS $ENV{OPENNI2_INCLUDE})
find_library(OpenNI2_LIBRARIES
	     NAMES OpenNI2 DummyDevice OniFile PS1090
	     HINTS $ENV{OPENNI2_REDIST}
	     PATH_SUFFIXES lib) 
message(STATUS ${OpenNI2_LIBRARIES})

add_executable(pcd_write_test pcd_write.cpp)
target_link_libraries(pcd_write_test ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES})

add_executable(viewer pcl_viewer.cpp)
target_link_libraries(viewer ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES}  pcl_common pcl_io pcl_kdtree pcl_visualization)

add_executable(visualizer pcl_visualizer.cpp)
target_link_libraries(visualizer ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES}  pcl_common pcl_io pcl_kdtree pcl_visualization)

#add_executable(openni2_viewer pcl_openni2_viewer.cpp)
#target_link_libraries(openni2_viewer ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES}  pcl_common pcl_io pcl_kdtree pcl_visualization)

add_executable(openni_viewer pcl_openni_viewer.cpp)
target_link_libraries(openni_viewer ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES}  pcl_common pcl_io pcl_kdtree pcl_visualization)

add_executable(openni_read openni_read.cpp)
target_link_libraries(openni_read ${OpenNI2_LIBRARIES})
