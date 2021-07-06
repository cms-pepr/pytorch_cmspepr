cmake_minimum_required(VERSION 3.0)
project(torchcmspepr)
set(CMAKE_CXX_STANDARD 14)
set(TORCHCMSPEPR_VERSION 0.0.1)

option(WITH_CUDA "Enable CUDA support" OFF)

if(WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
  add_definitions(-DWITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_35 --expt-relaxed-constexpr")
endif()

find_package(Python3 COMPONENTS Development)
find_package(Torch REQUIRED)

include_directories(csrc/)
file(GLOB HEADERS csrc/cmspepr.h csrc/helpers.h)
file(GLOB OPERATOR_SOURCES csrc/cpu/*.h csrc/cpu/*.cpp csrc/*.cpp)
if(WITH_CUDA)
  file(GLOB OPERATOR_SOURCES ${OPERATOR_SOURCES} csrc/cuda/*.h csrc/cuda/*.cu)
endif()

add_library(${PROJECT_NAME} SHARED ${OPERATOR_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} Python3::Python)
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME TorchCMSPepr)

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${HEADERS}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(TORCHCMSPEPR_CMAKECONFIG_INSTALL_DIR "share/cmake/TorchCMSPepr" CACHE STRING "install path for TorchCMSPeprConfig.cmake")

configure_package_config_file(cmake/TorchCMSPeprConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/TorchCMSPeprConfig.cmake"
  INSTALL_DESTINATION ${TORCHCMSPEPR_CMAKECONFIG_INSTALL_DIR})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/TorchCMSPeprConfigVersion.cmake
  VERSION ${TORCHCMSPEPR_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TorchCMSPeprConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/TorchCMSPeprConfigVersion.cmake
  DESTINATION ${TORCHCMSPEPR_CMAKECONFIG_INSTALL_DIR})

install(TARGETS ${PROJECT_NAME}
  EXPORT TorchCMSPeprTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install(EXPORT TorchCMSPeprTargets
  NAMESPACE TorchCMSPepr::
  DESTINATION ${TORCHCMSPEPR_CMAKECONFIG_INSTALL_DIR})

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
install(FILES
  csrc/cpu/accumulate_knn_cpu.h
  csrc/cpu/accumulate_knn_grad_cpu.h
  csrc/cpu/select_knn_cpu.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cpu)
if(WITH_CUDA)
  install(FILES
    csrc/cuda/accumulate_knn_cuda.h
    csrc/cuda/accumulate_knn_grad_cuda.h
    csrc/cuda/select_knn_cuda.h
    csrc/cuda/select_knn_grad_cuda.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cuda)
endif()

if(WITH_CUDA)
  set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
  set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
endif()
