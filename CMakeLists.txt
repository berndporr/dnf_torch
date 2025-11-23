cmake_minimum_required(VERSION 3.9.0)

project(dnf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

find_package(Boost REQUIRED)
find_package(Torch REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

include(GNUInstallDirs)

add_subdirectory(ecg_filt_demo)

find_package(OpenCV)

enable_testing ()

if (MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wconversion -Wextra -pedantic)
endif()

add_library(dnf STATIC dnf.cpp)

target_link_libraries(dnf ${TORCH_LIBRARIES})

set_target_properties(dnf PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  PUBLIC_HEADER dnf.h)

install(TARGETS dnf EXPORT dnf-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dnf)

install(EXPORT dnf-targets
  DESTINATION lib/cmake/dnf
  FILE dnf-config.cmake
)
