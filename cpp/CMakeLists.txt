# C29 — CMakeLists v3: core lib + detector CLI + one executable per tests/*.cpp
cmake_minimum_required(VERSION 3.16)
project(yolov8_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(ORT_DIR "" CACHE PATH "Path to onnxruntime-linux-x64-<ver>")
option(SANITIZE "Build with AddressSanitizer + UBSan" OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)

add_compile_options(-Wall -Wextra)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g")
if(SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

file(GLOB CORE_SRC ${CMAKE_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM CORE_SRC ${CMAKE_SOURCE_DIR}/src/main.cpp)
add_library(yolo_core STATIC ${CORE_SRC})
target_include_directories(yolo_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(yolo_core SYSTEM PUBLIC ${ORT_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo_core PUBLIC ${ORT_DIR}/lib/libonnxruntime.so ${OpenCV_LIBS})

add_executable(detector src/main.cpp)
target_link_libraries(detector PRIVATE yolo_core)
set_target_properties(detector PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)

file(GLOB TEST_SRC ${CMAKE_SOURCE_DIR}/tests/*.cpp)
foreach(t ${TEST_SRC})
  get_filename_component(name ${t} NAME_WE)
  add_executable(${name} ${t})
  target_link_libraries(${name} PRIVATE yolo_core)
  set_target_properties(${name} PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
endforeach()
