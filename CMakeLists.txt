cmake_minimum_required(VERSION 3.12)
project (torch-js)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(CUDA)
if(CUDA_FOUND)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip")
if(WIN32)
file(DOWNLOAD https://download.pytorch.org/libtorch/nightly/cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}/libtorch-win-shared-with-deps-latest.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip)
elseif(APPLE)
file(DOWNLOAD https://download.pytorch.org/libtorch/nightly/cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}/libtorch-macos-latest.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip)
else()
file(DOWNLOAD https://download.pytorch.org/libtorch/nightly/cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}/libtorch-shared-with-deps-latest.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip)
endif()
endif()
execute_process(
  COMMAND unzip -o ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip
)
else()
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cpu.zip")
if(WIN32)
file(DOWNLOAD https://download.pytorch.org/libtorch/nightly/cpu/libtorch-win-shared-with-deps-latest.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cpu.zip)
elseif(APPLE)
file(DOWNLOAD https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-latest.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cpu.zip)
else()
file(DOWNLOAD https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cpu.zip)
endif()
endif()
execute_process(
  COMMAND unzip -o ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cpu.zip
)
endif(CUDA_FOUND)

set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libtorch/)

# Use C++14
set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Torch REQUIRED)

include_directories(${CMAKE_JS_INC})
file(GLOB SOURCE_FILES "src/*.cc" "src/*.h")
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/node_modules/node-addon-api")
add_definitions(-DNAPI_VERSION=3)

target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

add_custom_command(TARGET torch-js POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/libtorch/lib
        $<TARGET_FILE_DIR:torch-js>)