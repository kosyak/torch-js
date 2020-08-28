cmake_minimum_required(VERSION 3.14)
project (torch-js)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(CUDA)
if(CUDA_FOUND)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip")
if(${CUDA_VERSION_MAJOR} EQUAL 10 AND ${CUDA_VERSION_MINOR} EQUAL 2)
if(WIN32)
file(DOWNLOAD https://download.pytorch.org/libtorch/cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}/libtorch-win-shared-with-deps-1.6.0.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip SHOW_PROGRESS)
else()
file(DOWNLOAD https://download.pytorch.org/libtorch/cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}/libtorch-cxx11-abi-shared-with-deps-1.6.0.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip SHOW_PROGRESS)
endif()
else()
if(WIN32)
file(DOWNLOAD https://download.pytorch.org/libtorch/cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}/libtorch-win-shared-with-deps-1.6.0%2Bcu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip SHOW_PROGRESS)
else()
file(DOWNLOAD https://download.pytorch.org/libtorch/cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip SHOW_PROGRESS)
endif()
endif()
endif()
execute_process(
  COMMAND unzip -o ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.zip
)
else()
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cpu.zip")
if(WIN32)
file(DOWNLOAD https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.6.0%2Bcpu.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cpu.zip)
elseif(APPLE)
file(DOWNLOAD https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.6.0.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cpu.zip)
else()
file(DOWNLOAD https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcpu.zip ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cpu.zip)
endif()
endif()
execute_process(
  COMMAND unzip -o ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cpu.zip
)
endif(CUDA_FOUND)

set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR}/libtorch/)

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

# Include N-API wrappers
execute_process(COMMAND node -p "require('node-addon-api').include"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE NODE_ADDON_API_DIR
        )
string(REPLACE "\n" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${NODE_ADDON_API_DIR})
add_definitions(-DNAPI_VERSION=3)

target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

add_custom_command(TARGET torch-js POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_BINARY_DIR}/libtorch/lib
        $<TARGET_FILE_DIR:torch-js>)