cmake_minimum_required(VERSION 3.21 FATAL_ERROR)   
project(mnist_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# 输出 exe 到项目根目录
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# LibTorch
set(CMAKE_PREFIX_PATH "./libtorch")
find_package(Torch REQUIRED)

# OpenCV (vcpkg)
set(OpenCV_DIR "D:/codeAI/vcpkg/installed/x64-windows/share/opencv4")
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(mnist mnist.cpp)
target_link_libraries(mnist ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(mnist PUBLIC ${TORCH_INCLUDE_DIRS})

# 自动复制所有运行时 DLL 到 exe 目录（MSVC）
if(MSVC)
    add_custom_command(TARGET mnist POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_RUNTIME_DLLS:mnist>"
            "$<TARGET_FILE_DIR:mnist>"
        COMMAND_EXPAND_LISTS
    )
endif()