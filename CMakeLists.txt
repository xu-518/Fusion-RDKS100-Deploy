
cmake_minimum_required(VERSION 3.10)
project(FUSION_INFER)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(
    ./include
    /usr/include/hobot
)

link_directories(/usr/hobot/lib)

add_executable(fusion_infer main.cpp)

target_link_libraries(fusion_infer 
    ${OpenCV_LIBS}
    dnn
    hbucp
    pthread
    dl
)