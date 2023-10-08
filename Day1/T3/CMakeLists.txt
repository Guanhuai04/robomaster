cmake_minimum_required(VERSION 3.10)
project(T1)


set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(T3 main.cpp)
target_link_libraries(  T3 ${OpenCV_LIBS}  )