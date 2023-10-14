cmake_minimum_required(VERSION 3.10)
project(T2)

set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} main.cpp)

find_package(PythonLibs REQUIRED)
set(PYTHON_INCLUDE_DIRS /home/guanhuai/anaconda3/envs/forcppEnvs/include/python3.10)
set(PYTHON_LIBRARIES /home/guanhuai/anaconda3/envs/forcppEnvs/lib/libpython3.10.so)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(/home/guanhuai/anaconda3/envs/forcppEnvs/lib/python3.10/site-packages/numpy/core/include)
message(STATUS "Python include dirs are: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Python library is: ${PYTHON_LIBRARIES}")

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${PYTHON_LIBRARIES})