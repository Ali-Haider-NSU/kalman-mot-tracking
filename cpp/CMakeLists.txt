cmake_minimum_required(VERSION 3.16)
project(kf_demo CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)      # sudo apt install libeigen3-dev
add_executable(kf_demo main.cpp)
target_link_libraries(kf_demo Eigen3::Eigen)
