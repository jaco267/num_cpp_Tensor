# CMakeLists.txt  #sudo update-alternatives --config c++ #change from g++ to clang
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project("main")

find_package(Boost COMPONENTS system thread iostreams)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


add_subdirectory(numcpp_2D)
# Collect all .cpp files in the project directory (not include sub dir)
file(GLOB SOURCES "*.cpp")

# Add the executable with all collected source files
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(
    ${PROJECT_NAME} 
    numcpp_2D
)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)