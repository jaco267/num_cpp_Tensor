# CMakeLists.txt  #sudo update-alternatives --config c++ #change from g++ to clang
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project("main")

find_package(Boost COMPONENTS system thread iostreams)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wsign-compare")


add_subdirectory(numcpp)
add_subdirectory(examples)
# Collect all .cpp files in the project directory (not include sub dir)
file(GLOB SOURCES "*.cpp")

# Add the executable with all collected source files
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(
    ${PROJECT_NAME} 
    numcpp
    examples
)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)