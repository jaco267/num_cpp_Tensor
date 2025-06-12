file(GLOB_RECURSE SRCS *.cpp)

add_library(examples ${SRCS})

target_include_directories(
    examples PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    examples
    numcpp
)