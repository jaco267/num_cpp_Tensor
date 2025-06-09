file(GLOB_RECURSE SRCS *.cpp)

add_library(numcpp_2D ${SRCS})

target_include_directories(
    numcpp_2D PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# target_link_libraries(
#     numcpp_2D
#     utils
# )