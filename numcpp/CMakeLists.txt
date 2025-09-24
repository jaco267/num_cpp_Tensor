file(GLOB_RECURSE SRCS "*.cpp" "*.cu")

add_library(numcpp ${SRCS})



target_include_directories(
    numcpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# target_link_libraries(
#     numcpp
#     utils
# )