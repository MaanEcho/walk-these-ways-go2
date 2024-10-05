project(go2_states)
cmake_minimum_required(VERSION 3.5)
SET(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include/ddscxx /usr/local/include/iceoryx/v2.0.2)
link_libraries(unitree_sdk2 ddsc ddscxx rt pthread)

add_executable(go2_states ./go2_states.cpp)
target_link_libraries(go2_states unitree_sdk2 ddsc ddscxx rt pthread)

# add_executable(go2_low_level ./go2_low_level.cpp)
# target_link_libraries(go2_low_level unitree_sdk2 ddsc ddscxx rt pthread)